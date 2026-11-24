import array
import random

import pytest

import numpy as np

from elvenreader_server.audio import TARGET_SPEECH_DB, SilenceTrimmer, level, speech_level_db, time_stretch
from elvenreader_server.delivery import level_offset_db

SR = 1000  # 1 sample per ms keeps the arithmetic readable


def pcm(*segments: tuple[int, int]) -> bytes:
    """Build PCM16LE from (value, n_samples) runs."""
    a = array.array("h")
    for value, n in segments:
        a.extend([value] * n)
    return a.tobytes()


def run(trimmer: SilenceTrimmer, data: bytes, chunks: list[int] | None = None) -> bytes:
    out = bytearray()
    if chunks is None:
        out += trimmer.feed(data)
    else:
        pos = 0
        for size in chunks:
            out += trimmer.feed(data[pos : pos + size])
            pos += size
        out += trimmer.feed(data[pos:])
    out += trimmer.flush()
    return bytes(out)


def make() -> SilenceTrimmer:
    return SilenceTrimmer(SR, threshold=400, keep_lead_ms=30, keep_trail_ms=80)


def test_trims_lead_and_tail_but_keeps_internal_pause():
    data = pcm((0, 150), (5000, 100), (10, 300), (-5000, 100), (0, 190))
    out = run(make(), data)
    assert out == pcm((0, 30), (5000, 100), (10, 300), (-5000, 100), (0, 80))


def test_short_padding_is_left_alone():
    data = pcm((0, 10), (5000, 50), (0, 40))
    assert run(make(), data) == data


def test_all_silence_yields_nothing():
    assert run(make(), pcm((0, 500))) == b""


def test_tighten_drops_dead_air_and_a_closing_click_but_keeps_natural_pauses():
    from elvenreader_server.audio import tighten

    rate = 1000  # 20-sample windows
    speech = pcm((3000, 400))
    # VoxCPM2 once trailed off for 6 s after "..." and ended on a click: all of it was kept.
    out = tighten(speech + pcm((0, 600)) + speech + pcm((0, 6000), (5000, 20), (0, 300)), rate)
    assert out == speech + pcm((0, 600)) + speech + pcm((0, 60))
    # A 3 s gap inside is shortened to 1 s; a short click before the speech goes too.
    out = tighten(pcm((4000, 20), (0, 500)) + speech + pcm((0, 3000)) + speech, rate)
    assert out == pcm((0, 20)) + speech + pcm((0, 1000)) + speech
    assert tighten(out, rate) == out  # running it again changes nothing
    assert tighten(pcm((0, 2000)), rate) == b""


@pytest.mark.parametrize("seed", range(20))
def test_arbitrary_chunking_matches_single_feed(seed):
    rng = random.Random(seed)
    data = pcm((0, 150), (3000, 37), (0, 200), (-3000, 61), (100, 5), (2000, 13), (0, 180))
    chunks = [rng.randint(1, 120) for _ in range(rng.randint(1, 30))]  # includes odd sizes
    assert run(make(), data, chunks) == run(make(), data)


def test_output_is_whole_samples():
    t = make()
    first = t.feed(pcm((5000, 3)) + b"\x01")  # odd byte held back
    assert len(first) % 2 == 0
    assert len(t.feed(b"\x00")) % 2 == 0



@pytest.mark.parametrize("speed", [0.75, 1.25, 1.5, 2.0])
def test_time_stretch_changes_duration_not_pitch(speed):
    import numpy as np

    rate = 16000
    t = np.arange(rate * 2) / rate
    tone = (8000 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes()
    out = np.frombuffer(time_stretch(tone, rate, speed), dtype="<i2").astype(np.float32)
    assert abs(len(out) / (rate * 2) - 1 / speed) < 0.05
    spectrum = np.abs(np.fft.rfft(out[rate // 4 : rate // 4 + 8192]))
    peak_hz = np.argmax(spectrum) * rate / 8192
    assert abs(peak_hz - 220) < 6


def test_time_stretch_identity_and_tiny_input():
    assert time_stretch(b"\x01\x00" * 10, 16000, 1.5) == b"\x01\x00" * 10
    assert time_stretch(b"abcd", 16000, 1.0) == b"abcd"


def _clip(rate: int, pause_level: float) -> bytes:
    import numpy as np

    t = np.arange(int(rate * 0.5)) / rate
    word = (0.06 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)  # about -27 dBFS, like real speech
    hum = (pause_level * np.sin(2 * np.pi * 170 * np.arange(int(rate * 0.6)) / rate)).astype(np.float32)
    x = np.concatenate([word, hum, word, hum, word])
    return (x * 32767).astype("<i2").tobytes()


def test_noise_gap_tells_a_humming_voice_from_a_clean_one():
    from elvenreader_server.audio import noise_gap_db

    rate = 16000
    assert noise_gap_db(_clip(rate, 0.011), rate) < 20  # a hum at -42 dBFS, as in VoxCPM2's buzzy designs
    assert noise_gap_db(_clip(rate, 0.0005), rate) > 40
    assert noise_gap_db(b"", rate) is None


async def test_a_humming_design_is_made_again_with_another_seed(monkeypatch):
    from elvenreader_server import app as app_module
    from elvenreader_server.engines.manager import AudioChunk, StreamStart

    seeds = []

    class Manager:
        def design(self, engine, description, text, seed):
            seeds.append(seed)

            async def stream():
                yield StreamStart(sample_rate=16000, engine=engine)
                yield AudioChunk(pcm=_clip(16000, 0.011 if len(seeds) == 1 else 0.0005))
            return stream()

    app = {app_module.ENGINES_KEY: Manager()}
    rate, pcm, seed = await app_module.design_clip(app, "voxcpm", "a calm man", "Hello.", 42)
    assert len(seeds) == 2 and seed == seeds[1] != 42 and rate == 16000
    from elvenreader_server.audio import noise_gap_db

    assert noise_gap_db(pcm, rate) > 40


def _speech(amplitude: float, rate: int = 24000) -> bytes:
    """Two 'words' of a 220 Hz tone with a pause between them."""
    t = np.arange(int(rate * 0.5)) / rate
    word = amplitude * np.sin(2 * np.pi * 220 * t)
    x = np.concatenate([word, np.zeros(int(rate * 0.3)), word * 0.8])
    return (x * 32767).astype("<i2").tobytes()


def _db(pcm: bytes) -> float:
    return speech_level_db(np.frombuffer(pcm, dtype="<i2"), 24000)


def test_level_brings_quiet_and_loud_clips_to_the_target_and_is_stable():
    quiet, loud = _speech(0.07), _speech(0.5)  # 8 dB under, 9 dB over
    a, b = level(quiet, 24000), level(loud, 24000)
    assert abs(_db(a) - TARGET_SPEECH_DB) < 0.6 and abs(_db(b) - TARGET_SPEECH_DB) < 0.6
    assert level(a, 24000) == a  # running it twice changes nothing
    assert len(a) == len(quiet)


def test_level_never_clips_or_boosts_silence():
    peaky = bytearray(_speech(0.02))
    peaky[1000:1002] = (30000).to_bytes(2, "little", signed=True)  # one loud sample
    out = np.frombuffer(level(bytes(peaky), 24000), dtype="<i2")
    assert np.abs(out.astype(np.int32)).max() <= 0.95 * 32768 + 1
    assert level(bytes(4800), 24000) == bytes(4800)
    very_quiet = _speech(0.003)
    assert _db(level(very_quiet, 24000)) < TARGET_SPEECH_DB - 5  # at most 10 dB of gain


def test_quiet_and_loud_deliveries_keep_their_offset():
    assert level_offset_db("very afraid, whisper") == -6.0
    assert level_offset_db("angry, shout") == 3.0
    assert level_offset_db("thought, loud") == -2.0  # the quieter one wins
    assert level_offset_db(None) == 0.0 and level_offset_db("sad") == 0.0


def _speechy(seconds, db, rate=24000, seed=0):
    """Noise shaped like speech (syllables of 200 ms with short gaps) at roughly `db` dBFS."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    env = (np.sin(np.arange(n) / rate * np.pi * 5) > -0.6).astype(float)
    x = rng.normal(0, 1, n) * env
    x *= 10 ** (db / 20) / np.sqrt(np.mean(x[env > 0] ** 2))
    return x


def _pcm(x):
    return np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes()


def _loudness_by_second(pcm, rate=24000):
    from elvenreader_server.audio import _k_weighting_power

    x = np.frombuffer(pcm, dtype="<i2").astype(float) / 32768
    k = np.fft.irfft(np.fft.rfft(x) * np.sqrt(_k_weighting_power(len(x), rate)), len(x))
    return [-0.691 + 10 * np.log10(np.mean(k[i:i + rate] ** 2) + 1e-15) for i in range(0, len(k) - rate + 1, rate)]


def test_a_passage_that_fades_is_evened_out_and_pauses_are_not_boosted():
    from elvenreader_server.audio import TARGET_LOUDNESS_LUFS, even_out

    rate = 24000
    x = np.concatenate([_speechy(4, -14, rate, 1), np.zeros(rate), _speechy(4, -28, rate, 2)])  # loud, pause, quiet
    before = _loudness_by_second(_pcm(x))
    out = even_out(_pcm(x), rate)
    after = _loudness_by_second(out)
    assert max(before) - min(b for b in before if b > -60) > 12
    speech = [a for i, a in enumerate(after) if i not in (3, 4, 5)]  # around the pause, the gain is gliding
    assert max(speech) - min(speech) < 3 and abs(np.mean(speech) - TARGET_LOUDNESS_LUFS) < 1.5
    y = np.frombuffer(out, dtype="<i2")
    assert np.abs(y[int(4.3 * rate):int(4.7 * rate)]).max() == 0  # the pause stays silent
    assert np.abs(y.astype(int)).max() <= 0.95 * 32768 + 1  # no clipping
    # Even already: a second pass changes it little.
    again = _loudness_by_second(even_out(out, rate))
    assert max(abs(a - b) for a, b in zip(again, after) if b > -60) < 1.0


def test_tagged_lines_keep_their_place_and_untagged_ones_are_evened():
    from elvenreader_server.audio import TARGET_LOUDNESS_LUFS, normalize

    rate = 24000
    x = _pcm(_speechy(3, -30, rate, 3))
    plain = _loudness_by_second(normalize(x, rate, None))
    whisper = normalize(x, rate, "whisper")
    assert abs(np.mean(plain) - TARGET_LOUDNESS_LUFS) < 1.5
    w = speech_level_db(np.frombuffer(whisper, dtype="<i2"), rate)
    assert abs(w - (TARGET_SPEECH_DB + level_offset_db("whisper"))) < 1.0

