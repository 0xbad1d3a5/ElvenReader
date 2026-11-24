"""PCM16 helpers: silence trimming, time-stretching and float conversion.

TTS models pad clips with silence at each end. The reader plays one sentence
after another, so that padding would add to every gap between sentences. Pauses
*inside* the audio are kept untouched.
"""
from __future__ import annotations

import array
import sys

import numpy as np


def _samples(data: bytes) -> array.array:
    a = array.array("h")
    a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    return a


class SilenceTrimmer:
    def __init__(
        self,
        sample_rate: int,
        *,
        threshold: int = 150,
        keep_lead_ms: int = 20,
        keep_trail_ms: int = 60,
    ):
        self.threshold = threshold
        self.lead_keep = int(sample_rate * keep_lead_ms / 1000) * 2
        self.trail_keep = int(sample_rate * keep_trail_ms / 1000) * 2
        self.started = False
        self._carry = b""  # odd byte split across chunks
        self._lead = bytearray()  # silence seen before the first sound (rolling)
        self._pending = bytearray()  # silence after the most recent sound

    def _loud_bounds(self, data: bytes) -> tuple[int, int] | None:
        """Sample indexes of the first and last sample above the threshold."""
        thr = self.threshold
        samples = _samples(data)
        first = next((i for i, s in enumerate(samples) if s > thr or s < -thr), None)
        if first is None:
            return None
        last = next(i for i in range(len(samples) - 1, -1, -1) if samples[i] > thr or samples[i] < -thr)
        return first, last

    def feed(self, data: bytes) -> bytes:
        data = self._carry + data
        if len(data) % 2:
            self._carry, data = data[-1:], data[:-1]
        else:
            self._carry = b""
        if not data:
            return b""

        bounds = self._loud_bounds(data)
        if bounds is None:
            if self.started:
                self._pending += data
            else:
                self._lead += data
                del self._lead[: max(0, len(self._lead) - self.lead_keep)]
            return b""

        first, last = bounds
        out = bytearray()
        if not self.started:
            self.started = True
            lead = self._lead + data[: first * 2]
            out += lead[len(lead) - min(len(lead), self.lead_keep):]
            self._lead.clear()
        else:
            out += self._pending
            out += data[: first * 2]
        out += data[first * 2 : (last + 1) * 2]
        self._pending = bytearray(data[(last + 1) * 2 :])
        return bytes(out)

    def flush(self) -> bytes:
        """End of stream: emit a short natural tail of the trailing silence."""
        tail = bytes(self._pending[: self.trail_keep])
        self._pending.clear()
        return tail


def trim_silence(pcm: bytes, sample_rate: int) -> bytes:
    trimmer = SilenceTrimmer(sample_rate)
    return trimmer.feed(pcm) + trimmer.flush()


def noise_gap_db(pcm: bytes, sample_rate: int) -> float | None:
    """How far the pauses between words sit below the speech, in dB (speech RMS over the RMS of 40 ms windows at
    least 80 ms from any speech). A designed voice is cloned with everything in it, background included; VoxCPM2
    designs ranged from 6 to 67 dB with the seed alone, and those under ~20 dB hum and hiss audibly in every line
    read with them. None when the clip has no pauses to measure."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    win = max(1, int(sample_rate * 0.04))
    n = len(x) // win
    if n < 5:
        return None
    frames = x[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = rms > 0.02
    if not loud.any():
        return None
    near = np.convolve(loud.astype(int), np.ones(5, dtype=int), "same") > 0
    quiet = frames[(~near) & (rms > 1e-6)]
    if len(quiet) < 2:
        return None
    speech = np.sqrt((frames[loud] ** 2).mean())
    return float(20 * np.log10(speech / (np.sqrt((quiet ** 2).mean()) + 1e-9)))


def tighten(pcm: bytes, sample_rate: int, *, threshold: float = 120.0, window_ms: int = 20, max_gap_ms: int = 1000,
            keep_lead_ms: int = 20, keep_trail_ms: int = 60, blip_ms: int = 60, blip_gap_ms: int = 300) -> bytes:
    """A finished utterance without dead air: silence (20 ms windows quieter than `threshold` RMS) at either end is
    cut to a short natural lead and tail, gaps inside longer than `max_gap_ms` are shortened to it, and a click
    after the speech (sound shorter than `blip_ms` after `blip_gap_ms` of silence) is dropped with the silence
    before it. Models such as VoxCPM2 can trail off for seconds after "..." and end on a click, which kept the
    whole silence for the per-sample trimmer. Pauses up to `max_gap_ms` are kept as they are; running it twice
    changes nothing."""
    if not pcm or sample_rate <= 0:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    win = max(1, int(sample_rate * window_ms / 1000))
    n = len(x) // win
    if n == 0:
        return pcm
    frames = x[: n * win].astype(np.float32).reshape(n, win)
    loud = np.sqrt((frames ** 2).mean(axis=1)) >= threshold
    if not loud.any():
        return b""
    # Runs of loud windows: [start, end) in windows.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], loud.view(np.int8), [0]))))
    runs = list(zip(edges[::2], edges[1::2]))
    blip, quiet_before = max(1, blip_ms // window_ms), max(1, blip_gap_ms // window_ms)
    while len(runs) > 1 and runs[-1][1] - runs[-1][0] < blip and runs[-1][0] - runs[-2][1] >= quiet_before:
        runs.pop()  # a click after the speech
    while len(runs) > 1 and runs[0][1] - runs[0][0] < blip and runs[1][0] - runs[0][1] >= quiet_before:
        runs.pop(0)  # a click before it
    max_gap = max(1, max_gap_ms // window_ms)
    start = max(0, runs[0][0] * win - int(sample_rate * keep_lead_ms / 1000))
    end = min(len(x), runs[-1][1] * win + int(sample_rate * keep_trail_ms / 1000))
    pieces, pos = [], start
    for (_, a_end), (b_start, _) in zip(runs, runs[1:]):
        if b_start - a_end > max_gap:
            half = max_gap * win // 2
            pieces.append(x[pos:a_end * win + half])
            pos = b_start * win - (max_gap * win - half)
    pieces.append(x[pos:end])
    return np.concatenate(pieces).astype("<i2").tobytes()


def speech_level_db(x: np.ndarray, sample_rate: int) -> float | None:
    """How loud the speech in a clip is (dBFS): the mean power of the 20 ms windows within 30 dB of the loudest,
    pauses left out. None for silence."""
    win = max(1, int(sample_rate * 0.02))
    n = len(x) // win
    if n == 0:
        return None
    power = (x[: n * win].astype(np.float32).reshape(n, win) / 32768) ** 2
    db = 10 * np.log10(power.mean(axis=1) + 1e-12)
    speech = db >= max(db.max() - 30, -55)
    if not speech.any():
        return None
    return float(10 * np.log10(power[speech].mean() + 1e-12))


TARGET_SPEECH_DB = -19.0  # the narration's typical level; VoxCPM2's clips ranged from -31 to -16
MAX_LEVEL_GAIN_DB = 10.0
PEAK_LIMIT = 0.95


def level(pcm: bytes, sample_rate: int, target_db: float = TARGET_SPEECH_DB) -> bytes:
    """The clip with its speech brought to `target_db`. Each take comes out of a model at its own level: in
    Moon Cradle II Vol. 20 short narration lines were quieter (median -20.9 dB under 60 characters, -18.7 over
    150), and some 6-11 dB under the lines around them ("One year, three months, and sixteen days, in fact.":
    -25.3 between -20s). At most MAX_LEVEL_GAIN_DB either way, and never past PEAK_LIMIT; within 0.5 dB it is
    left as it is, so running it twice changes nothing."""
    if not pcm or sample_rate <= 0:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    now = speech_level_db(x, sample_rate)
    if now is None:
        return pcm
    gain = max(-MAX_LEVEL_GAIN_DB, min(MAX_LEVEL_GAIN_DB, target_db - now))
    peak = int(np.abs(x.astype(np.int32)).max()) / 32768
    if peak > 0:
        gain = min(gain, 20 * np.log10(PEAK_LIMIT / peak))
    if abs(gain) < 0.5:
        return pcm
    y = x.astype(np.float32) * (10 ** (gain / 20))
    return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


def _k_weighting_power(n: int, sample_rate: int) -> np.ndarray:
    """|H(f)|^2 of the ITU-R BS.1770 K-weighting (the ear's loudness curve: a high shelf and a low cut) at the
    rfft bins of an n-sample signal."""
    def response(b, a, w):
        z = np.exp(-1j * w)
        return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)

    w = 2 * np.pi * np.fft.rfftfreq(n, 1 / sample_rate) / sample_rate
    gain_db, f0, q = 3.999843853973347, 1681.974450955533, 0.7071752369554196
    big_a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sample_rate
    alpha, c = np.sin(w0) / (2 * q), np.cos(w0)
    s = 2 * np.sqrt(big_a) * alpha
    shelf = response([big_a * ((big_a + 1) + (big_a - 1) * c + s), -2 * big_a * ((big_a - 1) + (big_a + 1) * c),
                      big_a * ((big_a + 1) + (big_a - 1) * c - s)],
                     [(big_a + 1) - (big_a - 1) * c + s, 2 * ((big_a - 1) - (big_a + 1) * c), (big_a + 1) - (big_a - 1) * c - s], w)
    f0, q = 38.13547087602444, 0.5003270373238773
    w0 = 2 * np.pi * f0 / sample_rate
    alpha, c = np.sin(w0) / (2 * q), np.cos(w0)
    low_cut = response([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + alpha, -2 * c, 1 - alpha], w)
    return np.abs(shelf * low_cut) ** 2


TARGET_LOUDNESS_LUFS = -20.5  # what clips levelled to TARGET_SPEECH_DB measured on average (BS.1770 loudness)
EVEN_WINDOW_S = 1.0  # loudness is measured over this much speech...
EVEN_HOP_S = 0.1
EVEN_SMOOTH_S = 1.2  # ...and the gain changes this slowly, so words don't pump
EVEN_MAX_GAIN_DB = 10.0


def even_out(pcm: bytes, sample_rate: int, target_lufs: float = TARGET_LOUDNESS_LUFS) -> bytes:
    """The clip at one loudness throughout: a slow gain rider. A model's take drifts inside a passage (narration in
    Moon Cradle II Vol. 20 swelled or faded by 4 LU in a typical 8 s+ clip, up to 10), which levelling the clip as a
    whole (level()) can't help. Loudness is measured as the ear hears it (BS.1770 K-weighting) over 1 s of audio
    every 0.1 s; each stretch is brought towards `target_lufs` (at most EVEN_MAX_GAIN_DB either way), pauses take
    the gain around them (never boosted on their own), the gain glides over EVEN_SMOOTH_S, and peaks are held
    under PEAK_LIMIT. A clip already even is left nearly as it is, so running it twice changes little."""
    if not pcm or sample_rate <= 0:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64) / 32768
    n = len(x)
    if n < int(sample_rate * 0.05):
        return pcm
    k = np.fft.irfft(np.fft.rfft(x) * np.sqrt(_k_weighting_power(n, sample_rate)), n)
    win = min(n, int(EVEN_WINDOW_S * sample_rate))
    hop = max(1, int(EVEN_HOP_S * sample_rate))
    # Mean square of each 1 s window, from a running sum (a clip of minutes stays cheap).
    csum = np.concatenate(([0.0], np.cumsum(k * k)))
    starts = np.arange(0, max(1, n - win + 1), hop)
    ms = (csum[starts + win] - csum[starts]) / win
    loud = -0.691 + 10 * np.log10(ms + 1e-15)
    speech = loud > max(-50.0, loud.max() - 25)  # pauses and breaths are left out of the measure
    if not speech.any():
        return pcm
    centers = starts + win / 2
    want = np.clip(target_lufs - loud, -EVEN_MAX_GAIN_DB, EVEN_MAX_GAIN_DB)
    gain_db = np.interp(centers, centers[speech], want[speech])  # pauses: the gain of the speech around them
    smooth = max(1, int(round(EVEN_SMOOTH_S / EVEN_HOP_S)))
    if smooth > 1 and len(gain_db) > 1:
        pad = np.pad(gain_db, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
        gain_db = np.convolve(pad, np.ones(smooth) / smooth, mode="valid")
    gain = 10 ** (np.interp(np.arange(n), centers, gain_db) / 20)
    y = x * gain
    # Peak limiter: where a stretch would pass PEAK_LIMIT, it is turned down (from 5 ms before to 50 ms after).
    block = max(1, int(sample_rate * 0.005))
    nb = -(-n // block)
    peaks = np.abs(np.pad(y, (0, nb * block - n))).reshape(nb, block).max(axis=1)
    cut = np.minimum(1.0, PEAK_LIMIT / np.maximum(peaks, 1e-9))
    if (cut < 1).any():
        hold = cut.copy()
        padded = np.pad(cut, (10, 1), constant_values=1.0)
        for shift in range(-1, 11):  # each block's cut also covers the block before and ten after
            hold = np.minimum(hold, padded[10 - shift: 10 - shift + nb])
        y = y * np.repeat(hold, block)[:n]
    return np.clip(np.round(y * 32768), -32768, 32767).astype("<i2").tobytes()


def normalize(pcm: bytes, sample_rate: int, delivery: str | None) -> bytes:
    """A clip at the common loudness as it is kept and served. A line without delivery tags is evened out
    throughout (even_out); one with them is levelled as a whole, to its delivery's place (a whisper sits under the
    rest, a shout over it: delivery.level_offset_db), so the dynamics it was asked for stay."""
    from .delivery import level_offset_db

    if not (delivery or "").strip():
        return even_out(pcm, sample_rate)
    return level(pcm, sample_rate, TARGET_SPEECH_DB + level_offset_db(delivery))


def time_stretch(pcm: bytes, sample_rate: int, speed: float) -> bytes:
    """Change tempo without changing pitch (WSOLA).

    Most quality models have no speed control, and cached clips are stored at
    their natural speed, so tempo is applied afterwards.
    """
    if abs(speed - 1.0) < 0.01 or not pcm:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32)
    frame = max(64, int(sample_rate * 0.030))
    hop_out = frame // 2
    hop_in = hop_out * speed
    tol = int(sample_rate * 0.012)
    if len(x) < frame * 2:
        return pcm
    window = np.hanning(frame).astype(np.float32)
    n_out = int(len(x) / speed) + 2 * frame
    out = np.zeros(n_out, dtype=np.float32)
    norm = np.zeros(n_out, dtype=np.float32)

    pos_in = 0.0
    pos_out = 0
    prev = 0
    first = True
    while pos_out + frame <= n_out:
        center = int(pos_in)
        if first:
            best = 0
            first = False
        else:
            natural = x[prev + hop_out : prev + hop_out + frame]
            lo = max(0, center - tol)
            hi = min(len(x) - frame, center + tol)
            if hi < lo or len(natural) < frame:
                break
            corr = np.correlate(x[lo : hi + frame], natural, mode="valid")
            best = lo + int(np.argmax(corr))
        if best + frame > len(x):
            break
        out[pos_out : pos_out + frame] += x[best : best + frame] * window
        norm[pos_out : pos_out + frame] += window
        prev = best
        pos_out += hop_out
        pos_in += hop_in
    end = pos_out + hop_out
    out, norm = out[:end], norm[:end]
    np.divide(out, norm, out=out, where=norm > 0.05)
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


def pcm_duration_s(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / 2 / sample_rate if sample_rate else 0.0


def float_to_pcm16(samples) -> bytes:
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap PCM16 mono in a WAV header."""
    import struct

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(pcm),
    )
    return header + pcm


# ---------------------------------------------------------------- compressed delivery

# What the website asks for: speech in Opus at 64 kbit/s is indistinguishable from the 24 kHz PCM it comes
# from for listening, at about a sixth of the size; MP3 is the fallback for browsers without Ogg Opus.
FORMATS = {
    "opus": (["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"], "audio/ogg; codecs=opus"),
    "mp3": (["-c:a", "libmp3lame", "-b:a", "96k", "-f", "mp3"], "audio/mpeg"),
}


class EncodeError(Exception):
    pass


async def encode(pcm: bytes, rate: int, fmt: str) -> tuple[bytes, str]:
    """Mono PCM16LE -> (encoded bytes, content type) with ffmpeg. Raises EncodeError when it can't."""
    import asyncio
    import shutil

    if fmt not in FORMATS:
        raise EncodeError(f"Unknown audio format {fmt!r}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise EncodeError("ffmpeg is not installed")
    codec, content_type = FORMATS[fmt]
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-v", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0", *codec, "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        creationflags=0x08000000 if sys.platform == "win32" else 0)
    out, err = await proc.communicate(pcm)
    if proc.returncode != 0 or not out:
        raise EncodeError(f"ffmpeg failed: {err.decode('utf-8', 'replace')[-300:]}")
    return out, content_type