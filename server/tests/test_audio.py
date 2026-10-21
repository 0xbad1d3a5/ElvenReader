import array
import random

import pytest

from vs_relay.audio import SilenceTrimmer

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
