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
