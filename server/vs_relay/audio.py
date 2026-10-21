"""Streaming trim of leading/trailing silence in PCM16LE mono audio.

VoiceStudio pads every clip with ~150 ms of silence at each end. A reader such
as ReadEra requests one sentence at a time and only after the previous one has
finished playing, so that padding adds directly to every gap between
sentences. Pauses *inside* the audio are kept untouched.
"""
from __future__ import annotations

import array
import sys


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
