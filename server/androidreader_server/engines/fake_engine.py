"""Test engine: a short tone per word, no GPU or model needed."""
from __future__ import annotations

import math
import struct
import time

from .base import BaseEngine


class Engine(BaseEngine):
    sample_rate = 16000

    def load(self) -> None:
        if self.options.get("fail_load"):
            raise RuntimeError("fake load failure")
        time.sleep(float(self.options.get("load_delay", 0)))

    def _tone(self, n_samples: int, freq: float) -> bytes:
        return b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / self.sample_rate)))
            for i in range(n_samples)
        )

    def synthesize(self, req: dict, cancelled):
        text = req["text"]
        if "explode" in text:
            raise RuntimeError("engine exploded")
        words = text.split()
        delay = float(self.options.get("chunk_delay", 0))
        if "slow" in text:
            delay = max(delay, 0.2)
        freq = 220.0 + 20 * len(req.get("delivery") or "")
        for _ in words:
            if cancelled():
                return
            if delay:
                time.sleep(delay)
            yield self._tone(self.sample_rate // 20, freq)  # 50 ms per word

    def design_voice(self, description: str, text: str, seed):
        yield self._tone(self.sample_rate, 330.0)
