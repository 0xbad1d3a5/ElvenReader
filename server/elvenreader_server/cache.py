"""On-disk cache of synthesized sentences.

The key covers everything that changes the sound: engine and its options,
voice fingerprint, language, delivery, seed and the *final* text after
pronunciation corrections. Editing a pronunciation therefore changes the key
and stale clips are never reused. Clips are stored untrimmed-of-pauses at
natural speed; tempo and sentence pauses are applied when serving.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from pathlib import Path

_MAGIC = b"ARPC"


def cache_key(**parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AudioCache:
    def __init__(self, root: Path, max_bytes: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._size: int | None = None
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pcm"

    def get(self, key: str) -> tuple[int, bytes] | None:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except OSError:
            self.misses += 1
            return None
        if len(data) < 8 or data[:4] != _MAGIC:
            self.misses += 1
            return None
        self.hits += 1
        try:
            os.utime(path)  # LRU by modification time
        except OSError:
            pass
        (rate,) = struct.unpack("<I", data[4:8])
        return rate, data[8:]

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def peek(self, key: str) -> tuple[int, bytes] | None:
        """Like get, without counting a hit or refreshing the clip's place in the LRU (book file syncs)."""
        try:
            data = self._path(key).read_bytes()
        except OSError:
            return None
        if len(data) < 8 or data[:4] != _MAGIC:
            return None
        (rate,) = struct.unpack("<I", data[4:8])
        return rate, data[8:]

    def put(self, key: str, sample_rate: int, pcm: bytes) -> None:
        if not pcm or self.max_bytes <= 0:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(_MAGIC + struct.pack("<I", sample_rate) + pcm)
        tmp.replace(path)
        with self._lock:
            if self._size is None:
                self._size = self._scan()
            else:
                self._size += len(pcm) + 8
            if self._size > self.max_bytes:
                self._prune(keep=path)

    def discard(self, key: str) -> None:
        """Forget one clip (a take that a later, stricter check would have made again)."""
        path = self._path(key)
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            return
        with self._lock:
            if self._size is not None:
                self._size -= size

    def _scan(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*/*.pcm"))

    def _prune(self, keep: Path | None = None) -> None:
        files = sorted(self.root.glob("*/*.pcm"), key=lambda p: p.stat().st_mtime)
        size = sum(p.stat().st_size for p in files)
        target = int(self.max_bytes * 0.8)
        for p in files:
            if size <= target:
                break
            if p == keep:
                continue
            try:
                s = p.stat().st_size
                p.unlink()
                size -= s
            except OSError:
                pass
        self._size = size

    def stats(self) -> dict:
        with self._lock:
            if self._size is None:
                self._size = self._scan()
            return {"bytes": self._size, "max_bytes": self.max_bytes, "hits": self.hits, "misses": self.misses}
