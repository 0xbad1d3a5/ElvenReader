"""Engine worker process: ``python -m androidreader_server.engines.worker --engine ID``.

Loads one model and synthesizes requests sent by the server over stdin,
streaming results back over a private copy of stdout (see protocol.py).
Libraries print freely to fd 1 (tqdm, native logs), so fd 1 is pointed at
stderr and frames go down a duplicated descriptor that only we write to.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import queue
import sys
import threading
import time
import traceback

from . import protocol


def _prepare_environment(models_dir: str) -> None:
    if models_dir:
        os.environ["ANDROIDREADER_MODELS"] = models_dir
        os.environ.setdefault("HF_HOME", os.path.join(models_dir, "hf"))
        os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(models_dir, "modelscope"))
        os.environ.setdefault("TORCH_HOME", os.path.join(models_dir, "torch"))
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(models_dir, "torch", "inductor"))
        os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(models_dir, "torch", "triton"))
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


class Worker:
    def __init__(self, engine_module: str, options: dict, out):
        self.engine_module = engine_module
        self.options = options
        self.out = out
        self.out_lock = threading.Lock()
        self.engine = None
        self.cancelled: set[str] = set()
        self.inbox: queue.Queue = queue.Queue()

    def send(self, obj: dict) -> None:
        with self.out_lock:
            self.out.write(protocol.encode_json(obj))
            self.out.flush()

    def send_audio(self, rid: str, pcm: bytes) -> None:
        with self.out_lock:
            self.out.write(protocol.encode_audio(rid, pcm))
            self.out.flush()

    def reader(self, stdin) -> None:
        """Reads requests; cancels are applied immediately, others are queued."""
        while True:
            try:
                body = protocol.read_frame_sync(stdin)
            except Exception:  # noqa: BLE001
                body = None
            if body is None:
                self.inbox.put({"op": "shutdown"})
                return
            _, msg = protocol.decode(body)
            if msg.get("op") == "cancel":
                self.cancelled.add(str(msg.get("id")))
            else:
                self.inbox.put(msg)

    def load(self) -> None:
        if self.engine is not None:
            return
        t0 = time.perf_counter()
        module = importlib.import_module(self.engine_module)
        self.engine = module.Engine(self.options)
        self.engine.load()
        self.send({
            "type": "loaded",
            "sample_rate": self.engine.sample_rate,
            "load_s": round(time.perf_counter() - t0, 2),
            "vram_mb": self.engine.vram_mb(),
        })

    def _stream(self, rid: str, chunks) -> None:
        t0 = time.perf_counter()
        samples = 0
        started = False
        for chunk in chunks:
            if not started:
                started = True
                self.send({"type": "start", "id": rid, "sample_rate": self.engine.sample_rate})
            if rid in self.cancelled:
                self.cancelled.discard(rid)
                self.send({"type": "cancelled", "id": rid})
                return
            pcm = chunk if isinstance(chunk, (bytes, bytearray)) else _to_pcm16(chunk)
            if pcm:
                samples += len(pcm) // 2
                self.send_audio(rid, bytes(pcm))
        if not started:
            self.send({"type": "start", "id": rid, "sample_rate": self.engine.sample_rate})
        rate = self.engine.sample_rate
        gen_s = time.perf_counter() - t0
        self.cancelled.discard(rid)
        self.send({
            "type": "done",
            "id": rid,
            "gen_s": round(gen_s, 3),
            "audio_s": round(samples / rate, 3),
            "vram_mb": self.engine.vram_mb(),
        })

    def handle(self, msg: dict) -> bool:
        op = msg.get("op")
        rid = str(msg.get("id") or "")
        try:
            if op == "shutdown":
                return False
            if op == "ping":
                self.send({"type": "pong"})
            elif op == "load":
                self.load()
            elif op == "synth":
                self.load()
                if rid in self.cancelled:
                    self.cancelled.discard(rid)
                    self.send({"type": "cancelled", "id": rid})
                else:
                    self._stream(rid, self.engine.synthesize(msg, lambda: rid in self.cancelled))
            elif op == "design":
                self.load()
                self._stream(rid, self.engine.design_voice(msg.get("description", ""), msg.get("text", ""),
                                                           msg.get("seed")))
            else:
                self.send({"type": "error", "id": rid, "detail": f"unknown op {op!r}"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            self.send({"type": "error", "id": rid, "op": op, "detail": f"{type(exc).__name__}: {exc}"})
        return True


def _to_pcm16(samples) -> bytes:
    import numpy as np

    if hasattr(samples, "detach"):
        samples = samples.detach().to("cpu").float().numpy()
    arr = np.asarray(samples)
    while arr.ndim > 1:
        arr = arr.mean(axis=int(np.argmin(arr.shape))) if arr.dtype != np.int16 else arr.reshape(-1)
    if arr.dtype == np.int16:
        return arr.astype("<i2").tobytes()
    arr = arr.astype(np.float32)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--options", default="{}")
    parser.add_argument("--models-dir", default="")
    args = parser.parse_args(argv)
    _prepare_environment(args.models_dir)

    frame_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(frame_fd, "wb")
    # Same for requests: on Windows, GetFileType() on a pipe blocks while another
    # thread waits in a synchronous read on it, and importing torch probes stdin.
    # Keep the request pipe private and give everyone else NUL as stdin.
    request_fd = os.dup(0)
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.close(null_fd)
    requests = os.fdopen(request_fd, "rb", buffering=0)
    sys.stdin = open(os.devnull, "r")
    worker = Worker(args.module, json.loads(args.options), out)
    threading.Thread(target=worker.reader, args=(requests,), daemon=True).start()
    worker.send({"type": "hello", "pid": os.getpid(), "python": sys.version.split()[0]})
    while worker.handle(worker.inbox.get()):
        pass
    out.flush()
    # The reader thread is still blocked on the request pipe; skip interpreter
    # finalisation (and CUDA teardown) and leave immediately.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
