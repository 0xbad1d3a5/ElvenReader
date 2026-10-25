"""Runs engine workers and serialises access to the GPU.

Only one engine is loaded at a time. Asking for another engine stops the
current worker (freeing its VRAM) and starts the requested one. Requests are
served one at a time in arrival order. When a client goes away mid-sentence,
the worker is told to cancel and the manager waits for it to acknowledge
before serving the next request, so frames never get mixed up.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import protocol
from .registry import EngineSpec

logger = logging.getLogger("androidreader.engines")

SERVER_ROOT = Path(__file__).resolve().parent.parent.parent


class EngineError(Exception):
    def __init__(self, detail: str, status: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class StreamStart:
    sample_rate: int
    engine: str


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes


@dataclass(frozen=True)
class Done:
    gen_s: float
    audio_s: float


@dataclass
class EngineStats:
    requests: int = 0
    gen_s: float = 0.0
    audio_s: float = 0.0
    last_rtf: float | None = None
    load_s: float | None = None
    vram_mb: float | None = None

    def record(self, gen_s: float, audio_s: float, vram_mb: float | None) -> None:
        self.requests += 1
        self.gen_s += gen_s
        self.audio_s += audio_s
        if audio_s > 0:
            self.last_rtf = round(gen_s / audio_s, 3)
        if vram_mb:
            self.vram_mb = max(self.vram_mb or 0, vram_mb)

    def to_json(self) -> dict:
        return {
            "requests": self.requests,
            "rtf": round(self.gen_s / self.audio_s, 3) if self.audio_s else None,
            "last_rtf": self.last_rtf,
            "audio_s": round(self.audio_s, 1),
            "load_s": self.load_s,
            "vram_mb": self.vram_mb,
        }


class _Worker:
    def __init__(self, spec: EngineSpec, proc: asyncio.subprocess.Process):
        self.spec = spec
        self.proc = proc
        self.queues: dict[str, asyncio.Queue] = {}
        self.control: asyncio.Queue = asyncio.Queue()
        self.alive = True
        self.sample_rate = 24000
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self.tasks = [
            asyncio.ensure_future(self._read_stdout()),
            asyncio.ensure_future(self._read_stderr()),
        ]

    async def _read_stdout(self) -> None:
        try:
            while True:
                body = await protocol.read_frame_async(self.proc.stdout)
                if body is None:
                    break
                kind, msg = protocol.decode(body)
                if kind == "audio":
                    rid, pcm = msg
                    q = self.queues.get(rid)
                    if q is not None:
                        q.put_nowait(("audio", pcm))
                    continue
                rid = str(msg.get("id") or "")
                q = self.queues.get(rid) if rid else None
                (q or self.control).put_nowait(("json", msg))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s worker stream broke: %s", self.spec.id, exc)
        finally:
            self.alive = False
            for q in [*self.queues.values(), self.control]:
                q.put_nowait(("exit", None))

    async def _read_stderr(self) -> None:
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self.stderr_tail.append(text)
                logger.debug("[%s] %s", self.spec.id, text)

    async def send(self, obj: dict) -> None:
        if not self.alive or self.proc.stdin is None:
            raise EngineError(f"{self.spec.label} worker is not running", 503)
        try:
            self.proc.stdin.write(protocol.encode_json(obj))
            await self.proc.stdin.drain()
        except (ConnectionError, OSError) as exc:
            self.alive = False
            raise EngineError(f"{self.spec.label} worker died", 503) from exc

    async def expect_control(self, types: set[str], timeout: float) -> dict:
        while True:
            try:
                kind, msg = await asyncio.wait_for(self.control.get(), timeout)
            except asyncio.TimeoutError as exc:
                raise EngineError(f"{self.spec.label} did not respond in {timeout:.0f}s", 504) from exc
            if kind == "exit":
                raise EngineError(self.crash_detail(), 503)
            if msg.get("type") == "error":
                raise EngineError(f"{self.spec.label}: {msg.get('detail')}", 503)
            if msg.get("type") in types:
                return msg

    def crash_detail(self) -> str:
        tail = [l for l in self.stderr_tail if l.strip()][-3:]
        return f"{self.spec.label} worker exited" + (": " + " | ".join(tail) if tail else "")

    async def stop(self) -> None:
        if self.proc.returncode is None:
            try:
                await self.send({"op": "shutdown"})
                await asyncio.wait_for(self.proc.wait(), 10)
            except (EngineError, asyncio.TimeoutError):
                self.proc.kill()
                await self.proc.wait()
        self.alive = False
        for t in self.tasks:
            t.cancel()


class EngineManager:
    def __init__(
        self,
        specs: dict[str, EngineSpec],
        models_dir: Path,
        *,
        options: dict | None = None,
        load_timeout_s: float = 900.0,
        synth_timeout_s: float = 180.0,
    ):
        self.specs = specs
        self.models_dir = Path(models_dir)
        self.options = options or {}
        self.load_timeout_s = load_timeout_s
        self.synth_timeout_s = synth_timeout_s
        self.stats: dict[str, EngineStats] = {k: EngineStats() for k in specs}
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._last_activity = time.monotonic() - 3600
        self._worker: _Worker | None = None
        self.state = "idle"
        self.state_detail = ""
        self.loading_engine: str | None = None

    # -- information ---------------------------------------------------------

    def spec(self, engine_id: str) -> EngineSpec:
        spec = self.specs.get(engine_id)
        if spec is None:
            raise EngineError(f"Unknown engine {engine_id!r}", 400)
        return spec

    @property
    def active(self) -> str | None:
        return self._worker.spec.id if self._worker and self._worker.alive else None

    def status(self) -> dict:
        return {
            "active": self.active,
            "state": self.state,
            "detail": self.state_detail,
            "loading": self.loading_engine,
            "busy": self._lock.locked(),
        }

    def engines(self) -> list[dict]:
        out = []
        for spec in self.specs.values():
            d = spec.public()
            d.update(
                installed=spec.installed(self.models_dir),
                active=spec.id == self.active,
                stats=self.stats[spec.id].to_json(),
            )
            out.append(d)
        return out

    # -- worker lifecycle ----------------------------------------------------

    async def _start(self, spec: EngineSpec) -> _Worker:
        python = spec.python(self.models_dir)
        if not python.exists():
            raise EngineError(
                f"{spec.label} is not installed (missing {python}). Run setup-engines.ps1 {spec.id}.", 503
            )
        options = {**spec.defaults, **self.options.get(spec.id, {})}
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SERVER_ROOT), env.get("PYTHONPATH")]))
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        cwd = self.models_dir / "work" / spec.id
        cwd.mkdir(parents=True, exist_ok=True)
        args = [
            str(python), "-m", "androidreader_server.engines.worker",
            "--module", spec.module,
            "--options", json.dumps(options),
            "--models-dir", str(self.models_dir),
        ]
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            creationflags=flags,
        )
        worker = _Worker(spec, proc)
        try:
            await worker.expect_control({"hello"}, 120)
            t0 = time.perf_counter()
            await worker.send({"op": "load"})
            loaded = await worker.expect_control({"loaded"}, self.load_timeout_s)
        except BaseException:
            await worker.stop()
            raise
        worker.sample_rate = int(loaded.get("sample_rate") or 24000)
        self.stats[spec.id].load_s = round(time.perf_counter() - t0, 1)
        self.stats[spec.id].vram_mb = loaded.get("vram_mb")
        logger.info("%s loaded in %.1fs", spec.label, self.stats[spec.id].load_s)
        return worker

    async def _ensure(self, spec: EngineSpec) -> _Worker:
        """Caller holds the lock."""
        if self._worker and self._worker.alive and self._worker.spec.id == spec.id:
            return self._worker
        await self._stop_worker()
        self.state, self.state_detail, self.loading_engine = "loading", f"Loading {spec.label}", spec.id
        try:
            self._worker = await self._start(spec)
        except EngineError as exc:
            self.state, self.state_detail = "error", exc.detail
            raise
        finally:
            self.loading_engine = None
        self.state, self.state_detail = "ready", ""
        return self._worker

    async def _stop_worker(self) -> None:
        if self._worker is not None:
            logger.info("stopping %s", self._worker.spec.label)
            await self._worker.stop()
            self._worker = None
            self.state = "idle"

    async def preload(self, engine_id: str) -> None:
        spec = self.spec(engine_id)
        async with self._lock:
            await self._ensure(spec)

    async def unload(self) -> None:
        """Free the GPU (e.g. before running a local LLM)."""
        async with self._lock:
            await self._stop_worker()

    @contextlib.asynccontextmanager
    async def exclusive(self):
        """Hold the GPU for something else (the dramatization LLM): no TTS model is loaded meanwhile."""
        async with self._lock:
            await self._stop_worker()
            yield

    @property
    def tts_waiting(self) -> bool:
        """True when synthesis requests are queued behind the current GPU holder."""
        return self._waiting > 0

    def idle_seconds(self) -> float:
        """Seconds since speech was last requested or produced."""
        if self._waiting:
            return 0.0
        return time.monotonic() - self._last_activity

    async def close(self) -> None:
        await self._stop_worker()

    # -- synthesis -----------------------------------------------------------

    def synthesize(self, engine_id: str, payload: dict):
        """Async iterator: StreamStart, AudioChunk..., Done. Raises EngineError."""
        return self._run(engine_id, {"op": "synth", **payload})

    def design(self, engine_id: str, description: str, text: str, seed: int | None = None):
        if not self.spec(engine_id).voice_design:
            raise EngineError(f"{self.spec(engine_id).label} cannot design voices", 400)
        return self._run(engine_id, {"op": "design", "description": description, "text": text, "seed": seed})

    async def _run(self, engine_id: str, msg: dict):
        spec = self.spec(engine_id)
        self._waiting += 1
        self._last_activity = time.monotonic()
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        worker: _Worker | None = None
        q: asyncio.Queue | None = None
        rid = uuid.uuid4().hex
        finished = False
        try:
            worker = await self._ensure(spec)
            q = asyncio.Queue()
            worker.queues[rid] = q
            await worker.send({**msg, "id": rid})
            while True:
                try:
                    kind, data = await asyncio.wait_for(q.get(), self.synth_timeout_s)
                except asyncio.TimeoutError as exc:
                    finished = True
                    await self._stop_worker()
                    raise EngineError(f"{spec.label} produced no audio for {self.synth_timeout_s:.0f}s", 504) from exc
                if kind == "exit":
                    finished = True
                    detail = worker.crash_detail()
                    await self._stop_worker()
                    raise EngineError(detail, 503)
                if kind == "audio":
                    yield AudioChunk(data)
                    continue
                t = data.get("type")
                if t == "start":
                    yield StreamStart(int(data.get("sample_rate") or worker.sample_rate), spec.id)
                elif t == "done":
                    finished = True
                    self.stats[spec.id].record(data.get("gen_s", 0.0), data.get("audio_s", 0.0), data.get("vram_mb"))
                    yield Done(data.get("gen_s", 0.0), data.get("audio_s", 0.0))
                    return
                elif t == "cancelled":
                    finished = True
                    return
                elif t == "error":
                    finished = True
                    raise EngineError(f"{spec.label}: {data.get('detail')}")
        finally:
            if not finished and worker is not None and q is not None and worker.alive:
                # The consumer went away mid-request: cancel, then release the
                # GPU once the worker confirms, without blocking the caller.
                asyncio.ensure_future(self._drain(worker, rid, q))
            else:
                if worker is not None:
                    worker.queues.pop(rid, None)
                self._last_activity = time.monotonic()
                self._lock.release()

    async def _drain(self, worker: _Worker, rid: str, q: asyncio.Queue) -> None:
        try:
            await worker.send({"op": "cancel", "id": rid})
            deadline = time.monotonic() + self.synth_timeout_s
            while True:
                kind, data = await asyncio.wait_for(q.get(), max(0.1, deadline - time.monotonic()))
                if kind == "exit" or (kind == "json" and data.get("type") in {"done", "cancelled", "error"}):
                    break
        except (asyncio.TimeoutError, EngineError):
            await self._stop_worker()
        finally:
            worker.queues.pop(rid, None)
            self._lock.release()
