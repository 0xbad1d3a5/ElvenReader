"""Runs engine workers and serialises access to the GPU.

The site's speech model stays loaded in the main slot; asking it for another engine stops the
current worker and starts the requested one (the server only does that when the speech model setting
changes). Voice design with another model runs in a second, temporary slot. Requests are
served one at a time per slot, the most urgent first (RankedLock: with several people reading, the passage
someone needs next goes before others' read-ahead), then in arrival order. When a client goes away mid-sentence,
the worker is told to cancel and the manager waits for it to acknowledge
before serving the next request, so frames never get mixed up.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import protocol
from .registry import EngineSpec

logger = logging.getLogger("elvenreader.engines")

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
        self.parked = False  # moved to system RAM for something else on the GPU (park/unpark)
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


class RankedLock:
    """An asyncio lock whose waiters are served most urgent first: the lowest rank() when it is released (asked
    then, so a rank may change while it waits, e.g. as a reader's playback moves on), then in arrival order.
    `async with` and acquire() without a rank are rank 0 (wanted now)."""

    def __init__(self):
        self._locked = False
        self._waiters: list[tuple[Callable[[], float], int, asyncio.Future]] = []
        self._order = itertools.count()

    def locked(self) -> bool:
        return self._locked

    async def acquire(self, rank: Callable[[], float] | None = None) -> bool:
        if not self._locked:
            self._locked = True
            return True
        fut = asyncio.get_running_loop().create_future()
        entry = (rank or (lambda: 0.0), next(self._order), fut)
        self._waiters.append(entry)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # it was handed over just as the waiter went away: pass it on
            raise
        finally:
            if entry in self._waiters:
                self._waiters.remove(entry)
        return True

    def release(self) -> None:
        self._waiters = [w for w in self._waiters if not w[2].done()]
        if not self._waiters:
            self._locked = False
            return

        def urgency(w) -> tuple[float, int]:
            try:
                return float(w[0]()), w[1]
            except Exception:  # noqa: BLE001 - a broken rank must not stall the GPU
                return 0.0, w[1]

        nxt = min(self._waiters, key=urgency)
        self._waiters.remove(nxt)
        nxt[2].set_result(True)  # still locked: it is the new holder's

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc) -> None:
        self.release()


class _Slot:
    """A place for one worker on the GPU, served one request at a time."""

    def __init__(self, name: str):
        self.name = name
        self.lock = RankedLock()
        self.worker: _Worker | None = None
        self.waiting = 0
        self.last_activity = time.monotonic() - 3600
        self.state = "idle"
        self.detail = ""
        self.loading: str | None = None
        self.since = 0.0  # when the current load started

    @property
    def active(self) -> str | None:
        return self.worker.spec.id if self.worker and self.worker.alive else None


class EngineManager:
    """Places on the GPU: the speech model the site reads with (`main`, loaded once and kept); a narration model
    (`narration`, optional) that reads the narrator's lines next to it, so two readers (or a reader's narration and
    dialogue) are made at the same time; and a voice designer (`design`) for when the design model is another one.
    The designer is loaded only while voices are being designed and unloaded after `designer_idle_s`; it never
    evicts the speech models. Each place serves one request at a time; the places run side by side."""

    def __init__(
        self,
        specs: dict[str, EngineSpec],
        models_dir: Path,
        *,
        options: dict | None = None,
        load_timeout_s: float = 900.0,
        synth_timeout_s: float = 180.0,
        designer_idle_s: float = 180.0,
        park_speech: bool = True,
        load_times_path: Path | None = None,
    ):
        self.specs = specs
        self.models_dir = Path(models_dir)
        self.options = options or {}
        self.load_timeout_s = load_timeout_s
        self.synth_timeout_s = synth_timeout_s
        self.designer_idle_s = designer_idle_s
        # Make room for the LLM by moving the speech model to system RAM (when its engine can), not stopping it.
        self.park_speech = park_speech
        self.stats: dict[str, EngineStats] = {k: EngineStats() for k in specs}
        self._main = _Slot("main")
        self._narrator = _Slot("narration")
        self._design = _Slot("design")
        # The narration model (loaded in the narration slot), when it isn't the speech model itself.
        self.narration: str | None = None
        self._reaper: asyncio.Task | None = None
        # The site's speech model (loaded in the main slot, now or on the first request).
        self.resident: str | None = None
        self.designer_load_s: dict[str, float] = {}
        # How long recent loads took, from starting the worker to ready ({"main"/"design": {engine: [s, ...]}}),
        # kept across restarts so the banner's first estimate after a reboot is not a guess.
        self.load_times_path = Path(load_times_path) if load_times_path else None
        self._load_times: dict[str, dict[str, list[float]]] = {}
        if self.load_times_path and self.load_times_path.exists():
            try:
                self._load_times = json.loads(self.load_times_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._load_times = {}
        self._speaking = 0
        self._last_speech = time.monotonic() - 3600
        # Called before the designer loads, to make room on the GPU (the server stops a GPU LLM here).
        self.before_designer = None

    # -- information ---------------------------------------------------------

    def spec(self, engine_id: str) -> EngineSpec:
        spec = self.specs.get(engine_id)
        if spec is None:
            raise EngineError(f"Unknown engine {engine_id!r}", 400)
        return spec

    @property
    def active(self) -> str | None:
        return self._main.active

    @property
    def designer(self) -> str | None:
        return self._design.active

    def _slot_for(self, engine_id: str) -> _Slot:
        """The narration model's requests go to its own place; everything else to the speech model's."""
        if self.narration and engine_id == self.narration and engine_id != self.resident:
            return self._narrator
        return self._main

    @property
    def state(self) -> str:
        return self._main.state

    def status(self) -> dict:
        def expected(slot: _Slot) -> float | None:
            if not slot.loading:
                return None
            kind = "design" if slot is self._design else "main"
            recent = (self._load_times.get(kind) or {}).get(slot.loading)
            if recent:
                return max(recent)  # loads just after a reboot are the slow ones (cold disk cache)
            if slot is self._design:
                return self.designer_load_s.get(slot.loading, 15.0)
            return (self.stats.get(slot.loading) or EngineStats()).load_s or 30.0

        def label(engine_id: str | None) -> str | None:
            return self.specs[engine_id].label if engine_id in self.specs else engine_id

        return {
            "active": self.active,
            "state": self._main.state,
            "detail": self._main.detail,
            "loading": self._main.loading,
            "busy": self._main.lock.locked(),
            "designer": self.designer,
            "designer_state": self._design.state,
            # For the website's banner: what is loading, for how long so far and about how long it takes.
            "speech": {"engine": self._main.loading or self.active, "label": label(self._main.loading or self.active),
                       "state": self._main.state, "loading_s": round(time.monotonic() - self._main.since, 1)
                       if self._main.loading else None, "expected_s": expected(self._main)},
            "narration": {"engine": self._narrator.loading or self._narrator.active,
                          "label": label(self._narrator.loading or self._narrator.active), "state": self._narrator.state,
                          "loading_s": round(time.monotonic() - self._narrator.since, 1)
                          if self._narrator.loading else None, "expected_s": expected(self._narrator)},
            "voice_designer": {"engine": self._design.loading or self.designer,
                               "label": label(self._design.loading or self.designer), "state": self._design.state,
                               "loading_s": round(time.monotonic() - self._design.since, 1)
                               if self._design.loading else None, "expected_s": expected(self._design)},
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

    async def _start(self, spec: EngineSpec, extra: dict | None = None) -> _Worker:
        python = spec.python(self.models_dir)
        if not python.exists():
            raise EngineError(
                f"{spec.label} is not installed (missing {python}). Run setup-engines.ps1 {spec.id}.", 503
            )
        options = {**spec.defaults, **self.options.get(spec.id, {}), **(extra or {})}
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SERVER_ROOT), env.get("PYTHONPATH")]))
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        cwd = self.models_dir / "work" / spec.id
        cwd.mkdir(parents=True, exist_ok=True)
        args = [
            str(python), "-m", "elvenreader_server.engines.worker",
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
        if not extra:  # the designer's numbers would mislead (it loads only part of the model)
            self.stats[spec.id].load_s = round(time.perf_counter() - t0, 1)
            self.stats[spec.id].vram_mb = loaded.get("vram_mb")
        else:
            self.designer_load_s[spec.id] = round(time.perf_counter() - t0, 1)
        logger.info("%s loaded in %.1fs%s", spec.label, time.perf_counter() - t0, " (voice designer)" if extra else "")
        return worker

    async def _ensure(self, slot: _Slot, spec: EngineSpec, extra: dict | None = None) -> _Worker:
        """Caller holds the slot's lock."""
        if slot.worker and slot.worker.alive and slot.worker.spec.id == spec.id:
            if not slot.worker.parked:
                return slot.worker
            # Back from system RAM (about a second), instead of loading again (about a minute).
            t0 = time.perf_counter()
            try:
                await slot.worker.send({"op": "unpark"})
                await slot.worker.expect_control({"unparked"}, 120)
                slot.worker.parked = False
                slot.state, slot.detail = "ready", ""
                logger.info("%s back on the GPU in %.1fs", spec.label, time.perf_counter() - t0)
                return slot.worker
            except EngineError as exc:
                logger.warning("%s could not come back from RAM (%s); loading it again", spec.label, exc.detail)
        await self._stop(slot)
        if slot is self._design and self.before_designer is not None:
            await self.before_designer()
        slot.state, slot.detail, slot.loading = "loading", f"Loading {spec.label}", spec.id
        slot.since = time.monotonic()
        try:
            slot.worker = await self._start(spec, extra)
        except EngineError as exc:
            slot.state, slot.detail = "error", exc.detail
            raise
        except BaseException:
            # Cancelled mid-load (the request went away, the speech model was switched): nothing is loading.
            slot.state, slot.detail = "idle", ""
            raise
        finally:
            slot.loading = None
        self._remember_load("design" if slot is self._design else "main", spec.id, time.monotonic() - slot.since)
        slot.state, slot.detail = "ready", ""
        return slot.worker

    LOAD_TIMES_KEPT = 3

    def _remember_load(self, kind: str, engine_id: str, seconds: float) -> None:
        times = self._load_times.setdefault(kind, {}).setdefault(engine_id, [])
        times.append(round(seconds, 1))
        del times[:-self.LOAD_TIMES_KEPT]
        if self.load_times_path:
            try:
                self.load_times_path.write_text(json.dumps(self._load_times), encoding="utf-8")
            except OSError as exc:
                logger.warning("could not save load times: %s", exc)

    async def _stop(self, slot: _Slot) -> None:
        if slot.worker is not None:
            logger.info("stopping %s%s", slot.worker.spec.label, " (voice designer)" if slot is self._design else "")
            await slot.worker.stop()
            slot.worker = None
            slot.state = "idle"

    async def _park_or_stop(self, slot: _Slot) -> None:
        """Free the GPU: the worker moves its model to system RAM when its engine can (BaseEngine.park), so
        coming back takes a second rather than a reload; otherwise it is stopped. Caller holds the slot's lock."""
        w = slot.worker
        if w is None or w.parked:
            return
        if w.alive and self.park_speech:
            t0 = time.perf_counter()
            try:
                await w.send({"op": "park"})
                reply = await w.expect_control({"parked"}, 60)
                if reply.get("ok"):
                    w.parked = True
                    slot.state, slot.detail = "parked", "In system RAM while the GPU does background work"
                    logger.info("%s moved to system RAM in %.1fs", w.spec.label, time.perf_counter() - t0)
                    return
            except EngineError as exc:
                logger.warning("%s could not move to RAM (%s); stopping it", w.spec.label, exc.detail)
        await self._stop(slot)

    @property
    def parked_vram_gb(self) -> float:
        """VRAM the parked speech models still hold (their CUDA context and graphs): the LLM leaves room for it."""
        return sum(w.spec.parked_vram_gb for w in (self._main.worker, self._narrator.worker)
                   if w is not None and w.alive and w.parked)

    async def wait_settled(self, timeout: float = 30.0) -> None:
        """Wait (at most `timeout`) until the speech model is back on the GPU and the speech asked for meanwhile
        is made: an LLM starting next to it at the same time (4 GB onto the GPU) made the first line after a
        return from idle take 23 s instead of 2."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            busy = self._speaking or any(
                slot.loading or (slot.worker is not None and slot.worker.parked) or slot.waiting or slot.lock.locked()
                for slot in (self._main, self._narrator))
            if not busy:
                return
            await asyncio.sleep(0.25)

    async def preload(self, engine_id: str) -> None:
        """Load the speech model the site reads with (replacing another one)."""
        spec = self.spec(engine_id)
        self.resident = engine_id
        async with self._main.lock:
            await self._ensure(self._main, spec)
        if self.narration == engine_id:  # it reads the narration too now: the narration place isn't needed
            async with self._narrator.lock:
                await self._stop(self._narrator)

    async def preload_narration(self, engine_id: str | None) -> None:
        """Load the narration model next to the speech model (None, or the speech model itself: stop it)."""
        self.narration = engine_id or None
        async with self._narrator.lock:
            if not engine_id or engine_id == self.resident:
                await self._stop(self._narrator)
                return
            await self._ensure(self._narrator, self.spec(engine_id))

    async def unload(self) -> None:
        """Free the GPU entirely (e.g. before running something else on it)."""
        async with self._main.lock, self._narrator.lock, self._design.lock:
            await self._stop(self._design)
            await self._stop(self._narrator)
            await self._stop(self._main)

    async def unload_designer(self) -> None:
        async with self._design.lock:
            await self._stop(self._design)

    @contextlib.asynccontextmanager
    async def exclusive(self, evict: bool = True):
        """Hold the GPU for something else (the dramatization LLM). The voice designer is unloaded; the speech
        model too only with `evict` (otherwise it stays loaded and speech keeps being served meanwhile)."""
        if evict:
            async with self._main.lock, self._narrator.lock, self._design.lock:
                await self._stop(self._design)
                await self._park_or_stop(self._narrator)
                await self._park_or_stop(self._main)
                yield
        else:
            # Someone is designing voices: wait until the designer has been idle (it is unloaded then), so
            # each "Make 3 options" doesn't reload it. Checked again once the lock is ours: a design that
            # was running when we queued has just finished.
            def designing() -> bool:
                return (self._design.worker is not None
                        and time.monotonic() - self._design.last_activity < self.designer_idle_s)

            while True:
                while designing():
                    await asyncio.sleep(1)
                await self._design.lock.acquire()
                if not designing():
                    break
                self._design.lock.release()
            try:
                await self._stop(self._design)
                yield
            finally:
                self._design.lock.release()

    @property
    def tts_waiting(self) -> bool:
        """True when synthesis requests are queued behind the current GPU holder."""
        return self._main.waiting > 0 or self._narrator.waiting > 0

    def idle_seconds(self) -> float:
        """Seconds since speech was last requested or produced (voice design doesn't count)."""
        if self._speaking:
            return 0.0
        return time.monotonic() - self._last_speech

    async def close(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        await self._stop(self._design)
        await self._stop(self._narrator)
        await self._stop(self._main)

    # -- synthesis -----------------------------------------------------------

    def synthesize(self, engine_id: str, payload: dict, rank: Callable[[], float] | None = None):
        """Async iterator: StreamStart, AudioChunk..., Done. Raises EngineError. `rank`: how urgent it is (lower
        first; see RankedLock), e.g. how far ahead of a reader's playback the passage is."""
        return self._speech(self._run(self._slot_for(engine_id), engine_id, {"op": "synth", **payload}, rank=rank))

    async def _speech(self, stream):
        """Marks speech (reading) as going on, for idle_seconds: voice design doesn't count as listening."""
        self._speaking += 1
        self._last_speech = time.monotonic()
        try:
            async with contextlib.aclosing(stream):
                async for ev in stream:
                    yield ev
        finally:
            self._speaking -= 1
            self._last_speech = time.monotonic()

    def design(self, engine_id: str, description: str, text: str, seed: int | None = None):
        """Design with the speech model when it is the design model; otherwise with the designer, next to it."""
        if not self.spec(engine_id).voice_design:
            raise EngineError(f"{self.spec(engine_id).label} cannot design voices", 400)
        msg = {"op": "design", "description": description, "text": text, "seed": seed}
        if engine_id in (self.active, self.resident, self._main.loading):
            return self._run(self._main, engine_id, msg)
        return self._run(self._design, engine_id, msg, extra={"design_only": True})

    def _reap_later(self) -> None:
        if self._reaper and not self._reaper.done():
            self._reaper.cancel()

        async def reap():
            await asyncio.sleep(self.designer_idle_s)
            if not self._design.lock.locked() and time.monotonic() - self._design.last_activity >= self.designer_idle_s - 1:
                async with self._design.lock:
                    await self._stop(self._design)

        self._reaper = asyncio.ensure_future(reap())

    async def _run(self, slot: _Slot, engine_id: str, msg: dict, extra: dict | None = None,
                   rank: Callable[[], float] | None = None):
        spec = self.spec(engine_id)
        slot.waiting += 1
        slot.last_activity = time.monotonic()
        try:
            await slot.lock.acquire(rank)
        finally:
            slot.waiting -= 1
        worker: _Worker | None = None
        q: asyncio.Queue | None = None
        rid = uuid.uuid4().hex
        finished = False
        try:
            worker = await self._ensure(slot, spec, extra)
            q = asyncio.Queue()
            worker.queues[rid] = q
            await worker.send({**msg, "id": rid})
            while True:
                try:
                    kind, data = await asyncio.wait_for(q.get(), self.synth_timeout_s)
                except asyncio.TimeoutError as exc:
                    finished = True
                    await self._stop(slot)
                    raise EngineError(f"{spec.label} produced no audio for {self.synth_timeout_s:.0f}s", 504) from exc
                if kind == "exit":
                    finished = True
                    detail = worker.crash_detail()
                    await self._stop(slot)
                    raise EngineError(detail, 503)
                if kind == "audio":
                    yield AudioChunk(data)
                    continue
                t = data.get("type")
                if t == "start":
                    yield StreamStart(int(data.get("sample_rate") or worker.sample_rate), spec.id)
                elif t == "done":
                    finished = True
                    if slot is self._main:
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
                # slot once the worker confirms, without blocking the caller.
                asyncio.ensure_future(self._drain(slot, worker, rid, q))
            else:
                if worker is not None:
                    worker.queues.pop(rid, None)
                slot.last_activity = time.monotonic()
                slot.lock.release()
                if slot is self._design:
                    self._reap_later()

    async def _drain(self, slot: _Slot, worker: _Worker, rid: str, q: asyncio.Queue) -> None:
        try:
            await worker.send({"op": "cancel", "id": rid})
            deadline = time.monotonic() + self.synth_timeout_s
            while True:
                kind, data = await asyncio.wait_for(q.get(), max(0.1, deadline - time.monotonic()))
                if kind == "exit" or (kind == "json" and data.get("type") in {"done", "cancelled", "error"}):
                    break
        except (asyncio.TimeoutError, EngineError):
            await self._stop(slot)
        finally:
            worker.queues.pop(rid, None)
            slot.lock.release()
            if slot is self._design:
                self._reap_later()
