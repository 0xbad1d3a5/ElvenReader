"""Synthesize the same passage with each installed engine and report speed.

    .venv\\Scripts\\python.exe scripts\\bench_engines.py [engine ...] [--voice ID]

Writes data\\bench\\<engine>-<voice>.wav and prints load time, real-time factor
(generation time / audio duration; below 1.0 is faster than real time) and VRAM.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elvenreader_server.audio import trim_silence, wav_bytes  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402
from elvenreader_server.engines.manager import AudioChunk, EngineManager, StreamStart  # noqa: E402
from elvenreader_server.engines.registry import ENGINES  # noqa: E402
from elvenreader_server.voices import VoiceStore  # noqa: E402

PASSAGE = [
    ("The rain had not stopped for three days, and the river was rising faster than anyone in the village remembered.", None),
    ("\u201cStop!\u201d", "shouting, urgent"),
    ("She shouted, but the boy was already halfway across the old stone bridge.", None),
    ("Hermione glanced at the clock: it was a quarter past eleven, on the fourteenth of March.", None),
    ("\u201cI don\u2019t think we should be here,\u201d he whispered.", "whispering, nervous"),
]


async def bench(engine_id: str, voice_id: str, out_dir: Path, options: dict | None = None, tag: str = "") -> dict:
    cfg = load_config()
    engine_options = {**cfg.engine_options, engine_id: {**cfg.engine_options.get(engine_id, {}), **(options or {})}}
    manager = EngineManager(ENGINES, cfg.models_path, options=engine_options, load_timeout_s=1800)
    voice = VoiceStore(cfg.data_path / "voices").get(voice_id)
    if voice is None:
        raise SystemExit(f"no voice {voice_id}")
    t0 = time.perf_counter()
    await manager.preload(engine_id)
    # One untimed sentence: the first call compiles kernels / fills caches.
    warm = manager.synthesize(engine_id, {"text": "Warming up the voice.", "voice": voice.worker_spec(),
                                          "language": "English", "seed": voice.seed, "delivery": "calm"})
    async with contextlib.aclosing(warm):
        async for _ in warm:
            pass
    load_s = time.perf_counter() - t0
    rate, clips, gen_total, audio_total = 24000, [], 0.0, 0.0
    first_latency = []
    for text, delivery in PASSAGE:
        t1 = time.perf_counter()
        first = None
        parts = []
        stream = manager.synthesize(engine_id, {
            "text": text, "voice": voice.worker_spec(), "language": "English",
            "seed": voice.seed, "delivery": delivery,
        })
        async with contextlib.aclosing(stream):
            async for ev in stream:
                if isinstance(ev, StreamStart):
                    rate = ev.sample_rate
                elif isinstance(ev, AudioChunk):
                    first = first or time.perf_counter() - t1
                    parts.append(ev.pcm)
        gen = time.perf_counter() - t1
        pcm = trim_silence(b"".join(parts), rate)
        dur = len(pcm) / 2 / rate
        gen_total += gen
        audio_total += dur
        first_latency.append(first or gen)
        clips.append(pcm + bytes(int(rate * 0.35) * 2))
        print(f"  {engine_id}: {dur:5.2f}s audio in {gen:5.2f}s  {text[:50]!r}", flush=True)
    out = out_dir / f"{engine_id}{tag}-{voice_id}.wav"
    out.write_bytes(wav_bytes(b"".join(clips), rate))
    stats = manager.stats[engine_id].to_json()
    await manager.close()
    return {
        "engine": engine_id + tag, "load_s": round(load_s, 1), "rtf": round(gen_total / audio_total, 3),
        "first_audio_s": round(sum(first_latency) / len(first_latency), 2),
        "vram_mb": stats.get("vram_mb"), "sample_rate": rate, "file": str(out),
    }


async def main() -> None:
    import logging
    import os

    logging.basicConfig(level=logging.DEBUG if os.environ.get("BENCH_DEBUG") else logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("engines", nargs="*", default=list(ENGINES))
    parser.add_argument("--voice", default="a304a8d5")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="engine option for this run, e.g. --set mode=reference (repeatable)")
    args = parser.parse_args()
    options = {}
    for kv in args.set:
        key, _, value = kv.partition("=")
        options[key] = int(value) if value.isdigit() else float(value) if value.replace(".", "", 1).isdigit() \
            else {"true": True, "false": False}.get(value.lower(), value)
    tag = "".join(f"-{k}={v}" for k, v in options.items())
    out_dir = load_config().data_path / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for engine in args.engines:
        try:
            results.append(await bench(engine, args.voice, out_dir, options, tag))
        except Exception as exc:  # noqa: BLE001
            print(f"{engine} FAILED: {exc}", flush=True)
            results.append({"engine": engine, "error": str(exc)})
    print("\n| engine | load + warm-up s | RTF | avg first audio s | peak VRAM MB | rate |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['engine']} | failed: {r['error'][:80]} |")
        else:
            print(f"| {r['engine']} | {r['load_s']} | {r['rtf']} | {r['first_audio_s']} | {r['vram_mb']} | {r['sample_rate']} |")


if __name__ == "__main__":
    asyncio.run(main())
