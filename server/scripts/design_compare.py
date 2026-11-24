"""Design the same character voices with several engines and compare: time, pitch, intelligibility.

    .venv\\Scripts\\python.exe scripts\\design_compare.py picks.json qwen3tts voxcpm

picks.json: [{"name", "gender", "age", "description"}, ...] (the descriptions casting uses). Writes
data\\bench\\design-<engine>-<n>-<name>.wav, prints seconds per design and median pitch (children and women should
be higher than men, the elderly lower), then the files to transcribe.
"""
import asyncio
import contextlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delivery_showcase import pitch_hz  # noqa: E402

from elvenreader_server.app import DESIGN_TEXT  # noqa: E402
from elvenreader_server.audio import trim_silence, wav_bytes  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402
from elvenreader_server.engines.manager import AudioChunk, EngineManager, StreamStart  # noqa: E402
from elvenreader_server.engines.registry import ENGINES  # noqa: E402


async def main():
    picks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = load_config()
    out_dir = cfg.data_path / "bench"
    for engine in sys.argv[2:]:
        manager = EngineManager(ENGINES, cfg.models_path, options=cfg.engine_options, load_timeout_s=1800)
        t0 = time.perf_counter()
        await manager.preload(engine)
        print(f"\n== {engine}: loaded in {time.perf_counter() - t0:.0f}s")
        for i, p in enumerate(picks):
            t1 = time.perf_counter()
            rate, parts = 24000, []
            stream = manager.design(engine, p["description"], DESIGN_TEXT, seed=1000 + i)
            async with contextlib.aclosing(stream):
                async for ev in stream:
                    if isinstance(ev, StreamStart):
                        rate = ev.sample_rate
                    elif isinstance(ev, AudioChunk):
                        parts.append(ev.pcm)
            secs = time.perf_counter() - t1
            pcm = trim_silence(b"".join(parts), rate)
            x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            name = re.sub(r"\W+", "", p["name"])
            (out_dir / f"design-{engine}-{i}-{name}.wav").write_bytes(wav_bytes(pcm, rate))
            print(f"  {p['name']:12} {p['gender']:6} {p['age']:12} {secs:5.1f}s for {len(x) / rate:4.1f}s audio, "
                  f"pitch {pitch_hz(x, rate):4.0f} Hz", flush=True)
        await manager.close()


asyncio.run(main())
