"""One line, several deliveries, through the server's engine path: does the model follow the delivery?

    .venv\\Scripts\\python.exe scripts\\delivery_showcase.py <engine> [--voice ID] [--set key=value]

Writes data\\bench\\deliveries-<engine>-<voice>.wav (one clip per delivery, in order) and prints loudness (RMS dBFS),
median pitch (Hz) and speaking rate per delivery: shouting should be louder and higher, whispering quieter, etc.
"""
import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elvenreader_server.audio import trim_silence, wav_bytes  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402
from elvenreader_server.engines.manager import AudioChunk, EngineManager, StreamStart  # noqa: E402
from elvenreader_server.engines.registry import ENGINES  # noqa: E402
from elvenreader_server.voices import VoiceStore  # noqa: E402

LINE = "I told you never to come back here. Do you have any idea what you've done?"
DELIVERIES = [None, "shouting, furious", "whispering, scared", "tearful, heartbroken", "cheerful, laughing",
              "cold, menacing", "sarcastic"]


def pitch_hz(x: np.ndarray, rate: int) -> float:
    """Median autocorrelation pitch over voiced 40 ms frames."""
    frame, hop, out = int(rate * 0.04), int(rate * 0.02), []
    lo, hi = int(rate / 400), int(rate / 70)
    for i in range(0, len(x) - frame, hop):
        f = x[i:i + frame] - x[i:i + frame].mean()
        if np.sqrt((f ** 2).mean()) < 0.02:
            continue
        ac = np.correlate(f, f, "full")[frame - 1:]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            out.append(rate / lag)
    return float(np.median(out)) if out else 0.0


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine")
    p.add_argument("--voice", default="a304a8d5")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    cfg = load_config()
    opts = dict(kv.split("=", 1) for kv in a.set)
    manager = EngineManager(ENGINES, cfg.models_path, options={a.engine: {**cfg.engine_options.get(a.engine, {}), **opts}},
                            load_timeout_s=1800)
    voice = VoiceStore(cfg.data_path / "voices").get(a.voice)
    await manager.preload(a.engine)
    clips, rate = [], 24000
    print(f"{'delivery':24} {'secs':>5} {'dBFS':>6} {'pitch':>6} {'words/s':>7}")
    for d in DELIVERIES:
        parts = []
        stream = manager.synthesize(a.engine, {"text": LINE, "voice": voice.worker_spec(), "language": "English",
                                               "seed": voice.seed, "delivery": d})
        async with contextlib.aclosing(stream):
            async for ev in stream:
                if isinstance(ev, StreamStart):
                    rate = ev.sample_rate
                elif isinstance(ev, AudioChunk):
                    parts.append(ev.pcm)
        pcm = trim_silence(b"".join(parts), rate)
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        secs = len(x) / rate
        db = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
        print(f"{d or '(none)':24} {secs:5.2f} {db:6.1f} {pitch_hz(x, rate):6.0f} {len(LINE.split()) / secs:7.2f}", flush=True)
        clips.append(pcm + bytes(int(rate * 0.5) * 2))
    out = cfg.data_path / "bench" / f"deliveries-{a.engine}{''.join('-' + k + '=' + v for k, v in opts.items())}-{a.voice}.wav"
    out.write_bytes(wav_bytes(b"".join(clips), rate))
    print("wrote", out)
    await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
