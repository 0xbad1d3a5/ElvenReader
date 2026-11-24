"""Can a loaded, compiled VoxCPM2 step aside for the LLM without being rebuilt? Loads it as the adapter does, makes a
line, moves everything to system RAM ("park": what VRAM is left?), brings it back ("unpark": how long?), and makes
lines again (recompiled? still the right words?). Run in the VoxCPM2 venv with the server stopped:

    ..\\models\\venvs\\voxcpm\\Scripts\\python.exe scripts\\voxcpm_park_probe.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server.engines.worker import _prepare_environment  # noqa: E402

_prepare_environment(str(ROOT.parent / "models"))
from elvenreader_server.engines.voxcpm_engine import Engine  # noqa: E402


def gpu_mb() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    return out


def line(engine, text, voice=None):
    import numpy as np

    t0 = time.perf_counter()
    chunks = list(engine.synthesize({"text": text, "voice": voice or {}, "whole": True}, lambda: False))
    took = time.perf_counter() - t0
    heard = re.sub(r"<\|[^|]*\|>", "", engine._heard([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]))
    print(f"   {took:5.1f} s  {text!r} -> heard {heard!r}", flush=True)


VOICE = None
if "--voice" in sys.argv:  # a voice of the library, as the server sends it (cloning from its clip)
    from elvenreader_server.voices import VoiceStore

    VOICE = VoiceStore(ROOT / "data" / "voices").get(sys.argv[sys.argv.index("--voice") + 1]).worker_spec()

t0 = time.perf_counter()
e = Engine({"normalize": True, "check_endings": True})
e.load()
print(f"load {time.perf_counter() - t0:.1f} s, GPU {gpu_mb()}", flush=True)
for t in ["The train leaves at seven.", "I put the letters on the table by the door."]:
    line(e, t, VOICE)

t0 = time.perf_counter()
e.park()
print(f"park {time.perf_counter() - t0:.1f} s, GPU {gpu_mb()}", flush=True)
time.sleep(2)
t0 = time.perf_counter()
e.unpark()
print(f"unpark {time.perf_counter() - t0:.1f} s, GPU {gpu_mb()}", flush=True)
for t in ["The train leaves at seven.", "I put the letters on the table by the door.", "Hello there, how are you?"]:
    line(e, t, VOICE)
# twice, as the server would over a night
e.park()
e.unpark()
print(f"second park/unpark, GPU {gpu_mb()}", flush=True)
line(e, "One year, three months, and sixteen days, in fact.", VOICE)
