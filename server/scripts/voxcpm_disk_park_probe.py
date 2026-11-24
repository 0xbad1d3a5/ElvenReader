"""Parking VoxCPM2 on the SSD instead of in RAM: how long writing the parked weights to a file and reading them back
onto the GPU takes, against keeping them in RAM. Run in the VoxCPM2 venv with the server stopped:

    ..\\models\\venvs\\voxcpm\\Scripts\\python.exe scripts\\voxcpm_disk_park_probe.py [--voice ID]
"""
import ctypes
import gc
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server.engines.worker import _prepare_environment  # noqa: E402

_prepare_environment(str(ROOT.parent / "models"))
import torch  # noqa: E402

from elvenreader_server.engines.voxcpm_engine import Engine  # noqa: E402

FILE = ROOT.parent / "models" / "work" / "voxcpm" / "parked.pt"


def ram_gb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                   [(n, ctypes.c_size_t) for n in ("Peak", "WorkingSetSize", "QPPP", "QPP", "QPNPP", "QNPP",
                                                   "Pagefile", "PeakPagefile")]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**30


def tensors(modules):
    """Every parameter and buffer of the modules, once each: (owner, kind, name, tensor)."""
    seen, out = set(), []
    for i, m in enumerate(modules):
        for sub_name, sub in m.named_modules():
            for kind in ("_parameters", "_buffers"):
                for n, t in getattr(sub, kind).items():
                    if t is not None and id(t) not in seen:
                        seen.add(id(t))
                        out.append((f"{i}.{sub_name}.{kind}.{n}", sub, kind, n, t))
    return out


def spill(engine):
    """Parked weights (in RAM) to the file, then their RAM freed."""
    items = tensors(engine._gpu_modules())
    torch.save({k: (t.data if kind == "_parameters" else t) for k, _, kind, _, t in items}, FILE)
    for _, sub, kind, n, t in items:
        if kind == "_parameters":
            t.data = torch.empty(0)
        else:
            sub._buffers[n] = torch.empty(0)
    gc.collect()


def fill(engine):
    """From the file straight onto the GPU."""
    sd = torch.load(FILE, mmap=True)
    items = tensors(engine._gpu_modules())
    for k, sub, kind, n, t in items:
        src = sd[k].to("cuda", non_blocking=True)
        if kind == "_parameters":
            t.data = src
        else:
            sub._buffers[n] = src
    torch.cuda.synchronize()


def line(engine, text, voice):
    import numpy as np

    t0 = time.perf_counter()
    chunks = list(engine.synthesize({"text": text, "voice": voice or {}, "whole": True}, lambda: False))
    took = time.perf_counter() - t0
    heard = re.sub(r"<\|[^|]*\|>", "", engine._heard([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]))
    print(f"   {took:5.1f} s  {text!r} -> heard {heard!r}", flush=True)


def drop_file_cache(path: Path):
    """Reading a file larger than the free RAM evicts most of the cache; instead, open the file unbuffered once
    so Windows forgets its cached pages (FILE_FLAG_NO_BUFFERING)."""
    h = ctypes.windll.kernel32.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x20000000, None)
    if h != -1:
        ctypes.windll.kernel32.CloseHandle(h)


VOICE = None
if "--voice" in sys.argv:
    from elvenreader_server.voices import VoiceStore

    VOICE = VoiceStore(ROOT / "data" / "voices").get(sys.argv[sys.argv.index("--voice") + 1]).worker_spec()
FILE.parent.mkdir(parents=True, exist_ok=True)
e = Engine({"normalize": True, "check_endings": True})
e.load()
line(e, "The train leaves at seven.", VOICE)
print(f"loaded: RAM {ram_gb():.1f} GB", flush=True)

t0 = time.perf_counter()
e.park()
print(f"park to RAM {time.perf_counter() - t0:.1f} s: RAM {ram_gb():.1f} GB", flush=True)
t0 = time.perf_counter()
e.unpark()
print(f"unpark from RAM {time.perf_counter() - t0:.1f} s", flush=True)
line(e, "I put the letters on the table by the door.", VOICE)

t0 = time.perf_counter()
e.park()
spill(e)
print(f"park to SSD {time.perf_counter() - t0:.1f} s ({FILE.stat().st_size / 2**30:.1f} GB written): RAM {ram_gb():.1f} GB",
      flush=True)
t0 = time.perf_counter()
fill(e)
print(f"unpark from SSD (file still cached by Windows) {time.perf_counter() - t0:.1f} s", flush=True)
line(e, "Hello there, how are you?", VOICE)

e.park()
spill(e)
drop_file_cache(FILE)
t0 = time.perf_counter()
fill(e)
print(f"unpark from SSD (read from the drive) {time.perf_counter() - t0:.1f} s", flush=True)
line(e, "One year, three months, and sixteen days, in fact.", VOICE)
FILE.unlink()
