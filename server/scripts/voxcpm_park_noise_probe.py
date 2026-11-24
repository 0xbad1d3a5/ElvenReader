"""Do designed voices come out noisier after VoxCPM2 has been parked (moved to system RAM and back)? Designs the same
voice (description and seed) on a freshly loaded model, then after park/unpark, and measures how far the pauses sit
under the speech (noise_gap_db: the server wants 25 dB or more). Also a cloned line with a library voice, both ways.
Run in the VoxCPM2 venv with the server stopped:

    ..\\models\\venvs\\voxcpm\\Scripts\\python.exe scripts\\voxcpm_park_noise_probe.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server.engines.worker import _prepare_environment  # noqa: E402

_prepare_environment(str(ROOT.parent / "models"))
from elvenreader_server.app import DESIGN_TEXT  # noqa: E402
from elvenreader_server.audio import noise_gap_db  # noqa: E402
from elvenreader_server.engines.registry import ENGINES  # noqa: E402
from elvenreader_server.engines.voxcpm_engine import Engine  # noqa: E402
from elvenreader_server.voices import VoiceStore  # noqa: E402

DESCRIPTION = "An adult woman with an ordinary, mid-pitched voice, natural and clear."
VOICE = VoiceStore(ROOT / "data" / "voices").get("ch-4938fdcb-dad").worker_spec()
LINE = "I told you already, the gate closes at sunset, and nobody goes out after that."


def pcm(chunks) -> bytes:
    a = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def measure(e, label: str) -> None:
    for seed in (7, 7926):
        d = pcm(list(e.design_voice(DESCRIPTION, DESIGN_TEXT, seed)))
        print(f"{label:8s} design seed {seed:5d}: gap {noise_gap_db(d, e.sample_rate):5.1f} dB", flush=True)
    s = pcm(list(e.synthesize({"text": LINE, "voice": VOICE, "whole": True}, lambda: False)))
    print(f"{label:8s} cloned line:       gap {noise_gap_db(s, e.sample_rate):5.1f} dB", flush=True)


e = Engine({**ENGINES["voxcpm"].defaults, **({"cfg_value": 1.75, "temperature": 0.95} if "--server" in sys.argv else {})})
e.load()
measure(e, "fresh")
if "--server" not in sys.argv:
    e.park()
    e.unpark()
    measure(e, "unparked")
