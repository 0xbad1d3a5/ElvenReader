"""Where VoxCPM2's load time goes: times each step the adapter takes when it loads, and the first lines made after
it (torch.compile compiles on first use), with the engine worker's environment (its compile caches). Run in the
VoxCPM2 venv with the server stopped (it needs the GPU):

    ..\\models\\venvs\\voxcpm\\Scripts\\python.exe scripts\\voxcpm_load_probe.py [--split]

--split loads the model twice first (plain; with the denoiser) to show what each part adds.
"""
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT.parent / "models"
sys.path.insert(0, str(ROOT))
from elvenreader_server.engines.worker import _prepare_environment  # noqa: E402

_prepare_environment(str(MODELS))
T0 = time.perf_counter()
last = T0


def step(name):
    global last
    now = time.perf_counter()
    print(f"{now - last:7.1f} s  {name}  (total {now - T0:.1f} s)", flush=True)
    last = now


import torch  # noqa: E402

step("import torch")
import torch._inductor.config as inductor  # noqa: E402

inductor.use_static_cuda_launcher = False
from voxcpm import VoxCPM  # noqa: E402

step("import voxcpm")
if "--split" in sys.argv:
    for name, kw in [("weights only", dict(load_denoiser=False, optimize=False)),
                     ("weights + denoiser", dict(load_denoiser=True, optimize=False))]:
        m = VoxCPM.from_pretrained("openbmb/VoxCPM2", **kw)
        step(f"from_pretrained: {name}")
        del m
        gc.collect()
        torch.cuda.empty_cache()
        last = time.perf_counter()
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=True, optimize=True)
step("VoxCPM.from_pretrained (weights, denoiser, optimize)")
from voxcpm.utils.text_normalize import TextNormalizer  # noqa: E402

tn = TextNormalizer()
step("TextNormalizer")
from funasr import AutoModel  # noqa: E402

asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda", disable_update=True, disable_pbar=True, log_level="ERROR")
step("SenseVoice (line checks)")
for i, text in enumerate(["The train leaves at seven.", "I put the letters on the table by the door this morning.",
                          "Hello there."]):
    for _ in model.generate_streaming(text=text):
        pass
    step(f"line {i + 1} ({len(text)} chars)")
print(f"VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
