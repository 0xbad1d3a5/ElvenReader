"""Where VoxCPM2's buzz comes from: streaming decode vs whole decode, diffusion steps, guidance.

Runs the model directly (no server). Writes data\\bench\\artifact-<variant>.wav and prints, per variant:
seam: energy of the streaming-minus-whole difference at chunk boundaries vs elsewhere (same seed -> same latents);
hf: share of energy above 12 kHz (hiss/buzz in a 48 kHz output whose reference only reaches 8 kHz);
flat: spectral flatness in the pauses (1 = noise, 0 = tonal hum).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch._inductor.config as inductor

inductor.use_static_cuda_launcher = False
from voxcpm import VoxCPM  # noqa: E402

DATA = Path(r"D:\Dev\ElvenReader\server\data")
voice = json.loads((DATA / "voices" / f"{sys.argv[1]}.json").read_text(encoding="utf-8"))
ref = str(DATA / "voices" / f"{voice['id']}.wav")
TEXT = sys.argv[2] if len(sys.argv) > 2 else (
    "I suppose it's just like him to have a perfectly good reason, but still manage to deliver it in a way "
    "that sounds like a child's excuse. Honestly, what are we going to do with him?")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=True)
rate = model.tts_model.sample_rate
print("reference:", sf.info(ref).samplerate, "Hz;", "output:", rate, "Hz", flush=True)


def run(stream, steps=10, cfg=2.0, mode="continuation", seed=1234):
    kw = dict(cfg_value=cfg, inference_timesteps=steps, reference_wav_path=ref)
    if mode == "continuation":
        kw.update(prompt_wav_path=ref, prompt_text=voice["ref_text"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t = time.time()
    if stream:
        wav = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in model.generate_streaming(text=TEXT, **kw)])
    else:
        wav = np.asarray(model.generate(text=TEXT, retry_badcase=False, **kw), dtype=np.float32).reshape(-1)
    return wav, time.time() - t


def stats(w):
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    f = np.fft.rfftfreq(len(w), 1 / rate)
    hf = spec[f > 12000].sum() / spec.sum()
    win = int(rate * 0.04)
    frames = [w[i:i + win] for i in range(0, len(w) - win, win)]
    quiet = [fr for fr in frames if 1e-4 < np.sqrt((fr ** 2).mean()) < 0.01]
    flat = np.mean([np.exp(np.mean(np.log(s + 1e-12))) / (np.mean(s) + 1e-12)
                    for s in (np.abs(np.fft.rfft(fr * np.hanning(win))) ** 2 + 1e-12 for fr in quiet)]) if quiet else 0
    return hf, flat


results = {}
for name, args in [("stream-s10-cfg2", (True, 10, 2.0)), ("whole-s10-cfg2", (False, 10, 2.0)),
                   ("whole-s20-cfg2", (False, 20, 2.0)), ("whole-s10-cfg1.5", (False, 10, 1.5)),
                   ("stream-s20-cfg1.5", (True, 20, 1.5))]:
    wav, secs = run(*args)
    results[name] = wav
    sf.write(DATA / "bench" / f"artifact-{name}.wav", wav, rate)
    hf, flat = stats(wav)
    print(f"{name:18} {len(wav) / rate:5.2f}s in {secs:5.2f}s  hf>12k {hf:6.2%}  pause flatness {flat:.3f}", flush=True)

a, b = results["stream-s10-cfg2"], results["whole-s10-cfg2"]
n = min(len(a), len(b))
d = a[:n] - b[:n]
patch = model.tts_model.patch_size * model.tts_model.chunk_size
idx = np.arange(n)
near = (idx % patch < int(rate * 0.005)) | (idx % patch > patch - int(rate * 0.005))
print(f"\nstreaming vs whole decode, same latents: lengths {len(a)} / {len(b)}, chunk {patch / rate * 1000:.0f} ms")
print(f"  difference energy / signal energy: {(d ** 2).sum() / (b[:n] ** 2).sum():.4%}")
print(f"  difference RMS near chunk seams {np.sqrt((d[near] ** 2).mean()):.5f} vs elsewhere {np.sqrt((d[~near] ** 2).mean()):.5f}")
