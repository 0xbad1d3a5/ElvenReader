"""Breeze TTS 2 on this PC: does it run (Windows, RTX 5080), how much VRAM, how fast, with a cloned voice and a
delivery instruction. Writes WAVs to the output folder. Run with the breeze venv's python from the breeze-tts repo."""
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(r"D:\Dev\ElvenReader\models\src\breeze-tts")
sys.path.insert(0, str(REPO))
from breeze_infer.runtime import load_runtime, resolve_device, set_all_seeds, update_generation_config_for_breeze  # noqa: E402
from breeze_infer.templates import get_template, prepare_inputs, select_template_name  # noqa: E402
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig  # noqa: E402

MODEL = Path(r"D:\Dev\ElvenReader\models\hf\hub\models--BreezeBlue--Breeze-TTS-2\snapshots\3e28c5151381a722f1d8661b4118c298caa77aa4")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REF = r"D:\Dev\ElvenReader\server\data\voices\09f57622.wav"
REF_TEXT = ("The valley had been quiet for a hundred years, and tonight, for the first time, something stirred "
            "beneath the old stone bridge.")

t0 = time.perf_counter()
tokenizer, model, audio_tokenizer = load_runtime(MODEL, device=resolve_device(), attn_implementation="eager")
update_generation_config_for_breeze(model)
import os
from dataclasses import replace
from models.warmup_profile import load_warmup_profile
FAST = [s for s in os.environ.get("BREEZE_FAST", "").split(",") if s]
flags = {f"fast_{s}": True for s in FAST if s != "all"}
runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, FastStreamingConfig(
    max_new_tokens=1500, max_seq_len=2048, fast_all=True if "all" in FAST else None, repetition_penalty=1.1,
    **flags), tokenizer=tokenizer)
if runtime.fast_enabled:
    tw = time.perf_counter()
    profile = replace(load_warmup_profile(REPO / "configs" / "fast.json"), codec_chunk_frames=runtime.codec_chunk_frames)
    runtime.warmup_from_profile(profile)
    print(f"warmup {time.perf_counter() - tw:.1f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
load_s = time.perf_counter() - t0
print(f"loaded in {load_s:.1f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB, sr {runtime.sample_rate}", flush=True)

cases = [
    ("narration", "Ronie watched her dragon nervously. Tsukigake blinked a few times, thinking it over, then stretched "
                  "her neck toward her master and dropped the trout onto the grass.", None),
    ("shout", "Stop right there! Don't you dare take another step!", "Shouting, furious and urgent."),
    ("whisper", "Quiet. They're right outside the door, so don't make a sound.", "Whispering, tense and afraid."),
    ("long", "It was still far too early to go to school, of course. School started at eight o'clock sharp, and it "
             "took them thirty minutes to get to school, including walking. They just needed to leave at seven "
             "thirty. Making breakfast, eating it, and cleaning up would still leave them with more than an hour of "
             "free time.", None),
]
results = []
for name, text, instruction in cases:
    request = {"id": name, "text": text, "speaker": "S0", "ref_audio_path": REF, "ref_text": REF_TEXT}
    if instruction:
        request["instruction"] = instruction
    torch.cuda.reset_peak_memory_stats()
    set_all_seeds(42)
    t = time.perf_counter()
    inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [request], get_template(select_template_name(request)),
                            guidance_scale=4.0 if instruction else 1.0, guidance_scale_ref=None, guidance_scale_ins=None)
    first = None
    chunks = []
    for chunk in runtime.iter_audio_chunks(inputs, request_id=name, seed=42):
        if first is None:
            first = time.perf_counter() - t
        chunks.append(chunk.audio)
    total = time.perf_counter() - t
    import numpy as np
    audio = np.concatenate(chunks) if chunks else np.zeros(0)
    secs = len(audio) / runtime.sample_rate
    sf.write(OUT / f"{name}.wav", audio, runtime.sample_rate)
    r = {"case": name, "audio_s": round(secs, 2), "first_audio_s": round(first or 0, 2), "total_s": round(total, 2),
         "rtf": round(total / secs, 2) if secs else None, "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    results.append(r)
    print(json.dumps(r), flush=True)
print(json.dumps({"load_s": round(load_s, 1), "results": results}))
