"""Breeze TTS 2 (BreezeBlue) adapter: cloning from a reference clip and its transcript, steered by a free-text
instruction ("voice direction": the line's delivery), and voice design from a description. English and Chinese,
24 kHz. Runs BreezeBlue's own streaming runtime (models/src/breeze-tts).

Measured on this PC (RTX 5080, Windows; the GPU's whole use, nvidia-smi):
* eager (`fast: "off"`): 9.0 GB, RTF 2.0-2.5: too slow to read with;
* every stage compiled and graphed (`fast: "all"`, the README's --fast-all): 15.9 GB, the whole GPU (its CUDA graphs
  for every text and prompt length), RTF 0.7-0.9;
* only the stages that run for every sound token compiled (`fast: "decode"`, default): backbone decode, depth
  decoder and codec; the text encoder and prefill, once a line, stay eager. 9.6 GB, RTF 0.41-0.50, 20 s of warm-up
  once the kernels are cached (models/torch/inductor; the very first time several minutes).
Windows needs TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0: PyTorch's static CUDA launcher passes GPU addresses as a C
long, 32 bits on Windows ("Python int too large to convert to C long"). Its licence is for research and
non-commercial use.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from .base import BaseEngine
from .voxcpm_styles import render as render_delivery

os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

DEFAULT_REPO = "BreezeBlue/Breeze-TTS-2"
CFG_PLAIN, CFG_DIRECTED = 1.0, 4.0  # the README: CFG 4 for instructions to come through
# Breeze reads a parenthesis as a vocal event ("(laugh)", "(sigh)"): asides in parentheses are read as dashes.
_PARENS = re.compile(r"\(([^()]*)\)")
_QUOTES = re.compile(r"[\"\u201c\u201d\u201e\u00ab\u00bb\u300c\u300d\u300e\u300f]")


def instruction_for(delivery: str | None) -> str:
    """A line's delivery tags ("very angry, shout") as an instruction ("Very angry, speaking loudly."); the same
    wording VoxCPM2 is given (voxcpm_styles)."""
    words, _sound = render_delivery(delivery)
    words = (words or (delivery or "")).strip().strip(".")
    return (words[:1].upper() + words[1:] + ".") if words else ""


def spoken(text: str) -> str:
    text = _PARENS.sub(lambda m: f"\u2014 {m.group(1).strip()} \u2014", text)
    return re.sub(r"\s{2,}", " ", _QUOTES.sub("", text)).strip()


class Engine(BaseEngine):
    sample_rate = 24000

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download

        src = self.options.get("source_dir") or os.path.join(os.environ.get("ELVENREADER_MODELS", "."), "src", "breeze-tts")
        if src not in sys.path:
            sys.path.insert(0, src)
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

        repo = self.options.get("repo", DEFAULT_REPO)
        try:
            path = snapshot_download(repo, local_files_only=True)
        except Exception:  # noqa: BLE001 - not downloaded yet
            path = snapshot_download(repo)
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(
            Path(path), device=resolve_device(), attn_implementation="eager")
        update_generation_config_for_breeze(self.model)
        fast = str(self.options.get("fast", "decode"))
        stages = {"fast_backbone_decode": True, "fast_depth_decoder": True, "fast_codec": True} if fast == "decode" else {}
        self.runtime = FastBreezeStreamingRuntime(self.model, self.audio_tokenizer, FastStreamingConfig(
            max_new_tokens=int(self.options.get("max_new_tokens", 1500)), max_seq_len=2048,
            fast_all=True if fast == "all" else None, repetition_penalty=1.1, **stages), tokenizer=self.tokenizer)
        if self.runtime.fast_enabled:
            from dataclasses import replace

            from models.warmup_profile import load_warmup_profile

            profile = load_warmup_profile(os.path.join(src, "configs", "fast.json"))
            self.runtime.warmup_from_profile(replace(profile, codec_chunk_frames=self.runtime.codec_chunk_frames))
        self.sample_rate = int(self.runtime.sample_rate)
        torch.cuda.reset_peak_memory_stats()

    def _speak(self, request: dict, cfg: float, seed: int, cancelled=lambda: False):
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name

        inputs = prepare_inputs(self.tokenizer, self.audio_tokenizer, self.model, [request],
                                get_template(select_template_name(request)), guidance_scale=cfg,
                                guidance_scale_ref=None, guidance_scale_ins=None)
        chunks = self.runtime.iter_audio_chunks(inputs, request_id=request["id"], seed=seed)
        try:
            for chunk in chunks:
                if cancelled():
                    return
                yield chunk.audio
        finally:
            chunks.close()

    def synthesize(self, req: dict, cancelled):
        voice = req.get("voice") or {}
        if not voice.get("ref_audio") or not (voice.get("ref_text") or "").strip():
            raise RuntimeError("Breeze TTS 2 needs the voice's reference clip and its transcript")
        seed = req.get("seed") if req.get("seed") is not None else voice.get("seed")
        seed = int(seed) if seed is not None else 42
        request = {"id": "line", "text": spoken(req["text"]), "speaker": "S0",
                   "ref_audio_path": voice["ref_audio"], "ref_text": voice["ref_text"].strip()}
        instruction = instruction_for(req.get("delivery"))
        if instruction:
            request["instruction"] = instruction
        yield from self._speak(request, CFG_DIRECTED if instruction else CFG_PLAIN, seed, cancelled)

    def design_voice(self, description: str, text: str, seed):
        request = {"id": "design", "text": spoken(text), "speaker": "S0", "instruction": description.strip()}
        yield from self._speak(request, CFG_DIRECTED, int(seed) if seed is not None else 42)
