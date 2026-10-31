"""Fish Audio S2 Pro adapter: top-quality cloning with free-form inline tags.

A delivery such as "shouting, urgent" becomes an inline tag ``[shouting, urgent]``
in front of the text, which S2 Pro reads as a performance instruction.
"""
from __future__ import annotations

import os
import re

from .base import BaseEngine

_TAG_SAFE = re.compile(r"[\[\]<>|]")


def delivery_tag(delivery: str | None) -> str:
    d = _TAG_SAFE.sub("", (delivery or "")).strip()
    return f"[{d}] " if d else ""


class Engine(BaseEngine):
    sample_rate = 44100

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download

        # fish-speech resolves some resources relative to the project root.
        src = self.options.get("source_dir") or os.path.join(
            os.environ.get("ELVENREADER_MODELS", "."), "src", "fish-speech"
        )
        if os.path.isdir(src):
            os.chdir(src)

        from tools.server.model_manager import ModelManager
        import fish_speech.models.text2semantic.inference as fish_inference

        # The KV cache is allocated for the model's full 32k context, which needs
        # ~20 GB and spills out of a 16 GB card. One sentence plus a voice
        # reference fits easily in a few thousand tokens.
        max_seq_len = int(self.options.get("max_seq_len", 4096))
        original_init = fish_inference.init_model

        def init_model(*args, **kwargs):
            model, decode_one_token = original_init(*args, **kwargs)
            if hasattr(model, "config") and getattr(model.config, "max_seq_len", 0) > max_seq_len:
                model.config.max_seq_len = max_seq_len
                # The causal mask is built for the full context (1 GB); it is only
                # ever sliced up to the KV cache length.
                if getattr(model, "causal_mask", None) is not None and model.causal_mask.shape[-1] > max_seq_len:
                    model.causal_mask = model.causal_mask[:max_seq_len, :max_seq_len].clone()
            return model, decode_one_token

        fish_inference.init_model = init_model

        # The DAC codec loads in fp32 (~5 GB). fish-speech's own bf16 loader folds
        # weight norm first, halving that with no audible difference.
        import tools.server.model_manager as fish_manager

        if self.options.get("codec_bf16", True):
            fish_manager.load_decoder_model = lambda config_name, checkpoint_path, device: (
                fish_inference.load_codec_model(checkpoint_path, device, torch.bfloat16)
            )

        repo = self.options.get("repo", "fishaudio/s2-pro")
        checkpoint = snapshot_download(repo, local_files_only=bool(self.options.get("offline", True)))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.manager = ModelManager(
            mode="tts",
            device=device,
            half=False,
            compile=bool(self.options.get("compile", False)),
            llama_checkpoint_path=checkpoint,
            decoder_checkpoint_path=None,
            decoder_config_name="modded_dac_vq",
        )
        self.engine = self.manager.tts_inference_engine
        dm = self.engine.decoder_model
        self.sample_rate = int(getattr(getattr(dm, "spec_transform", None), "sample_rate", 0) or dm.sample_rate)
        self._ref_bytes: dict[str, bytes] = {}

    def _reference(self, voice: dict):
        from fish_speech.utils.schema import ServeReferenceAudio

        path = voice["ref_audio"]
        if path not in self._ref_bytes:
            with open(path, "rb") as f:
                self._ref_bytes[path] = f.read()
        return ServeReferenceAudio(audio=self._ref_bytes[path], text=voice.get("ref_text") or "")

    def synthesize(self, req: dict, cancelled):
        from fish_speech.utils.schema import ServeTTSRequest

        voice = req.get("voice") or {}
        seed = req.get("seed") if req.get("seed") is not None else voice.get("seed")
        text = delivery_tag(req.get("delivery")) + req["text"]
        request = ServeTTSRequest(
            text=text,
            references=[self._reference(voice)] if voice.get("ref_audio") else [],
            seed=int(seed) if seed is not None else None,
            use_memory_cache="on",
            streaming=True,
            normalize=True,
            format="pcm",
            temperature=float(self.options.get("temperature", 0.7)),
            top_p=float(self.options.get("top_p", 0.7)),
        )
        for result in self.engine.inference(request):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "segment" and result.audio is not None:
                sr, audio = result.audio
                self.sample_rate = int(sr)
                yield audio
                if cancelled():
                    return
