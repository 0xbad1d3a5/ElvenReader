"""IndexTTS 2.5 adapter: expressive cloning with emotion taken from a delivery text.

A delivery such as "shouting, urgent" is turned into an emotion vector by the
model's bundled Qwen emotion classifier (``emo_text``) and blended with the
speaker's own timbre. Without a delivery the speaker reference sets the tone.
"""
from __future__ import annotations

import os

from .base import BaseEngine

_LANG = {"en": "EN", "zh": "ZH", "ja": "JA", "es": "ES", "ar": "AR"}


class Engine(BaseEngine):
    sample_rate = 22050

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        model_dir = self.options.get("model_dir") or os.path.join(
            os.environ.get("ANDROIDREADER_MODELS", "."), "checkpoints", "IndexTTS-2.5"
        )
        model_dir = os.path.abspath(model_dir)
        if not os.path.exists(os.path.join(model_dir, "config.yaml")):
            snapshot_download(self.options.get("repo", "IndexTeam/IndexTTS-2.5"), local_dir=model_dir)
        # infer_v2_5 points HF_HUB_CACHE at ./checkpoints/hf_cache on import.
        os.chdir(os.path.dirname(model_dir))
        from indextts.infer_v2_5 import IndexTTS2

        self.tts = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_bf16=True,
            use_cuda_kernel=False,
            use_qwen_emo=True,
            use_torch_compile=bool(self.options.get("compile", False)),
        )

    def synthesize(self, req: dict, cancelled):
        voice = req.get("voice") or {}
        seed = req.get("seed") if req.get("seed") is not None else voice.get("seed")
        self.seed_everything(seed)
        code = self.language_code(req.get("language") or voice.get("language")) or "en"
        kwargs = {}
        delivery = (req.get("delivery") or "").strip()
        if delivery:
            kwargs.update(use_emo_text=True, emo_text=delivery,
                          emo_alpha=float(self.options.get("emo_alpha", 0.65)))
        result = self.tts.infer(
            spk_audio_prompt=voice["ref_audio"],
            text=req["text"],
            output_path=None,
            lang=_LANG.get(code, "EN"),
            use_random=False,
            verbose=False,
            **kwargs,
        )
        if result is None:
            raise RuntimeError("IndexTTS produced no audio")
        sr, wav = result
        self.sample_rate = int(sr)
        yield wav
