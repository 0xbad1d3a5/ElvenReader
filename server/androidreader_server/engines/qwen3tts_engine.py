"""Qwen3-TTS adapter: 1.7B Base for cloning, 1.7B VoiceDesign for new voices."""
from __future__ import annotations

from .base import BaseEngine


class Engine(BaseEngine):
    sample_rate = 24000

    def _load_model(self, repo: str):
        import torch
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel.from_pretrained(
            repo,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    def load(self) -> None:
        self.designer = None
        self._prompts: dict[tuple, object] = {}
        if self.options.get("design_only"):
            # The voice designer next to the site's speech model: only the VoiceDesign half (about 4 GB).
            self.model = None
            self.designer = self._load_model(
                self.options.get("design_repo", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
            return
        self.model = self._load_model(self.options.get("repo", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))

    def _language(self, value: str | None) -> str:
        return self.language_name(value) or "Auto"

    def _prompt(self, voice: dict):
        key = (voice.get("ref_audio"), voice.get("ref_text"))
        if key not in self._prompts:
            self._prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=voice["ref_audio"],
                ref_text=voice.get("ref_text") or None,
                x_vector_only_mode=not voice.get("ref_text"),
            )
        return self._prompts[key]

    def synthesize(self, req: dict, cancelled):
        if self.model is None:
            raise RuntimeError("This Qwen3-TTS worker only designs voices")
        voice = req.get("voice") or {}
        seed = req.get("seed") if req.get("seed") is not None else voice.get("seed")
        self.seed_everything(seed)
        language = self._language(req.get("language") or voice.get("language"))
        kwargs = {k: self.options[k] for k in ("temperature", "top_p", "top_k") if k in self.options}
        try:
            wavs, sr = self.model.generate_voice_clone(
                text=req["text"], language=language, voice_clone_prompt=self._prompt(voice), **kwargs
            )
        except ValueError as exc:
            if "language" not in str(exc).lower():
                raise
            wavs, sr = self.model.generate_voice_clone(
                text=req["text"], language="Auto", voice_clone_prompt=self._prompt(voice), **kwargs
            )
        self.sample_rate = int(sr)
        yield wavs[0]

    def design_voice(self, description: str, text: str, seed):
        if self.designer is None:
            self.designer = self._load_model(
                self.options.get("design_repo", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
            )
        self.seed_everything(seed)
        wavs, sr = self.designer.generate_voice_design(text=text, instruct=description, language="Auto")
        self.sample_rate = int(sr)
        yield wavs[0]
