"""OmniVoice (k2-fsa) adapter: zero-shot cloning, 600+ languages, CMU phoneme overrides."""
from __future__ import annotations

from .base import BaseEngine

# OmniVoice has no free-form emotion control, only a few non-verbal sounds.
_NONVERBAL = (
    (("laugh", "giggl", "chuckl"), "[laughter]"),
    (("sigh",), "[sigh]"),
)


class Engine(BaseEngine):
    sample_rate = 24000

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            self.options.get("repo", "k2-fsa/OmniVoice"),
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )
        self.sample_rate = int(getattr(self.model, "sampling_rate", None) or 24000)
        self._prompts: dict[tuple, object] = {}

    def _gen_kwargs(self) -> dict:
        return {
            "num_step": int(self.options.get("num_step", 32)),
            "guidance_scale": float(self.options.get("guidance_scale", 2.0)),
        }

    def _prompt(self, voice: dict):
        key = (voice.get("ref_audio"), voice.get("ref_text"))
        if key not in self._prompts:
            self._prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=voice["ref_audio"], ref_text=voice.get("ref_text") or None
            )
        return self._prompts[key]

    def synthesize(self, req: dict, cancelled):
        voice = req.get("voice") or {}
        seed = req.get("seed") if req.get("seed") is not None else voice.get("seed")
        self.seed_everything(seed)
        text = req["text"]
        delivery = (req.get("delivery") or "").lower()
        for keys, tag in _NONVERBAL:
            if any(k in delivery for k in keys):
                text = f"{tag} {text}"
                break
        kwargs = self._gen_kwargs()
        language = self.language_name(req.get("language") or voice.get("language"))
        if language:
            kwargs["language"] = language
        if voice.get("ref_audio"):
            kwargs["voice_clone_prompt"] = self._prompt(voice)
        elif voice.get("instruct"):
            kwargs["instruct"] = voice["instruct"]
        audio = self.model.generate(text=text, **kwargs)
        yield audio[0]

    def design_voice(self, description: str, text: str, seed):
        self.seed_everything(seed)
        audio = self.model.generate(text=text, instruct=description, **self._gen_kwargs())
        yield audio[0]
