"""OmniVoice (k2-fsa) adapter: zero-shot cloning, 600+ languages, CMU phoneme overrides."""
from __future__ import annotations

import re

from .base import BaseEngine

# OmniVoice has no free-form emotion control, only a few non-verbal sounds.
_NONVERBAL = (
    (("laugh", "giggl", "chuckl"), "[laughter]"),
    (("sigh",), "[sigh]"),
)

# OmniVoice designs voices only from a fixed set of tags ("male, low pitch, british accent").
_TAGS = {"american accent", "australian accent", "british accent", "canadian accent", "child", "chinese accent",
         "elderly", "female", "high pitch", "indian accent", "japanese accent", "korean accent", "low pitch", "male",
         "middle-aged", "moderate pitch", "portuguese accent", "russian accent", "teenager", "very high pitch",
         "very low pitch", "whisper", "young adult"}
_AGE_TAG = {"child": "child", "teen": "teenager", "young adult": "young adult", "middle-aged": "middle-aged",
            "elderly": "elderly"}
_ACCENTS = {"british": "british", "english accent": "british", "england": "british", "american": "american",
            "australian": "australian", "canadian": "canadian", "chinese": "chinese", "indian": "indian",
            "japanese": "japanese", "korean": "korean", "portuguese": "portuguese", "brazilian": "portuguese",
            "russian": "russian"}


def to_instruct(description: str) -> str:
    """A free-text voice description -> the OmniVoice tags it implies (gender, age, pitch, accent, whisper).
    Tag-only descriptions pass through; what has no tag (timbre, manner) can't be expressed and is dropped."""
    text = " ".join((description or "").split())
    items = [i.strip().lower() for i in text.split(",") if i.strip()]
    if items and all(i in _TAGS for i in items):
        return ", ".join(items)
    t = text.lower()
    tags = []
    female = re.search(r"\b(female|woman|women|girl|lady|she|her|mother|grandmother|queen|princess|maiden)\b", t)
    male = re.search(r"\b(male|man|men|boy|gentleman|he|his|father|grandfather|king|prince|lord)\b", t)
    if female and (not male or female.start() < male.start()):
        tags.append("female")
    elif male:
        tags.append("male")
    try:
        from androidreader_server.voice_age import current_age

        age = _AGE_TAG.get(current_age(text) or "")
    except Exception:  # noqa: BLE001 - age is a nicety
        age = None
    if age:
        tags.append(age)
    if re.search(r"\bvery (deep|low)\b|\bbass\b", t):
        tags.append("very low pitch")
    elif re.search(r"\bvery high\b|\bsqueaky\b|\bshrill\b", t):
        tags.append("very high pitch")
    elif re.search(r"\b(deep|low|low-pitched|baritone|gravelly|husky)\b", t):
        tags.append("low pitch")
    elif re.search(r"\b(high|high-pitched)\b", t):
        tags.append("high pitch")
    for word, accent in _ACCENTS.items():
        if word in t:
            tags.append(f"{accent} accent")
            break
    if re.search(r"\bwhisper", t):
        tags.append("whisper")
    return ", ".join(tags) or "moderate pitch"


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
            kwargs["instruct"] = to_instruct(voice["instruct"])
        audio = self.model.generate(text=text, **kwargs)
        yield audio[0]

    def design_voice(self, description: str, text: str, seed):
        self.seed_everything(seed)
        audio = self.model.generate(text=text, instruct=to_instruct(description), **self._gen_kwargs())
        yield audio[0]
