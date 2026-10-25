"""Base class for engine adapters. Runs inside a worker process."""
from __future__ import annotations

import random
from typing import Callable, Iterable, Iterator


class BaseEngine:
    """Adapter around one TTS model.

    ``synthesize`` receives a request dict::

        {"text": str, "voice": {"ref_audio", "ref_text", "language", "instruct", "seed", ...},
         "language": str | None, "seed": int | None, "delivery": str | None}

    and yields audio as PCM16LE bytes or float arrays in [-1, 1] at
    ``self.sample_rate``. It should check ``cancelled()`` between chunks when
    the model can stop early.
    """

    sample_rate = 24000

    def __init__(self, options: dict):
        self.options = dict(options or {})

    def load(self) -> None:  # pragma: no cover - model specific
        raise NotImplementedError

    def synthesize(self, req: dict, cancelled: Callable[[], bool]) -> Iterable:  # pragma: no cover
        raise NotImplementedError

    def design_voice(self, description: str, text: str, seed: int | None) -> Iterator:
        """Speak ``text`` in a new voice described by ``description`` (voice design)."""
        raise NotImplementedError("this engine cannot design voices")

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def seed_everything(seed: int | None) -> None:
        if seed is None:
            return
        seed = int(seed)
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    @staticmethod
    def device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def vram_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 2**20, 1)
        except ImportError:
            pass
        return 0.0

    @staticmethod
    def language_name(code_or_name: str | None) -> str | None:
        """"en"/"eng"/"English" -> "English"; None stays None."""
        if not code_or_name:
            return None
        s = code_or_name.strip()
        return _LANG_NAMES.get(s.lower(), s[:1].upper() + s[1:])

    @staticmethod
    def language_code(code_or_name: str | None) -> str | None:
        """"English"/"eng"/"en" -> "en"."""
        if not code_or_name:
            return None
        s = code_or_name.strip().lower()
        if len(s) == 2:
            return s
        for code, name in _CODE_NAMES.items():
            if s in (name.lower(), _ISO3.get(code, "")):
                return code
        return None


_CODE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "pl": "Polish", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "tr": "Turkish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "no": "Norwegian", "el": "Greek", "he": "Hebrew",
    "hi": "Hindi", "ms": "Malay", "sw": "Swahili", "cs": "Czech", "uk": "Ukrainian",
}
_ISO3 = {
    "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "nl": "nld",
    "ru": "rus", "pl": "pol", "zh": "zho", "ja": "jpn", "ko": "kor", "ar": "ara", "tr": "tur",
    "sv": "swe", "da": "dan", "fi": "fin", "no": "nor", "el": "ell", "he": "heb", "hi": "hin",
    "ms": "msa", "sw": "swa", "cs": "ces", "uk": "ukr",
}
_LANG_NAMES = {**{k: v for k, v in _CODE_NAMES.items()}, **{v3: _CODE_NAMES[k] for k, v3 in _ISO3.items()},
               **{v.lower(): v for v in _CODE_NAMES.values()}}
