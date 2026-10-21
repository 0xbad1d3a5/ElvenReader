"""Text clean-up applied before sending a reader's utterance to VoiceStudio."""
from __future__ import annotations

import re
import unicodedata

MAX_TEXT_CHARS = 4000

# Soft hyphens and zero-width characters are common in EPUB text and must not be spoken.
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove invisible/control characters and collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    text = "".join(
        ch if not unicodedata.category(ch).startswith("C") else " " for ch in text
    )
    return _WHITESPACE.sub(" ", text).strip()


def is_speakable(text: str) -> bool:
    """True when the text contains at least one letter or digit.

    Readers sometimes send lone punctuation ("...", "*") or whitespace; those
    produce no speech, so the relay answers with silence instead of asking
    VoiceStudio to synthesize them.
    """
    return any(ch.isalnum() for ch in text)


def clamp_speed(value: object, default: float = 1.0) -> float:
    try:
        speed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if speed != speed:  # NaN
        return default
    return max(0.5, min(2.5, speed))
