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


_CLOSERS = "\"'\u201d\u2019\u00bb)]}\u300d\u300f"
_SENTENCE_END = ".!?\u2026\u3002\uff01\uff1f"
_CLAUSE_END = ",;:\u2014\u2013\u3001\uff0c\uff1b\uff1a"


def ending_kind(text: str) -> str | None:
    """"sentence", "clause" or None, from the text's final punctuation (closing quotes ignored)."""
    t = text.rstrip().rstrip(_CLOSERS).rstrip()
    if not t:
        return None
    if t[-1] in _SENTENCE_END:
        return "sentence"
    if t[-1] in _CLAUSE_END or t.endswith("-"):
        return "clause"
    return None


def trailing_pause_ms(text: str, sentence_pause_ms: int) -> int:
    """Silence to append after an utterance so sentence boundaries breathe."""
    kind = ending_kind(text)
    if kind == "sentence":
        return sentence_pause_ms
    if kind == "clause":
        return sentence_pause_ms // 3
    return 0


def clamp_pause_ms(value: object, default: int) -> int:
    try:
        ms = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0, min(2000, ms))
