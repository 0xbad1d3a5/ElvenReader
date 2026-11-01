"""Text clean-up and sentence splitting applied before synthesis."""
from __future__ import annotations

import re
import unicodedata

MAX_TEXT_CHARS = 4000
# Quality models degrade (or run out of memory) on very long inputs, so longer
# requests are synthesized piece by piece and streamed back in order.
MAX_CHUNK_CHARS = 280

# Soft hyphens and zero-width characters are common in EPUB text and must not be spoken.
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
_WHITESPACE = re.compile(r"\s+")
# "I…I", "Um...yes", "H‥huh": an ellipsis run straight into the next word makes the models slur or skip the
# pause, so a space goes after it ("I… I").
_TIGHT_ELLIPSIS = re.compile(r"(\.{2,}|[\u2026\u2025]+)(?=\w)")


def clean_text(text: str) -> str:
    """Remove invisible/control characters, collapse whitespace and put a space after ellipses."""
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    text = "".join(
        ch if not unicodedata.category(ch).startswith("C") else " " for ch in text
    )
    text = _TIGHT_ELLIPSIS.sub(r"\1 ", text)
    return _WHITESPACE.sub(" ", text).strip()


def is_speakable(text: str) -> bool:
    """True when the text contains at least one letter or digit.

    Readers sometimes send lone punctuation ("...", "*") or whitespace; those
    produce no speech, so the server answers with silence instead.
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


_SPLIT_SENTENCE = re.compile(r"(?<=[.!?\u2026])[\"'\u201d\u2019\u00bb)\]]*\s+(?=\S)")
_SPLIT_CLAUSE = re.compile(r"(?<=[,;:\u2014])\s+")


def split_for_synthesis(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into pieces no longer than ``limit`` characters.

    Sentences are kept whole when they fit; over-long sentences are split at
    clause punctuation, then at spaces. Pieces are never empty.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    for sentence in _split_keep(text, _SPLIT_SENTENCE):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        for clause in _split_keep(sentence, _SPLIT_CLAUSE):
            if len(clause) <= limit:
                pieces.append(clause)
                continue
            words = clause.split(" ")
            buf = ""
            for w in words:
                if buf and len(buf) + 1 + len(w) > limit:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}" if buf else w
            if buf:
                pieces.append(buf)
    # Merge tiny neighbours back together so the model gets natural phrases.
    merged: list[str] = []
    for p in pieces:
        if merged and len(merged[-1]) + 1 + len(p) <= limit:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def _split_keep(text: str, pattern: re.Pattern) -> list[str]:
    out, start = [], 0
    for m in pattern.finditer(text):
        part = text[start : m.end()].strip()
        if part:
            out.append(part)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
