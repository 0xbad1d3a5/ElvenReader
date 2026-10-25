"""User-corrected pronunciations ("Hermione" -> "her-MY-oh-nee").

Entries are global or belong to one book; a book entry wins over a global
entry for the same term. They are applied on the server before synthesis, so
the reader and the system TTS engine (which has no book context and therefore
uses only global entries) both benefit.

A respelling works with every engine. An entry may also carry ARPAbet
phonemes (the CMU dictionary notation, e.g. ``HH ER0 M AY1 AH0 N IY0``);
engines that accept phoneme input use them, the others use the respelling.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# Possessives and plurals keep working: "Hermione's" -> "her-my-oh-nee's".
_SUFFIX = r"(?:['\u2019]s|s['\u2019]|['\u2019]|es|s)?"
_ARPABET = re.compile(r"^[A-Z]{1,2}[0-2]?(?: [A-Z]{1,2}[0-2]?)*$")


@dataclass
class Entry:
    term: str
    respelling: str
    phonemes: str = ""
    case_sensitive: bool = False
    book: str | None = None  # None = global
    id: str = field(default_factory=lambda: secrets.token_hex(5))
    updated: float = field(default_factory=time.time)

    @property
    def scope(self) -> str:
        return "book" if self.book else "global"

    def to_json(self) -> dict:
        d = asdict(self)
        d["scope"] = self.scope
        return d

    def key(self) -> str:
        return self.term if self.case_sensitive else self.term.casefold()


def normalize_phonemes(value: str) -> str:
    """Upper-case, single-spaced ARPAbet; empty if it does not look like ARPAbet."""
    s = " ".join(value.replace("[", " ").replace("]", " ").upper().split())
    return s if s and _ARPABET.match(s) else ""


def speakable_respelling(respelling: str) -> str:
    """Stress capitals help people read a respelling but some models spell
    capitalised syllables letter by letter, so they are lower-cased."""
    s = respelling.strip()
    if "-" in s or any(part.isupper() and len(part) > 1 for part in s.split()):
        s = s.lower()
    return s


def _term_pattern(term: str) -> str:
    words = [re.escape(w) for w in term.split()]
    return r"\s+".join(words)


def effective_entries(entries: Iterable[Entry], book: str | None) -> list[Entry]:
    """Global entries plus this book's entries; the book's win on the same term."""
    chosen: dict[str, Entry] = {}
    for e in entries:
        if e.book is None:
            chosen.setdefault(e.key(), e)
    for e in entries:
        if book and e.book == book:
            chosen[e.key()] = e
            # A case-insensitive book entry also overrides a case-sensitive global one.
            if not e.case_sensitive:
                for k in [k for k, v in chosen.items() if v.book is None and v.term.casefold() == e.key()]:
                    del chosen[k]
    return list(chosen.values())


PHONEME_FORMATS: dict[str, Callable[[str, str], str]] = {
    # OmniVoice: CMU phonemes in square brackets replace the word.
    "cmu_brackets": lambda word, ph: f"[{ph}]",
    # IndexTTS 2.5: <word|PHONEMES> annotation.
    "cmu_annotated": lambda word, ph: f"<{word}|{ph}>",
}


def apply(text: str, entries: list[Entry], phoneme_format: Callable[[str, str], str] | None = None) -> str:
    """Replace every whole-word occurrence of each entry's term.

    ``phoneme_format(word, phonemes)`` renders phonemes for engines that accept
    them; without it (or for entries without phonemes) the respelling is used.
    """
    if not entries or not text:
        return text
    ordered = sorted(entries, key=lambda e: len(e.term), reverse=True)
    parts = []
    for i, e in enumerate(ordered):
        body = _term_pattern(e.term)
        if not e.case_sensitive:
            body = f"(?i:{body})"
        parts.append(f"(?P<t{i}>{body})(?P<s{i}>{_SUFFIX})")
    pattern = re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w])")

    def repl(m: re.Match) -> str:
        for i, e in enumerate(ordered):
            if m.group(f"t{i}") is not None:
                suffix = m.group(f"s{i}") or ""
                if phoneme_format and e.phonemes and not suffix:
                    return phoneme_format(m.group(f"t{i}"), e.phonemes)
                spoken = speakable_respelling(e.respelling) or e.term
                return spoken + suffix
        return m.group(0)

    return pattern.sub(repl, text)


def entries_version(entries: list[Entry]) -> str:
    """Stable digest of the entries that affect a book; part of audio cache keys."""
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda e: (e.key(), e.id)):
        h.update(json.dumps([e.term, e.respelling, e.phonemes, e.case_sensitive], ensure_ascii=False).encode())
    return h.hexdigest()[:12]


class PronunciationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: list[Entry] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
            known = set(Entry.__dataclass_fields__)
            self._entries = [Entry(**{k: v for k, v in r.items() if k in known}) for r in raw.get("entries", [])]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"entries": [asdict(e) for e in self._entries]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def all(self) -> list[Entry]:
        return list(self._entries)

    def for_book(self, book: str | None) -> list[Entry]:
        return effective_entries(self._entries, book)

    def listing(self, book: str | None) -> dict:
        return {
            "global": [e.to_json() for e in self._entries if e.book is None],
            "book": [e.to_json() for e in self._entries if book and e.book == book],
            "version": entries_version(self.for_book(book)),
        }

    def add(self, term: str, respelling: str, *, phonemes: str = "", case_sensitive: bool = False,
            book: str | None = None) -> Entry:
        term = " ".join(term.split())
        if not term or not any(ch.isalnum() for ch in term):
            raise ValueError("term must contain a letter or digit")
        phonemes = normalize_phonemes(phonemes)
        if not respelling.strip() and not phonemes:
            raise ValueError("give a respelling or phonemes")
        entry = Entry(term=term, respelling=respelling.strip(), phonemes=phonemes,
                      case_sensitive=bool(case_sensitive), book=book or None)
        with self._lock:
            # Re-adding the same term in the same scope replaces the old entry.
            self._entries = [e for e in self._entries if not (e.book == entry.book and e.key() == entry.key())]
            self._entries.append(entry)
            self._save()
        return entry

    def update(self, entry_id: str, **changes) -> Entry | None:
        with self._lock:
            for e in self._entries:
                if e.id != entry_id:
                    continue
                if "term" in changes and str(changes["term"]).strip():
                    e.term = " ".join(str(changes["term"]).split())
                if "respelling" in changes:
                    e.respelling = str(changes["respelling"] or "").strip()
                if "phonemes" in changes:
                    e.phonemes = normalize_phonemes(str(changes["phonemes"] or ""))
                if "case_sensitive" in changes:
                    e.case_sensitive = bool(changes["case_sensitive"])
                e.updated = time.time()
                self._save()
                return e
        return None

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            if len(self._entries) != before:
                self._save()
                return True
        return False
