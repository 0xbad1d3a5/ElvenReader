"""Dramatization: who says each quotation in a book, and how.

A book's paragraphs are stored with the quotations found in them (epub.py:
stable IDs and offsets, so highlighting always matches the EPUB). A local LLM
then labels each quotation, chunk by chunk, with a speaker and a short delivery
("shouting, urgent"). The LLM never rewrites text: it only returns labels for
the IDs it was given, constrained by a JSON schema. Results are saved after
every chunk, so playback can use them while the analysis is still running.

A roster pass then describes the main characters (gender, age, accent,
temperament), the cast is sorted into tiers, and voices are created with a voice-design
model; extras share a small pool of voices. Users can rename, merge and re-voice
characters, and correct any line; corrections always win.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from aiohttp import ClientConnectionError, web

logger = logging.getLogger("androidreader.drama")

NARRATOR = "NARRATOR"
# In "auto" mode the LLM uses the GPU only after this long without speech requests.
GPU_IDLE_S = 120.0
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
AGES = ["child", "teen", "young adult", "adult", "middle-aged", "elderly", "unknown"]
GENDERS = ["male", "female", "unknown"]


# ----------------------------------------------------------------- quotes & chunks


def quote_regions(text: str) -> list[tuple[int, int]]:
    """Quotation ranges, including the quote marks (an unclosed quote runs to the paragraph end)."""
    pairs = {"\u201c": "\u201d", "\u00ab": "\u00bb", "\u201e": "\u201c", "\u2018": "\u2019", '"': '"'}
    out, open_at, closer = [], -1, ""
    for i, c in enumerate(text):
        if open_at < 0:
            close = pairs.get(c)
            if close is None:
                continue
            if c in "\u2018\"":
                prev = text[i - 1] if i > 0 else " "
                nxt = text[i + 1] if i + 1 < len(text) else " "
                if not ((prev.isspace() or prev in "([\u2014-") and not nxt.isspace()):
                    continue
            open_at, closer = i, close
        elif c == closer:
            if c in "\u2019\"":
                nxt = text[i + 1] if i + 1 < len(text) else " "
                if nxt.isalpha():
                    continue
            out.append((open_at, i + 1))
            open_at = -1
    if open_at >= 0:
        out.append((open_at, len(text)))
    return out


@dataclass
class Chunk:
    key: str  # "c3k1"
    chapter: int
    paragraphs: list[int]
    quotes: list[tuple[str, int, int, int]]  # (quote id, paragraph, start, end)
    tense: str | None = None  # narration tense, for books that switch narrators by tense


_POV_RE = re.compile(r"\bPOV\b|point of view|['\u2019]s side\b|\bside\s*[:\u2014-]", re.IGNORECASE)


def is_pov_marker(text: str) -> bool:
    """A short heading such as "(The Neighbor's POV)" or "Suiran's Side" that switches the first-person
    narrator. Ordinary sentences that merely mention someone's side ("...into Suiran's side.") are not."""
    t = text.strip()
    if len(t) > 60 or not _POV_RE.search(t):
        return False
    if re.search(r"\bPOV\b|point of view", t, re.IGNORECASE):
        return True
    return t[:1] in "([<\u3010\u300c" or t[-1:] not in ".!?\u2026\u201d\"'\u2019"


_MENTION_SPEECH = re.compile(
    r"\b(?:said|says|say|saying|asked|asks|replied|replies|answered|cried|cries|shouted|shouts|whispered|muttered|"
    r"murmured|exclaimed|added|went|go|goes|sang|sing|sings|singing|called|calls|yelled|screamed|began|continued|"
    r"repeated|remarked|uttered|read|reads|wrote|tell|told)(?:\s+to\s+\w+|\s+\w+ly)?$", re.IGNORECASE)


def is_mention(text: str, start: int, end: int) -> bool:
    """A word or phrase in quotation marks inside a sentence (scare quotes, a term, a name: 'Chinese for
    "cat."', 'all the "inspections" we did'), not dialogue. It directly follows a word, is short, has no
    inner punctuation, and is not introduced by a speech verb (old-style 'then said "The fourth."')."""
    before = text[:start].rstrip()
    if not before or not before[-1].isalnum():
        return False
    inner = text[start:end].strip("\u201c\u201d\"\u2018\u2019\u00ab\u00bb\u300c\u300d ").rstrip(".,")
    if not inner or len(inner.split()) > 6 or re.search(r"[,!?;:\u2014\u2013\u2026]|\.\S", inner):
        return False
    return not _MENTION_SPEECH.search(before)


def mark_mentions(book: dict, script: dict) -> int:
    """Backstop for analysed books: quoted terms ("cat.", "inspections") are narration. Corrections still win."""
    n = 0
    for ch in book.get("chapters", []):
        for qid, p, s, e in ch.get("quotes", []):
            if is_mention(ch["paragraphs"][int(p)], int(s), int(e)):
                line = script["lines"].get(qid) or {}
                if line.get("speaker") is not None or not line.get("mention"):
                    n += 1
                script["lines"][qid] = {"speaker": None, "delivery": None, "mention": True}
    script["mentions_v"] = 1
    return n


def make_chunks(book: dict, max_words: int = 600, runs: list[tuple[int, int, str]] | None = None) -> list[Chunk]:
    """Chunks of about `max_words` with quotations; `runs` (from mixed_tense_runs) adds tense breaks."""
    switches = {(c, p): t for c, p, t in runs or []}
    tense = None
    chunks: list[Chunk] = []
    for ci, ch in enumerate(book.get("chapters", [])):
        paras: list[str] = ch.get("paragraphs", [])
        by_para: dict[int, list] = {}
        for q in ch.get("quotes", []):
            qid, p, s, e = q
            by_para.setdefault(int(p), []).append((str(qid), int(p), int(s), int(e)))
        current: list[int] = []
        current_tense = tense
        words = 0
        k = 0

        def flush():
            nonlocal current, words, k
            if any(p in by_para for p in current):
                chunks.append(Chunk(f"c{ci}k{k}", ci, current, [q for p in current for q in by_para.get(p, [])],
                                    current_tense))
                k += 1
            current, words = [], 0

        for pi, text in enumerate(paras):
            # A point-of-view change starts a new chunk, so each chunk has one narrator.
            if current and (is_pov_marker(text) or (ci, pi) in switches):
                flush()
            tense = switches.get((ci, pi), tense)
            if not current:
                current_tense = tense
            current.append(pi)
            words += len(text.split())
            if words >= max_words:
                flush()
        if current:
            flush()
    return chunks


def pov_change(book: dict, chunk: Chunk, use_title: bool = True) -> str | None:
    """The point-of-view heading this chunk starts with (or its chapter title), if any."""
    chapter = book["chapters"][chunk.chapter]
    first = chapter["paragraphs"][chunk.paragraphs[0]]
    if is_pov_marker(first):
        return first.strip()
    if use_title and chunk.paragraphs[0] == 0 and is_pov_marker(chapter.get("title", "")):
        return chapter["title"].strip()
    return None


_CONTINUES = re.compile(r"[,;:\u2014\u2013-]\s*$")


def render_chunk(book: dict, chunk: Chunk) -> tuple[str, dict[str, list[str]]]:
    """Chunk text with a number before each quotation; returns (text, number -> quote ids).

    Both halves of an interrupted quotation ("Yes," he said, "I will.") get the same
    number, so the model labels speaker turns rather than fragments.
    """
    paras = book["chapters"][chunk.chapter]["paragraphs"]
    local: dict[str, list[str]] = {}
    out = []
    by_para: dict[int, list] = {}
    for q in chunk.quotes:
        by_para.setdefault(q[1], []).append(q)
    n = 0
    for pi in chunk.paragraphs:
        text = paras[pi]
        pieces, last, prev_end = [], 0, None
        for qid, _, s, e in sorted(by_para.get(pi, []), key=lambda q: q[2]):
            between = text[prev_end:s] if prev_end is not None else None
            if between is None or len(between) > 80 or not _CONTINUES.search(between):
                n += 1
                local[str(n)] = []
            local[str(n)].append(qid)
            pieces.append(text[last:s])
            pieces.append(f"[{n}]{text[s:e]}")
            last = prev_end = e
        pieces.append(text[last:])
        out.append("".join(pieces))
    return "\n\n".join(out), local


# ----------------------------------------------------------------- names


def norm_name(name: str) -> str:
    s = name.lower().replace(".", "").strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return " ".join(s.split())


def slug(name: str) -> str:
    return _SLUG_RE.sub("-", norm_name(name)).strip("-")[:40] or "someone"


# ----------------------------------------------------------------- prompts

SYSTEM_PROMPT = """You label dialogue in novels for a multi-voice audiobook.
For every numbered quotation [n] in the passage, decide:
- speaker: who says it, as one name without parentheses. Use the exact roster name for known characters, even when the passage calls them by a nickname or title the roster lists under "also called". If a new character speaks, use the name the text uses for them (e.g. "the Dormouse", "Mrs. Long"); if they are known by a real name and a nickname, give the real name and put the nickname in their aliases. If an unnamed person speaks, describe them briefly ("a footman"). A character's quoted thoughts ("..." thought Alice) belong to that character. Use "NARRATOR" only for quotations nobody speaks or thinks aloud at that moment: titles, signs, written notes, and words or phrases set in quotation marks inside a sentence (a term, a nickname, scare quotes: the so-called "experts"; what she calls "fun"; French for "dog."; his "plan" failed). Such quoted terms are part of the narration and are NARRATOR even in first-person passages and even when the text says who coined the word.
- delivery: how an actor should perform the line, 1-4 lowercase words such as "shouting, urgent", "whispering", "sarcastic", "tearful", "excited", "nervous", "cold, angry", "teasing", "calm". Use "neutral" when nothing special is called for. Base it on speech verbs and adverbs near the quote ("she shrieked", "he said softly"), punctuation and context.
Rules: in a back-and-forth conversation without speech tags, speakers usually alternate. A number appearing twice marks one quotation interrupted by narration ("Yes," [3] he said, [3] "I will."): label it once. Never change the text; label each number once, in order, giving the first two or three words of its quotation in "starts" so you keep your place. If several characters speak together, give the main one.
First-person narration: when the story is told by "I", every line the narrator speaks ("...," I said / I asked / I replied, or "I" answering in a conversation) belongs to the narrating character, so give that character's name (never "I", and never another character). Work out who "I" is from the text and the narrator given for the previous passage; chapter titles such as "(Mary's POV)" mean that character narrates. Report the narrator in "narrator" (their name, or "third person" if the passage is not told by "I").
Also list every character who speaks in this passage with their gender, age group and any other names, titles or nicknames the passage uses for them (aliases)."""


def chunk_schema(local_ids: list[str], revisable: list[str] | None = None, profiles: bool = False) -> dict:
    schema = _chunk_schema(local_ids)
    if profiles:
        from .profile import UPDATES_SCHEMA

        schema["properties"]["profile_updates"] = UPDATES_SCHEMA
        schema["required"] = [*schema["required"], "profile_updates"]  # optional fields get skipped; [] is fine
    if revisable:
        # Earlier lines the passage shows were given to the wrong speaker (see label_chunk's revise).
        schema["properties"]["revisions"] = {
            "type": "array", "maxItems": len(revisable),
            "items": {"type": "object",
                      "properties": {"id": {"type": "string", "enum": revisable},
                                     # the earlier line's first words: a revision aimed at the wrong line is dropped
                                     "starts": {"type": "string", "maxLength": 40},
                                     "speaker": {"type": "string"},
                                     "reason": {"type": "string", "maxLength": 160}},
                      "required": ["id", "starts", "speaker", "reason"]},
        }
        schema["required"] = ["narrator", "quotes", "revisions", "speakers"] + (["profile_updates"] if profiles else [])
        schema["properties"] = {k: schema["properties"][k] for k in ("narrator", "quotes", "revisions", "speakers",
                                                                     "profile_updates") if k in schema["properties"]}
    return schema


def _chunk_schema(local_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "narrator": {"type": "string", "maxLength": 60},
            "quotes": {
                "type": "array",
                "minItems": len(local_ids),
                "maxItems": len(local_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": local_ids},
                        # Echoing the opening words keeps long lists aligned with their ids.
                        "starts": {"type": "string", "maxLength": 40},
                        "speaker": {"type": "string"},
                        "delivery": {"type": "string"},
                    },
                    "required": ["id", "starts", "speaker", "delivery"],
                },
            },
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                        "gender": {"type": "string", "enum": GENDERS},
                        "age": {"type": "string", "enum": AGES},
                    },
                    "required": ["name", "aliases", "gender", "age"],
                },
            },
        },
        "required": ["narrator", "quotes", "speakers"],
    }


ROSTER_PROMPT = """You are casting voices for an audiobook. For each character, using their lines and what the book says about how they look, give:
- appearance: one short sentence on how the character looks and comes across to the people around them, based on the "Described as" lines (apparent age, body, form). Note when their true age or nature differs ("looks like a young girl but is centuries old").
- gender and age group: as they APPEAR and as a listener would expect them to sound, not their true age. A centuries-old being in a little girl's body is "child"; a teenager is "teen".
- accent (e.g. "british", "american", "scottish", "neutral"), temperament (2-4 words).
- voice: one sentence for a voice-design model covering gender, apparent age, pitch, pace, timbre and manner. Follow the apparent age and physical form; put the true age and personality into the manner, e.g. "A young girl's high, light voice, speaking in an old-fashioned, knowing and teasing way." If the book describes the voice itself, follow that. Keep each character distinct from the others.
If you recognise the book or its series, also use what you know about how each character looks and sounds; later volumes often stop describing the regular cast."""

ROSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "appearance": {"type": "string"},
                    "gender": {"type": "string", "enum": GENDERS},
                    "age": {"type": "string", "enum": AGES},
                    "accent": {"type": "string"},
                    "temperament": {"type": "string"},
                    "voice": {"type": "string"},
                },
                "required": ["name", "appearance", "gender", "age", "accent", "temperament", "voice"],
            },
        }
    },
    "required": ["characters"],
}

_LOOKS_STRONG = re.compile(
    r"\b(?:girls?|boys?|child(?:ren)?|kids?|little|young\w*|youth\w*|teen\w*|old|older|elder\w*|aged|"
    r"years? old|grand(?:mother|father|ma|pa)|granny|middle[- ]school\w*|high[- ]school\w*|grade|student|"
    r"petite|tiny|tall|beard\w*|wrinkl\w*|birds?|sparrows?|robot\w*|android|alien|demon|angel|voice|"
    r"look(?:s|ed)? like|appearance|kimono-clad|uniform)\b", re.IGNORECASE)
_LOOKS_WEAK = re.compile(r"\b(?:wom[ae]n|m[ae]n|gentlem[ae]n|daughters?|sons?|sisters?|brothers?|small|short|"
                         r"looked|appear\w*|seem\w*|body)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?\u201d])\s+")


def appearance_evidence(book: dict, names: list[str], limit: int = 8, width: int = 240) -> list[str]:
    """Narration sentences that mention a character together with appearance or age words.

    Dialogue is left out (it is mostly characters addressing each other); the best
    `limit` sentences are returned in book order.
    """
    pats = [re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE) for n in names if len(n) > 2]
    if not pats:
        return []
    found: list[tuple[int, int, str]] = []
    seen = set()
    for ch in book.get("chapters", []):
        for text in ch.get("paragraphs", []):
            if not any(p.search(text) for p in pats):
                continue
            narration = text
            for s, e in sorted(quote_regions(text), reverse=True):
                narration = narration[:s] + "\u2026" + narration[e:]
            for sent in _SENTENCE.split(narration):
                sent = sent.strip()
                if len(sent) < 15 or sent in seen or not any(p.search(sent) for p in pats):
                    continue
                cues = 2 * len(_LOOKS_STRONG.findall(sent)) + len(_LOOKS_WEAK.findall(sent))
                if cues >= 2:
                    seen.add(sent)
                    found.append((cues, len(found), sent[:width]))
    best = sorted(found, key=lambda f: (-f[0], f[1]))[:limit]
    return [s for _, _, s in sorted(best, key=lambda f: f[1])]


# ----------------------------------------------------------------- script state


def book_summary(book: dict) -> dict:
    chapters = book.get("chapters", [])
    return {
        "title": book.get("title", ""),
        "author": book.get("author", ""),
        "language": book.get("language"),
        "chapters": len(chapters),
        "quotes": sum(len(c.get("quotes", [])) for c in chapters),
        "words": sum(len(p.split()) for c in chapters for p in c.get("paragraphs", [])),
    }


PICTURE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
                 ".webp": "image/webp", ".bmp": "image/bmp"}


def picture_refs(book: dict) -> list[str]:
    """Every picture the book shows (zip paths in its EPUB), in reading order."""
    out: list[str] = []
    for ch in book.get("chapters", []):
        for _, paths in sorted((ch.get("images") or {}).items(), key=lambda kv: int(kv[0])):
            out += paths
        out += ch.get("end_images") or []
    return list(dict.fromkeys(out))


class ScriptStore:
    """data/books/<id>/{book.json, script.json}."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.locks: dict[str, asyncio.Lock] = {}

    def lock(self, book_id: str) -> asyncio.Lock:
        return self.locks.setdefault(book_id, asyncio.Lock())

    def dir(self, book_id: str) -> Path:
        if not _ID_RE.match(book_id):
            raise ValueError("bad book id")
        return self.root / book_id

    def save_book(self, book_id: str, payload: dict) -> None:
        d = self.dir(book_id)
        d.mkdir(parents=True, exist_ok=True)
        cover = payload.get("cover")
        payload = {k: payload[k] for k in ("title", "author", "language", "chapters") if k in payload}
        (d / "book.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if isinstance(cover, str) and cover:
            try:
                (d / "cover.jpg").write_bytes(base64.b64decode(cover))
            except ValueError:
                pass
        meta_path = d / "meta.json"
        keep = {}
        if meta_path.exists():
            try:
                keep = {k: v for k, v in json.loads(meta_path.read_text(encoding="utf-8")).items() if k == "series"}
            except ValueError:
                pass
        meta_path.write_text(json.dumps(book_summary(payload) | {"uploaded": time.time()} | keep,
                                        ensure_ascii=False), encoding="utf-8")

    def summary(self, book_id: str) -> dict | None:
        d = self.dir(book_id)
        if (d / "meta.json").exists():
            return json.loads((d / "meta.json").read_text(encoding="utf-8"))
        book = self.book(book_id)
        return book_summary(book) if book else None

    def ids(self) -> list[str]:
        return [d.name for d in self.root.iterdir() if (d / "book.json").exists() and _ID_RE.match(d.name)]

    def cover(self, book_id: str) -> Path | None:
        p = self.dir(book_id) / "cover.jpg"
        return p if p.exists() else None

    def picture_file(self, book_id: str, path: str) -> Path | None:
        """Where a picture of the book (its zip path in the EPUB) is stored; None for unsupported types."""
        ext = Path(path).suffix.lower()
        if ext not in PICTURE_TYPES:
            return None
        return self.dir(book_id) / "pictures" / (hashlib.sha1(path.encode("utf-8")).hexdigest()[:20] + ext)

    def book(self, book_id: str) -> dict | None:
        p = self.dir(book_id) / "book.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def script(self, book_id: str) -> dict:
        p = self.dir(book_id) / "script.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"status": "none", "progress": 0.0, "detail": "", "characters": {}, "lines": {},
                "corrections": {}, "chunks_done": [], "model": ""}

    def save_script(self, book_id: str, script: dict) -> None:
        d = self.dir(book_id)
        d.mkdir(parents=True, exist_ok=True)
        script["updated"] = time.time()
        tmp = d / "script.tmp"
        tmp.write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(d / "script.json")

    @staticmethod
    def public(script: dict) -> dict:
        """What clients see: lines with corrections applied, characters with line counts."""
        lines = {}
        for qid, line in script.get("lines", {}).items():
            lines[qid] = {"speaker": line.get("speaker"), "delivery": line.get("delivery"), "corrected": False}
            if line.get("mention") and line.get("speaker") is None:
                lines[qid]["mention"] = True  # a quoted term inside narration, not dialogue
            if line.get("revised"):
                lines[qid]["revised"] = True  # changed by a later passage's evidence
        for qid, fix in script.get("corrections", {}).items():
            base = lines.get(qid, {"speaker": None, "delivery": None})
            lines[qid] = {
                "speaker": fix.get("speaker", base["speaker"]),
                "delivery": fix.get("delivery", base["delivery"]),
                "corrected": True,
            }
        counts: dict[str, int] = {}
        for line in lines.values():
            if line["speaker"]:
                counts[line["speaker"]] = counts.get(line["speaker"], 0) + 1
        chars = [
            {**c, "id": cid, "lines": counts.get(cid, 0)}
            for cid, c in script.get("characters", {}).items()
        ]
        chars.sort(key=lambda c: -c["lines"])
        return {
            "status": script.get("status", "none"),
            "progress": script.get("progress", 0.0),
            "detail": script.get("detail", ""),
            "model": script.get("model", ""),
            "characters": chars,
            "lines": lines,
            # [chapter, first paragraph, character id or None]: who narrates from there on.
            "narration": script.get("narration", []),
            # Sorting the cast into tiers: {"running": true} / {"done": true, ...} / {"error": "..."}.
            "roles_state": script.get("roles_state"),
        }


_SAY = (r"(?:said|asked|replied|answered|attempted|told|added|muttered|whispered|shouted|called|responded|"
        r"continued|explained|offered|admitted|insisted|murmured|went on|yelled|cried|mumbled|remarked|"
        r"suggested|ventured|managed|blurted|countered|pressed|repeated|say|ask|reply|answer|add|mutter|whisper|"
        r"shout|call|respond|continue|explain|offer|admit|insist|murmur|yell|cry|mumble|remark|tell)")
_I_SAID_AFTER = re.compile(rf"^[\s,;:\u2014\u2013-]*(?:\w+ly\s+)?I\s+(?:\w+ly\s+)?{_SAY}\b")
_I_SAID_BEFORE = re.compile(rf"\bI\s+(?:\w+ly\s+)?{_SAY}(?:\s+\w+)?[,:]?\s*$")


def compute_narration(book: dict, script: dict) -> list[list]:
    """Who narrates each section, for first-person books.

    Evidence is deterministic: the character that quotations tagged "…," I said
    are attributed to. Sections are split at chapters and point-of-view headings;
    a section without evidence keeps the book's main narrator, unless it starts
    with a point-of-view change (then it is left to the book's narrator voice).
    Third-person books (little or no evidence) get an empty list. Books with two
    first-person narrators in different tenses are split by narration tense.
    """
    lines = ScriptStore.public(script)["lines"]
    narrator = narrators_by_tense(book, lines)
    if narrator is not None:
        return tense_narration(book, narrator)
    sections: list[tuple[int, int, dict[str, int], bool]] = []
    for ci, ch in enumerate(book.get("chapters", [])):
        paras = ch.get("paragraphs", [])
        by_para: dict[int, list] = {}
        for qid, p, s, e in ch.get("quotes", []):
            by_para.setdefault(int(p), []).append((qid, int(s), int(e)))
        start, votes = 0, {}
        pov = is_pov_marker(ch.get("title", ""))
        for pi, text in enumerate(paras):
            if pi > 0 and is_pov_marker(text):
                sections.append((ci, start, votes, pov))
                start, votes, pov = pi, {}, True
            for qid, s, e in by_para.get(pi, []):
                speaker = (lines.get(qid) or {}).get("speaker")
                if speaker and (_I_SAID_AFTER.match(text[e:e + 60]) or _I_SAID_BEFORE.search(text[max(0, s - 60):s])):
                    votes[speaker] = votes.get(speaker, 0) + 1
        sections.append((ci, start, votes, pov))
    totals: dict[str, int] = {}
    for _, _, votes, _ in sections:
        for k, v in votes.items():
            totals[k] = totals.get(k, 0) + v
    if sum(totals.values()) < 3:
        return []
    main = max(totals, key=totals.get)
    out: list[list] = []
    for ci, start, votes, pov in sections:
        who = max(votes, key=votes.get) if votes else (None if pov else main)
        if not out or out[-1][2] != who:
            out.append([ci, start, who])
    return out


_PRESENT_SAY = {"say", "ask", "reply", "answer", "add", "mutter", "whisper", "shout", "call", "respond", "continue",
                "explain", "offer", "admit", "insist", "murmur", "yell", "cry", "mumble", "remark", "tell"}


def first_person_tags(book: dict) -> list[tuple[str, str]]:
    """(quote id, "past" or "present") for quotations tagged '…,' I said / I say."""
    out = []
    for ch in book.get("chapters", []):
        paras = ch.get("paragraphs", [])
        for qid, p, s, e in ch.get("quotes", []):
            text = paras[int(p)]
            m = _I_SAID_AFTER.match(text[int(e):int(e) + 60]) or _I_SAID_BEFORE.search(text[max(0, int(s) - 60):int(s)])
            if m:
                verb = re.search(r"\bI\s+(?:\w+ly\s+)?(\w+)", m.group(0)).group(1)
                out.append((qid, "present" if verb in _PRESENT_SAY else "past"))
    return out


def narrators_by_tense(book: dict, lines: dict) -> dict[str, str] | None:
    """{"past": cid, "present": cid} when two first-person narrators use different tenses.

    Some books alternate two first-person narrators who narrate in different
    tenses (the hero in the past, a point-of-view character in the present).
    Requires the '…,' I said tags to split cleanly: the main tense's top speaker
    differs from the other tense's, who holds most of that tense's tags.
    """
    by_tense: dict[str, dict[str, int]] = {"past": {}, "present": {}}
    for qid, tense in first_person_tags(book):
        who = (lines.get(qid) or {}).get("speaker")
        if who:
            by_tense[tense][who] = by_tense[tense].get(who, 0) + 1
    main_tense = max(by_tense, key=lambda t: sum(by_tense[t].values()))
    other = "present" if main_tense == "past" else "past"
    if sum(by_tense[other].values()) < 3 or not by_tense[main_tense]:
        return None
    narrator = {t: max(v, key=v.get) for t, v in by_tense.items()}
    if narrator[main_tense] == narrator[other] or by_tense[other][narrator[other]] < 0.6 * sum(by_tense[other].values()):
        return None
    return narrator


def reconcile_narrators(book: dict, script: dict) -> int:
    """Fix '…,' I said lines given to the wrong first-person narrator, using tense.

    The LLM tends to keep the previous narrator after a point-of-view section
    ends. In books whose two narrators use different tenses, each tagged line
    goes to the narrator of its tense. Other books are left alone.
    """
    narrator = narrators_by_tense(book, script["lines"])
    if narrator is None:
        return 0
    pair, changed = set(narrator.values()), 0
    where: dict[str, tuple[int, int]] = {}
    same_paragraph: dict[tuple[int, int], list[str]] = {}
    for ci, ch in enumerate(book.get("chapters", [])):
        for qid, p, *_ in ch.get("quotes", []):
            where[qid] = (ci, int(p))
            same_paragraph.setdefault((ci, int(p)), []).append(qid)
    for qid, tense in first_person_tags(book):
        line = script["lines"].get(qid)
        if line and line.get("speaker") in pair and line["speaker"] != narrator[tense]:
            wrong = line["speaker"]
            # the rest of an interrupted quotation ("…," I said. "…") is the same speaker
            for other in same_paragraph[where[qid]]:
                o = script["lines"].get(other)
                if o and o.get("speaker") == wrong:
                    o["speaker"] = narrator[tense]
                    changed += 1
    return changed


_PAST_WORDS = re.compile(r"\b(?:was|were|had|did|wasn't|weren't|hadn't|didn't|said|asked|thought|felt|seemed|looked|"
                         r"went|came|knew|saw|got|began|took|made|told|replied|answered|turned|nodded)\b", re.I)
_PRESENT_WORDS = re.compile(r"\b(?:is|are|am|has|does|isn't|aren't|doesn't|hasn't|I'm|we're|they're|it's|that's|"
                            r"there's|she's|he's|says|asks|seems|looks|thinks|knows|comes|goes|gets|takes|makes|"
                            r"nods|turns|replies)\b", re.I)


def _exact_switch(raw: list[int], labels: list, guess: int, before: str, after: str, lo: int, hi: int,
                  reach: int = 10) -> int:
    """Move a smoothed tense switch to the paragraph where the narration's tense really changes.

    Picks the split with the fewest paragraphs on the wrong side (narration that clearly uses the other
    tense); among equally good splits (dialogue-only paragraphs in between) the one just before the first
    paragraph in the new tense. A chapter start between the old narration and that point wins.
    """
    sign = {"present": 1, "past": -1}
    a, b = max(lo, guess - reach), min(hi, guess + reach)
    if a > b:
        return guess
    costs = {k: sum(1 for i in range(a, k) if raw[i] * sign[before] < 0) +
                sum(1 for i in range(k, b + 1) if raw[i] * sign[after] < 0) for k in range(a, b + 1)}
    low = min(costs.values())
    # the best splits form a run ending at the first paragraph in the new tense: take that one
    best = max(k for k, v in costs.items() if v == low)
    last_old = max((i for i in range(a, best) if raw[i] * sign[before] > 0), default=a - 1)
    for i in range(last_old + 1, best + 1):
        if labels[i][1] == 0:
            return i
    return best


def tense_runs(book: dict, window: int = 4, min_run: int = 12) -> list[tuple[int, int, str]]:
    """Where the narration switches between past and present tense: [(chapter, paragraph, tense)].

    Scores each paragraph by present- minus past-tense markers (dialogue removed),
    smooths the score over neighbouring paragraphs, absorbs runs shorter than
    `min_run` paragraphs (inner thoughts, system messages) and snaps switches to
    nearby point-of-view headings.
    """
    labels: list[tuple[int, int, str | None]] = []
    raw: list[int] = []  # per-paragraph score, same order as labels
    markers: set[tuple[int, int]] = set()
    current = None
    for ci, ch in enumerate(book.get("chapters", [])):
        paras = ch.get("paragraphs", [])
        scores = []
        for pi, text in enumerate(paras):
            if is_pov_marker(text):
                markers.add((ci, pi))
            t = text
            for s, e in sorted(quote_regions(text), reverse=True):
                t = t[:s] + " " + t[e:]
            t = t.replace("\u2019", "'")
            scores.append(len(_PRESENT_WORDS.findall(t)) - len(_PAST_WORDS.findall(t)))
        raw.extend(scores)
        for pi in range(len(paras)):
            v = sum(scores[max(0, pi - window): pi + window + 1])
            current = "present" if v > 0 else "past" if v < 0 else current
            labels.append((ci, pi, current))
    runs: list[list] = []  # [index into labels, tense]
    for i, (_, _, tense) in enumerate(labels):
        if tense and (not runs or runs[-1][1] != tense):
            runs.append([i, tense])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k in range(1, len(runs)):
            end = runs[k + 1][0] if k + 1 < len(runs) else len(labels)
            if end - runs[k][0] < min_run:
                del runs[k]
                changed = True
                break
        merged = [runs[0]]
        for r in runs[1:]:
            if r[1] != merged[-1][1]:
                merged.append(r)
        runs = merged
    position = {(c, p): i for i, (c, p, _) in enumerate(labels)}
    for k, r in enumerate(runs[1:], start=1):
        c, p, _ = labels[r[0]]
        near = [position[m] for m in markers if m[0] == c and abs(m[1] - p) <= 8]
        if near:
            r[0] = min(near, key=lambda i: abs(i - r[0]))
        else:
            r[0] = _exact_switch(raw, labels, r[0], runs[k - 1][1], r[1],
                                 lo=runs[k - 1][0] + 1, hi=runs[k + 1][0] - 1 if k + 1 < len(runs) else len(labels) - 1)
    first_story = next((ci for ci, ch in enumerate(book.get("chapters", [])) if ch.get("quotes")), 0)
    if len(runs) > 1 and labels[runs[1][0]][0] <= first_story and labels[runs[1][0]][1] == 0:
        del runs[0]  # front matter (copyright pages etc.) goes with the first story narrator
    out = []
    for i, tense in runs:
        c, p, _ = labels[i]
        out.append((c, p, tense) if out else (0, 0, tense))
    return out


def tense_narration(book: dict, narrator: dict[str, str]) -> list[list]:
    return [[c, p, narrator[t]] for c, p, t in tense_runs(book)]


def mixed_tense_runs(book: dict, share: float = 0.05) -> list[tuple[int, int, str]] | None:
    """The tense runs, if the narration alternates between the past and the present tense.

    Needs both tenses to recur (at least two runs each) and cover `share` of the
    paragraphs, so present-tense back matter doesn't count.
    """
    runs = tense_runs(book)
    if min(sum(1 for r in runs if r[2] == t) for t in ("past", "present")) < 2:
        return None
    sizes = {"past": 0, "present": 0}
    chapters = book.get("chapters", [])
    flat = [(ci, pi) for ci, ch in enumerate(chapters) for pi in range(len(ch.get("paragraphs", [])))]
    index = {k: i for i, k in enumerate(flat)}
    for k, (c, p, t) in enumerate(runs):
        end = index[runs[k + 1][:2]] if k + 1 < len(runs) else len(flat)
        sizes[t] += end - index[(c, p)]
    return runs if min(sizes.values()) >= share * max(1, len(flat)) else None


def refresh_narration(store: "ScriptStore", book_id: str, script: dict) -> None:
    book = store.book(book_id)
    if book is not None:
        script["narration"] = compute_narration(book, script)

def find_character(script: dict, name: str) -> str | None:
    key = norm_name(name)
    for cid, c in script["characters"].items():
        if key == norm_name(c["name"]) or key in (norm_name(a) for a in c.get("aliases", [])):
            return cid
    return None


_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_SMALL_WORDS = {"of", "in", "on", "at", "to", "and", "or", "the", "a", "an", "for", "with", "from", "by"}
_NAME_PARTICLES = {"van", "von", "de", "da", "di", "du", "del", "della", "der", "den", "la", "le", "bin", "ibn",
                   "al", "el", "y", "zu", "ten", "ter"}
# Words that say how a character is addressed or related rather than who they are.
_DESCRIPTOR_WORDS = {"mom", "mother", "mama", "mum", "dad", "father", "papa", "grandma", "grandpa", "granny",
                     "grandmother", "grandfather", "sister", "brother", "aunt", "uncle", "boss", "master",
                     "captain", "chief", "sir", "madam", "lady", "lord", "miss", "mister", "mr.", "mrs.", "ms.",
                     "wife", "husband", "son", "daughter", "girl", "boy", "man", "woman", "child", "kid"}


def _is_descriptor(name: str) -> bool:
    """"Dad", "Urano's mother", "Old Man": how someone is called, not their name."""
    words = name.lower().replace("\u2019", "'").split()
    return bool(words) and ("'s" in name.lower().replace("\u2019", "'") or all(w in _DESCRIPTOR_WORDS for w in words))


def canonical_names(book: dict, script: dict, ratio: int = 3) -> list[tuple[str, str]]:
    """Shows each character under the name the book uses for them. After labelling, a character the model
    named by description ("Urano's mother", "Dad") takes a proper name it was also called (Effa, Gunther),
    and any character takes an alias the text uses `ratio` times as often (the heroine of Ascendance of a
    Bookworm is "Myne" 367 times and "Urano", her past life, 95). The old name stays an alias; names the
    reader typed are kept."""
    text = "\n".join(p for ch in book.get("chapters", []) for p in ch.get("paragraphs", []))
    counts: dict[str, int] = {}

    def count(n: str) -> int:
        if n not in counts:
            counts[n] = len(re.findall(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text)) if n else 0
        return counts[n]

    renamed = []
    for cid, c in script.get("characters", {}).items():
        if c.get("name_user") or not c.get("aliases"):
            continue
        name = c["name"]
        options = [a for a in c["aliases"] if a != name and a[:1].isupper() and not _is_descriptor(a)
                   and find_character(script, a) == cid and count(a) >= 3]
        if not options:
            continue
        best = max(options, key=count)
        if (_is_descriptor(name) and not _is_descriptor(best)) or count(best) >= ratio * max(1, count(name)):
            if name not in c["aliases"]:
                c["aliases"].append(name)
            c["name"] = best
            renamed.append((name, best))
    return renamed


def normalize_character_name(name: str) -> tuple[str, bool]:
    """How a character's name is shown, and whether it is a description rather than a name:
    "a prisoner" -> ("Prisoner", True), "the tattoo artist" -> ("Tattoo Artist", True),
    "the Neighbor" -> ("The Neighbor", False); deliberate casing ("Type Twelve", "Mr. Akutsu") is kept."""
    n = " ".join((name or "").split())
    if not n or not n[:1].isalpha():
        return n, False
    if n[:1].isupper():
        # "Bookstore employee", "Urano's mother": a capitalised description gets the rest of its words
        # capitalised too; name particles ("Ludwig van Beethoven") and small words stay as they are.
        words = n.split(" ")
        if "(" not in n and any(w[:1].islower() and w not in _SMALL_WORDS and w not in _NAME_PARTICLES
                                for w in words[1:]):
            n = " ".join(w if i == 0 or w in _SMALL_WORDS or w in _NAME_PARTICLES else w[:1].upper() + w[1:]
                         for i, w in enumerate(words))
        return n, False
    rest = _ARTICLE.sub("", n)
    if any(ch.isupper() for ch in rest):  # "the Neighbor", "a Starsage": a name after all
        return n[:1].upper() + n[1:], False
    words = rest.split(" ")
    return " ".join(w if i and w in _SMALL_WORDS else w[:1].upper() + w[1:] for i, w in enumerate(words)), True


def normalize_names(script: dict) -> int:
    """Shows descriptive names as names ("a prisoner" -> "Prisoner", flagged unnamed); the old form stays an
    alias, so matching and the series memory still find them."""
    n = 0
    for c in script.get("characters", {}).values():
        if c.get("name_user"):
            continue
        shown, unnamed = normalize_character_name(c.get("name", ""))
        if unnamed:
            c["unnamed"] = True
        if shown and shown != c.get("name"):
            aliases = c.setdefault("aliases", [])
            if c["name"] not in aliases:
                aliases.append(c["name"])
            c["name"] = shown
            n += 1
    script["names_v"] = 2
    return n


def ensure_character(script: dict, name: str, gender: str = "unknown", age: str = "unknown",
                     aliases: list[str] | tuple = (), exact: bool = False) -> str | None:
    """The one place characters are created. Names from the model go through normalize_character_name
    ("a prisoner" -> "Prisoner"); `exact` keeps a name the reader typed as it is."""
    if not name or name.strip().upper() == NARRATOR:
        return None
    cid = find_character(script, name)
    if cid is None:
        for alias in aliases:
            cid = find_character(script, alias) if is_proper_alias(alias) else None
            if cid:
                break
    if cid is None:
        cid = slug(name)
        base, n = cid, 2
        while cid in script["characters"]:
            cid, n = f"{base}-{n}", n + 1
        shown, unnamed = (" ".join(name.split()), False) if exact else normalize_character_name(name)
        script["characters"][cid] = {"name": shown, "aliases": [name.strip()] if shown != name.strip() else [],
                                     "gender": gender, "age": age,
                                     "accent": "", "temperament": "", "description": "", "voice": None}
        if unnamed:
            script["characters"][cid]["unnamed"] = True
    c = script["characters"][cid]
    for alias in [name, *aliases]:
        alias = alias.strip()
        if not is_proper_alias(alias) or norm_name(alias) == norm_name(c["name"]):
            continue
        if norm_name(alias) in (norm_name(a) for a in c.setdefault("aliases", [])):
            continue
        owner = find_character(script, alias)
        if owner is None:  # never steal another character's name
            c["aliases"].append(alias)
    if c.get("gender", "unknown") == "unknown" and gender in GENDERS:
        c["gender"] = gender
    if c.get("age", "unknown") == "unknown" and age in AGES:
        c["age"] = age
    return cid


def is_proper_alias(alias: str) -> bool:
    """Nicknames and names ("Makeup", "Robot Girl", "Zhang Lee"), not descriptions ("the man")."""
    alias = alias.strip()
    return 0 < len(alias) <= 40 and any(ch.isupper() for ch in alias) and alias.upper() != NARRATOR


_FIRST_PERSON = {"i", "me", "myself", "narrator (i)", "the narrator", "narrator", "i (narrator)"}


def resolve_speaker(script: dict, raw: str, narrator: str = "") -> str | None:
    """Maps a model's speaker answer to a character id (None = narration)."""
    name = (raw or "").strip()
    if not name or name.upper() == NARRATOR:
        return None
    first_person = narrator if narrator and narrator.lower() not in ("third person", "unknown", "none") else ""
    if name.lower() in _FIRST_PERSON:
        return ensure_character(script, first_person) if first_person else None
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", name)
    if m:
        outer, inner = m.group(1).strip(), m.group(2).strip()
        if inner.lower() in _FIRST_PERSON or outer.lower() in _FIRST_PERSON:
            other = outer if inner.lower() in _FIRST_PERSON else inner
            return ensure_character(script, first_person or other)
        for candidate in (inner, outer):
            if candidate and find_character(script, candidate):
                return find_character(script, candidate)
        if is_proper_alias(inner) and inner.lower() not in ("unnamed", "unknown"):
            return ensure_character(script, inner)
        return ensure_character(script, outer or inner)
    return ensure_character(script, split_group(name)[0])


def roster_text(script: dict, limit: int = 40) -> str:
    counts: dict[str, int] = {}
    for line in script["lines"].values():
        if line.get("speaker"):
            counts[line["speaker"]] = counts.get(line["speaker"], 0) + 1
    ranked = sorted(script["characters"].items(), key=lambda kv: -counts.get(kv[0], 0))[:limit]
    if not ranked:
        return "(none yet)"
    parts = []
    for _, c in ranked:
        extra = ", ".join(x for x in (c.get("gender"), c.get("age")) if x and x != "unknown")
        alias = f"; also called {', '.join(c['aliases'])}" if c.get("aliases") else ""
        parts.append(f"- {c['name']}" + (f" ({extra})" if extra else "") + alias)
    return "\n".join(parts)


def recent_text(script: dict, book: dict, chunk: Chunk, n: int = 6) -> str:
    """The last few attributed quotations before this chunk, for continuity."""
    ch = book["chapters"][chunk.chapter]
    prior = [q for q in ch.get("quotes", []) if int(q[1]) < chunk.paragraphs[0] and q[0] in script["lines"]][-n:]
    out = []
    for qid, p, s, e in prior:
        sp = script["lines"][qid].get("speaker")
        name = script["characters"].get(sp, {}).get("name", "NARRATOR") if sp else "NARRATOR"
        text = ch["paragraphs"][int(p)][int(s):int(e)]
        out.append(f"{name}: {text[:80]}")
    return "\n".join(out) or "(start of chapter)"


def apply_chunk_result(script: dict, result: dict, local: dict[str, list[str]]) -> int:
    narrator = (result.get("narrator") or "").strip()
    if narrator.lower() in _FIRST_PERSON:
        narrator = script.get("narrator", "")
    if narrator:
        script["narrator"] = narrator
        if narrator.lower() not in ("third person", "unknown", "none", NARRATOR.lower()):
            ensure_character(script, narrator)
    else:
        narrator = script.get("narrator", "")
    for sp in result.get("speakers", []):
        name = sp.get("name", "")
        if name.strip().lower() in _FIRST_PERSON:
            continue
        for part in split_group(name):
            if "(" in part:
                resolve_speaker(script, part, narrator)
                continue
            ensure_character(script, part, sp.get("gender", "unknown"), sp.get("age", "unknown"),
                             [a for a in sp.get("aliases") or [] if isinstance(a, str)])
    n = 0
    for q in result.get("quotes", []):
        qids = local.get(str(q.get("id")))
        if not qids:
            continue
        if isinstance(qids, str):
            qids = [qids]
        delivery = (q.get("delivery") or "").strip().lower()
        if delivery in ("neutral", "none", "normal", ""):
            delivery = None
        # "the Hatter and the March Hare" in chorus: voiced by the first.
        speaker = resolve_speaker(script, q.get("speaker", ""), narrator)
        for qid in qids:
            script["lines"][qid] = {"speaker": speaker, "delivery": delivery}
            if q.get("mention"):
                script["lines"][qid]["mention"] = True
            n += 1
    if result.get("revisions"):
        apply_revisions(script, result)
    if result.get("profile_updates"):
        from .profile import apply_updates

        apply_updates(script, result["profile_updates"])
    return n


def split_group(name: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*(?:,|&|\band\b)\s*", name or "") if p.strip()]
    return parts or [name or ""]


# ----------------------------------------------------------------- LLM


class LlmError(Exception):
    pass


class LlmPreempted(Exception):
    """The LLM was stopped because someone started using the site (llm_mode "adaptive"); the step is retried later."""


class LocalLlm:
    """OpenAI-compatible chat endpoint (llama.cpp's llama-server), started on demand."""

    def __init__(self, cfg, session_factory=None):
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self.model_name = Path(cfg.llm_model).stem if cfg.llm_model else ""
        # Where the next request should run; the server is restarted when this changes.
        self.gpu = cfg.llm_mode == "gpu"
        # On the GPU next to the speech model (experts of llm_cpu_moe_shared layers in RAM), not on all of it.
        self.shared = cfg.llm_mode == "shared"
        self.preempted = False
        self.starting_since: float | None = None
        self.running_gpu: bool | None = None
        self.running: tuple | None = None  # (model file, context) of the running server

    async def preempt(self) -> None:
        """Stop the LLM now to give the GPU back; a request in flight ends with LlmPreempted."""
        self.preempted = True
        await self.stop()

    @property
    def url(self) -> str:
        return self.cfg.llm_url.rstrip("/")

    async def _alive(self, session) -> bool:
        try:
            async with session.get(self.url.rsplit("/v1", 1)[0] + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    async def ensure(self, session, model: str | None = None, context: int | None = None) -> None:
        model_file = model or self.cfg.llm_model
        context = context or self.cfg.llm_context
        budget = getattr(self.cfg, "llm_reasoning_budget", -1)
        # On the GPU next to the speech model ("shared"), or with the whole GPU to itself.
        shared = self.gpu and self.shared
        cpu_moe = self.cfg.llm_cpu_moe_shared if shared else self.cfg.llm_cpu_moe
        if self.proc is not None and self.proc.returncode is None and (
                self.running_gpu != self.gpu or self.running != (model_file, context, budget, cpu_moe)):
            await self.stop()
        if await self._alive(session):
            return
        model = Path(model_file) if model_file else None
        if model and not model.is_absolute():
            model = self.cfg.models_path / "llm" / model
        exe = self.cfg.models_path / "llama.cpp" / ("cuda" if self.gpu else "cpu") / "llama-server.exe"
        if not model or not model.exists() or not exe.exists():
            raise LlmError(
                f"No local LLM {model_file!r}. Run download-llm.ps1, or point llm_url at a running "
                "OpenAI-compatible server (LM Studio, Ollama, llama-server)."
            )
        port = int(self.url.rsplit(":", 1)[1].split("/")[0])
        args = [str(exe), "-m", str(model), "--port", str(port), "--host", "127.0.0.1",
                "-c", str(context), "-t", str(self.cfg.llm_threads), "--jinja", "-np", "1"]
        if budget >= 0:
            args += ["--reasoning-budget", str(budget)]
        if self.gpu:
            args += ["-ngl", "99", "--n-cpu-moe", str(cpu_moe)]
        else:
            args += ["-ngl", "0"]
        logger.info("starting LLM: %s", " ".join(args))
        log = open(self.cfg.data_path / "llm.log", "ab")
        self.proc = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log,
                                                         creationflags=0x08000000)
        self.running_gpu = self.gpu
        self.running = (model_file, context, budget, cpu_moe)
        self.starting_since = time.monotonic()
        try:
            for _ in range(600):
                await asyncio.sleep(1)
                if self.proc is None or self.proc.returncode is not None:  # stopped (preempted) or crashed
                    if self.preempted:
                        raise LlmPreempted()
                    raise LlmError("llama-server exited; see data/llm.log")
                if await self._alive(session):
                    return
            raise LlmError("llama-server did not start in 10 minutes")
        finally:
            self.starting_since = None

    def info(self) -> dict:
        """For the website's banner: whether the LLM is starting, running or stopped, and where."""
        alive = self.proc is not None and self.proc.returncode is None
        state = "starting" if alive and self.starting_since else "running" if alive else "stopped"
        where = None if not alive else "cpu" if not self.running_gpu else "shared" if self.shared else "whole_gpu"
        return {"state": state, "where": where}

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.proc = None
        self.running_gpu = None
        self.running = None

    async def complete(self, session, system: str, user: str, schema: dict, max_tokens: int = 2048,
                       model: str | None = None, thinking: bool = False, effort: str = "high") -> dict:
        """One chat completion constrained to `schema`. With `thinking`, reasoning models reason
        first (llama.cpp returns the reasoning separately and applies the schema to the answer)."""
        context = self.cfg.llm_dossier_context if thinking else None
        if self.preempted:
            raise LlmPreempted()
        await self.ensure(session, model, context)
        body = {
            "model": Path(model).stem if model else (self.model_name or "local"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.6 if thinking else 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}},
            "chat_template_kwargs": ({"enable_thinking": True, "reasoning_effort": effort} if thinking
                                     else {"enable_thinking": False, "reasoning_effort": "low"}),
        }
        for attempt in range(3):
            try:
                async with session.post(f"{self.url}/chat/completions", json=body, timeout=1800) as r:
                    data = await r.json(content_type=None)
                    if r.status >= 400:
                        raise LlmError(f"LLM HTTP {r.status}: {str(data)[:300]}")
                break
            except asyncio.TimeoutError as exc:
                raise LlmError("LLM timed out") from exc
            except (OSError, ClientConnectionError) as exc:
                if self.preempted:  # stopped on purpose: someone started using the site
                    raise LlmPreempted() from exc
                # A local llama-server that crashed mid-request (seen once: a transient page-in
                # fault) is restarted and the request retried, rather than failing the whole book.
                if attempt == 2 or self.proc is None:
                    raise
                logger.warning("LLM connection lost (%s); restarting it and retrying", exc)
                await self.stop()
                await asyncio.sleep(2)
                await self.ensure(session, model, context)
        content = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage", {})
        try:
            parsed = json.loads(content[content.find("{"):])
        except ValueError as exc:
            raise LlmError(f"LLM returned invalid JSON: {content[:200]}") from exc
        parsed["_usage"] = usage
        parsed["_timings"] = data.get("timings", {})
        return parsed


def preceding_text(book: dict, chunk: Chunk, words: int) -> str:
    """Up to `words` words of the chapter just before this chunk (unnumbered, for context)."""
    paras = book["chapters"][chunk.chapter]["paragraphs"]
    out, n, k = [], 0, chunk.paragraphs[0] - 1
    while k >= 0 and n < words:
        out.insert(0, paras[k])
        n += len(paras[k].split())
        k -= 1
    return "\n\n".join(out)


REVIEW_NOTE = ("A first reading proposed the labels below. Check every one against the speech tags, who is "
               "addressed, turn-taking, who is present and who \"I\" is, and correct the ones that are wrong "
               "(keep the right ones). Return the full list.")


REVISE_NOTE = ("Earlier lines, already labelled, which you may correct: [R1], [R2]... with the speaker each was "
               "given. Something in the passage can show that one of them was wrong: a character gives their name "
               "(so \"a prisoner\" was really that person), a reply shows who asked, or a tag names a speaker the "
               "earlier text left unclear. Label the passage first; then, only where the passage gives clear evidence, "
               "list the earlier line in \"revisions\" with its R number, its first two or three words in \"starts\", its "
               "correct speaker and a reason that quotes, word for word in quotation marks, the words of the passage that "
               "show it (or \"same as R2\" when it rests on the evidence of another revision). R numbers are not the "
               "passage's [n] numbers. A description such as \"a prisoner\" can cover several different people: revise "
               "only the lines the evidence is about. Most of the time \"revisions\" is empty.")


def _words(text: str, n: int = 3) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())[:n]


def _quotes_passage(reason: str, passage: str) -> bool:
    """The reason quotes at least three consecutive words of the passage."""
    flat = " ".join(re.findall(r"[^\W_]+", passage.lower()))
    for quoted in re.findall(r"[\"'\u201c\u2018]([^\"\u201d\u2019]{6,200})[\"'\u201d\u2019]", reason):
        words = re.findall(r"[^\W_]+", quoted.lower())
        for i in range(len(words) - 2):
            if " ".join(words[i:i + 3]) in flat:
                return True
    return False


def is_placeholder(name: str) -> bool:
    """An unnamed speaker the model described ("a prisoner", "the girl") rather than named."""
    return bool(name) and (name[:1].islower() or re.match(r"^(?:a|an|the)\s", name) is not None)


def revisable_lines(script: dict, book: dict, chunk: Chunk, n: int, unnamed_only: bool = False) -> list[tuple[str, str, str]]:
    """The last `n` labelled quotations before this chunk in its chapter, as (quote id, speaker, text); the
    reader's corrections and quoted terms are never offered for revision."""
    ch = book["chapters"][chunk.chapter]
    fixed = script.get("corrections", {})
    out = []
    for qid, p, s, e in ch.get("quotes", []):
        line = script["lines"].get(qid)
        if int(p) >= chunk.paragraphs[0] or line is None or qid in fixed or line.get("mention"):
            continue
        sp = line.get("speaker")
        name = script["characters"].get(sp, {}).get("name", NARRATOR) if sp else NARRATOR
        if unnamed_only and not (is_placeholder(name) or script["characters"].get(sp, {}).get("unnamed")):
            continue
        out.append((qid, name, ch["paragraphs"][int(p)][int(s):int(e)]))
    return out[-n:] if n else []


def apply_revisions(script: dict, result: dict) -> list[dict]:
    """Applies the passage's revisions of earlier lines (never the reader's corrections) and logs them."""
    editable = result.get("_editable") or {}
    narrator = script.get("narrator", "")
    done = []
    accepted: dict[str, str | None] = {}
    for r in result.get("revisions") or []:
        qid = editable.get(str(r.get("id")))
        line = script["lines"].get(qid) if qid else None
        if line is None or qid in script.get("corrections", {}) or not (r.get("reason") or "").strip():
            continue
        said, text = _words(r.get("starts") or "", 2), _words(result.get("_texts", {}).get(qid, ""), 2)
        if not said or said != text[:len(said)]:
            continue  # aimed at a different line than its R number
        new = resolve_speaker(script, r.get("speaker", ""), narrator)
        reason = r.get("reason") or ""
        same = re.search(r"same as (R\d+)", reason, re.IGNORECASE)
        if same:
            if accepted.get(same.group(1).upper()) != new:
                continue  # leans on a revision that was not accepted, or names someone else
        elif not _quotes_passage(reason, result.get("_passage", "")):
            continue  # no evidence from the passage itself
        if new == line.get("speaker"):
            continue
        accepted[str(r.get("id")).upper()] = new
        event = {"id": qid, "from": line.get("speaker"), "to": new, "reason": r["reason"].strip()[:160]}
        script["lines"][qid] = {**line, "speaker": new, "revised": True}
        script.setdefault("revisions", []).append(event)
        done.append(event)
    return done


async def label_chunk(llm: LocalLlm, session, book: dict, script: dict, chunk: Chunk,
                      thinking: bool = False, model: str | None = None, effort: str = "high",
                      context_words: int = 0, recent: int = 6,
                      proposed: dict[str, str] | None = None, revise: int = 0,
                      revise_unnamed: bool = False, profiles: bool = False) -> tuple[dict, dict[str, str]]:
    """Label one chunk. `context_words` adds the preceding text, `recent` sets how many earlier
    attributions are shown, and `proposed` (number -> speaker) asks the model to review a first pass."""
    text, local = render_chunk(book, chunk)
    chapter = book["chapters"][chunk.chapter]
    narrator = script.get("narrator") or "unknown (work it out from the passage)"
    if chunk.tense:
        known = (script.get("narrator_by_tense") or {}).get(chunk.tense)
        previous = script.get("last_tense")
        if known:
            narrator = known
            if previous and previous != chunk.tense:
                narrator = (f"{known}. The narration switches to the {chunk.tense} tense here; earlier "
                            f"{chunk.tense}-tense passages were narrated by {known}, so \"I\" is {known} again.")
        elif previous and previous != chunk.tense:
            narrator = (f"{narrator}. But the narration switches from the {previous} to the {chunk.tense} tense "
                        f"here, which usually means a different first-person narrator: work out who \"I\" is.")
    pov = pov_change(book, chunk, use_title=chunk.tense is None)
    if pov:
        narrator = (f"{narrator}. But this passage starts with a point-of-view change, \"{pov}\": from here "
                    f"\"I\" is that character, not the previous narrator. Lines \"I\" says belong to them.")
    if script.get("dossier_state"):
        from .dossier import render_dossier, scene_cast

        present = [script["characters"][c]["name"] for c in scene_cast(script, chunk.chapter, chunk.paragraphs)]
        known = ("Characters (from a first read of the whole book: use their speech styles, verbal tics and "
                 "relationships to recognise who speaks in untagged dialogue, and \"first appears as\" to give a "
                 "character's real name even before the book reveals it):\n"
                 f"{render_dossier(script, limit=30, width=110)}"
                 + (f"\n\nPresent in this part of the story: {', '.join(present)}" if present else ""))
    else:
        # Profiles (and the reader's profile edits) only shape voices; they are not given to attribution.
        known = f"Known characters:\n{roster_text(script)}"
    before = preceding_text(book, chunk, context_words) if context_words else ""
    editable: dict[str, str] = {}
    revise_block = ""
    if revise:
        prior = revisable_lines(script, book, chunk, revise, revise_unnamed)
        editable = {f"R{i}": qid for i, (qid, _, _) in enumerate(prior, 1)}
        texts = {qid: text for qid, _, text in prior}
        if prior:
            revise_block = (f"{REVISE_NOTE}\n" + "\n".join(f"[R{i}] {name}: {text[:100]}"
                                                         for i, (_, name, text) in enumerate(prior, 1)) + "\n\n")
    review = ""
    if proposed:
        review = (f"{REVIEW_NOTE}\n" + "\n".join(f"[{n}] {proposed.get(n, '?')}" for n in local) + "\n\n")
    user = (
        f"Book: {book.get('title', '')} by {book.get('author', '')}\n"
        f"Chapter: {chapter.get('title', '')}\n\n"
        f"{known}\n\n"
        f"Narrator of the previous passage: {narrator}\n\n"
        + (f"Story just before this passage (for context only):\n{before}\n\n" if before else "")
        + f"Most recent lines before this passage:\n{recent_text(script, book, chunk, n=recent)}\n\n"
        f"Passage:\n{text}\n\n"
        f"{review}{revise_block}"
        f"Label every number from 1 to {len(local)} once."
    )
    # Generous budget: reasoning models (gpt-oss) spend tokens before the JSON, and the
    # schema grammar ends the output as soon as the JSON is complete anyway.
    budget = 1024 + 60 * len(local)
    extra = {"thinking": True, "effort": effort} if thinking else {}
    if model:
        extra["model"] = model
    if thinking:
        budget += 8000
    if editable:
        budget += 60 * min(len(editable), 8)
    schema = chunk_schema(list(local), list(editable), profiles=profiles)
    system = SYSTEM_PROMPT
    if profiles:
        from .profile import PROMPT_RULE

        system += "\n" + PROMPT_RULE
        budget += 200
    try:
        result = await llm.complete(session, system, user, schema, max_tokens=budget, **extra)
    except LlmError as exc:
        if "invalid JSON" not in str(exc):
            raise
        result = await llm.complete(session, system, user, schema, max_tokens=budget * 3, **extra)
    result["_editable"] = editable
    if editable:
        result["_texts"] = texts
        result["_passage"] = text
    # Backstop for the prompt rule: a quoted term inside a sentence is narration.
    paras = book["chapters"][chunk.chapter]["paragraphs"]
    spans = {q[0]: q for q in chunk.quotes}
    for q in result.get("quotes", []):
        ids = local.get(str(q.get("id"))) or []
        if ids and all(is_mention(paras[spans[i][1]], spans[i][2], spans[i][3]) for i in ids):
            q.update(speaker=NARRATOR, delivery="neutral", mention=True)
    return result, local


def _roster_batches(blocks: list[str], fixed_chars: int, context: int) -> list[list[str]]:
    """Splits the roster's character blocks so each request (prompt plus its answer budget) fits the
    LLM's context; a big cast with long evidence overflowed 12k tokens in one request."""
    tokens = lambda chars: chars // 3 + 1  # conservative; English runs about 4 chars per token
    batches: list[list[str]] = []
    used = 0
    for block in blocks:
        cost = tokens(len(block) + 2) + 300
        if batches and used + cost <= context - 1024 - 256 - tokens(fixed_chars):
            batches[-1].append(block)
            used += cost
        else:
            batches.append([block])
            used = cost
    return batches


async def describe_cast(llm: LocalLlm, session, book: dict, script: dict, max_chars: int = 12) -> None:
    public = ScriptStore.public(script)
    # Characters that already have a voice (re-analysis) keep their description.
    main = [c for c in public["characters"] if c["lines"] >= 2 and not c.get("voice")][:max_chars]
    if not main:
        return
    samples = {}
    for qid, line in public["lines"].items():
        if line["speaker"] in {c["id"] for c in main}:
            samples.setdefault(line["speaker"], []).append(qid)
    blocks = []
    for c in main:
        texts = []
        for qid in samples.get(c["id"], [])[:6]:
            m = re.match(r"c(\d+)p(\d+)q(\d+)", qid)
            if not m:
                continue
            ci, pi, s = map(int, m.groups())
            para = book["chapters"][ci]["paragraphs"][pi]
            texts.append(para[max(0, s - 60): s + 160])
        aliases = [a for a in c.get("aliases", []) if a != c["name"]]
        raw = script["characters"][c["id"]]
        from .profile import effective as profile_of

        prof = profile_of(raw)
        about = ("Profile (noted while reading; follow its gender, apparent age and appearance):\n"
                 + "\n".join(f"- {k.replace('_', ' ')}: {prof[k]}" for k in
                             ("gender", "apparent_age", "appearance", "speech_style", "notes") if prof.get(k)) + "\n"
                 if prof else "")
        looks = appearance_evidence(book, [c["name"], *aliases])
        blocks.append(
            f"## {c['name']}" + (f" (also called: {', '.join(aliases)})" if aliases else "") + "\n" + about
            + "Described as:\n" + ("\n".join(f"- {t}" for t in looks) or "- (nothing found)") + "\n"
            + "Lines:\n" + "\n".join(f"- {t}" for t in texts))
    head = f"Book: {book.get('title', '')} by {book.get('author', '')}\n\n"
    described = []
    for batch in _roster_batches(blocks, len(ROSTER_PROMPT) + len(head), llm.cfg.llm_context):
        result = await llm.complete(session, ROSTER_PROMPT, head + "\n\n".join(batch), ROSTER_SCHEMA,
                                    max_tokens=1024 + 300 * len(batch))
        described += result.get("characters", [])
    for d in described:
        cid = find_character(script, d.get("name", ""))
        if cid is None:
            continue
        c = script["characters"][cid]
        for key in ("gender", "age", "accent", "temperament", "appearance"):
            if d.get(key):
                c[key] = d[key]
        c["description"] = d.get("voice", "")
        from . import profile

        # What the roster step saw fills what the reading left empty (the fallback when nothing was noted).
        prof = c.setdefault("profile", {})
        if d.get("appearance") and not prof.get("appearance"):
            prof["appearance"] = " ".join(d["appearance"].split())[:profile.TEXT_MAX]
        for key, src in (("gender", "gender"), ("apparent_age", "age")):
            if d.get(src) and d[src] != "unknown" and prof.get(key, "unknown") == "unknown":
                prof[key] = d[src]
        extra = "; ".join(x for x in (d.get("temperament"), d.get("accent") and d["accent"] != "neutral"
                                      and f"{d['accent']} accent") if x)
        if extra and not prof.get("notes"):
            prof["notes"] = extra
        profile.sync_character(c)  # the profile's gender and apparent age win (the reader's above all)


# ----------------------------------------------------------------- analysis job


class Analyzer:
    def __init__(self, app: web.Application, store: ScriptStore, llm: LocalLlm):
        self.app = app
        self.store = store
        self.llm = llm
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.current: str | None = None
        # Background GPU work while the site is idle runs one thing at a time (an LLM step, or a book's
        # automatic casting), so the LLM and the voice designer don't keep evicting each other.
        self.background = asyncio.Lock()

    def start(self) -> None:
        from .app import ENGINES_KEY

        async def make_room():  # the voice designer needs the VRAM a GPU LLM holds
            if self.llm.running_gpu:
                await self.llm.preempt()

        if ENGINES_KEY in self.app:
            self.app[ENGINES_KEY].before_designer = make_room
        from .activity import ACTIVITY_KEY

        if ACTIVITY_KEY in self.app:
            self.app[ACTIVITY_KEY].on_use.append(self.on_use)
            self.app[ACTIVITY_KEY].on_design.append(self.on_design)
        self.task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
        await self.llm.stop()

    async def enqueue(self, book_id: str) -> None:
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            if s["status"] in ("queued", "running"):
                return
            s.update(status="queued", detail="waiting for the analyser")
            self.store.save_script(book_id, s)
        await self.queue.put(book_id)

    async def _run(self) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            while True:
                book_id = await self.queue.get()
                self.current = book_id
                try:
                    await self.analyse(session, book_id)
                    s = self.store.script(book_id)
                    if s.get("auto_cast") and s.get("status") == "done":  # added from the library: cast it too
                        async with self.store.lock(book_id):
                            s = self.store.script(book_id)
                            s.update(status="casting", detail="creating character voices")
                            self.store.save_script(book_id, s)
                        engine = s.get("auto_cast_engine") or self.llm.cfg.voice_design_engine
                        # In the background: it waits until the site is idle (or nobody listens), and the
                        # next book's analysis goes on meanwhile.
                        self._spawn_cast(book_id, engine)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("analysis of %s failed", book_id)
                    async with self.store.lock(book_id):
                        s = self.store.script(book_id)
                        s.update(status="error", detail=str(exc)[:300])
                        self.store.save_script(book_id, s)
                finally:
                    self.current = None
                    if self.queue.empty():
                        await self.llm.stop()  # free RAM/VRAM when idle

    def _spawn_cast(self, book_id: str, engine: str) -> None:
        tasks = self.app[CAST_TASKS_KEY] if CAST_TASKS_KEY in self.app else set()
        task = asyncio.ensure_future(run_cast(self.app, book_id, engine, background=True))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _label(self, session, book: dict, script: dict, chunk: Chunk):
        thinking = self.llm.cfg.llm_label_thinking
        cfg = self.llm.cfg
        return await self._gpu(lambda: label_chunk(self.llm, session, book, script, chunk, thinking=thinking,
                                                   context_words=cfg.llm_label_context_words,
                                                   recent=cfg.llm_label_recent_lines,
                                                   revise=cfg.llm_label_revise_window,
                                                   revise_unnamed=cfg.llm_label_revise_unnamed,
                                                   profiles=cfg.llm_label_profiles))

    async def _read_for_dossier(self, session, book: dict, book_id: str) -> None:
        """Pass 1: the reasoning model reads the whole book and builds the character dossier."""
        from . import dossier, series

        cfg = self.llm.cfg
        s = self.store.script(book_id)
        if not cfg.llm_dossier_model or (s.get("dossier_state") or {}).get("done"):
            return
        if not s.get("dossier_state"):
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                series.seed(self.app, book_id, s)
                self.store.save_script(book_id, s)

        async def progress(s: dict, frac: float, detail: str) -> None:
            s.update(status="running", progress=round(0.5 * frac, 3), detail=detail)
            self.store.save_script(book_id, s)

        await dossier.build(self.llm, session, book, self.store, book_id, cfg.llm_dossier_model, self._gpu,
                            progress)

    @staticmethod
    def narrators_from_dossier(s: dict) -> None:
        """Record who pass 1 thinks narrates, as dossier context only. Pass 2 keeps tracking the
        narrator itself: forcing pass 1's per-tense votes onto it measured worse (a confused vote
        mislabels every "I said" line of a section)."""
        from .dossier import narrators

        for key, cid in narrators(s).items():
            if cid in s["characters"]:
                d = s["characters"][cid].setdefault("dossier", {})
                d["narrates"] = "yes" if key == "any" else f"the {key}-tense sections"

    async def _gpu(self, make_coro, interactive: bool = False):
        """Runs one LLM step.

        "adaptive" (default) follows what the site is used for (activity.py):
        * reading: next to the speech model (experts of llm_cpu_moe_shared layers in RAM);
        * designing (and design_pause_s after the last design): waits; steps someone asked for
          (`interactive`, e.g. sorting the cast) still run, next to the speech model;
        * idle: the whole GPU (the speech model is unloaded), one background thing at a time.
        A step stopped because the state changed (LlmPreempted) is retried in the new state.
        "shared": always next to the speech model. "auto": the whole GPU when the speech model has been idle,
        otherwise the CPU. "gpu": always the whole GPU. "cpu": never the GPU."""
        from .activity import ACTIVITY_KEY
        from .app import ENGINES_KEY

        manager = self.app[ENGINES_KEY]
        mode = self.llm.cfg.llm_mode
        if mode == "adaptive" and ACTIVITY_KEY in self.app:
            activity = self.app[ACTIVITY_KEY]
            while True:
                if not interactive:
                    await activity.wait_for(lambda: not activity.designing, on_wait=self._note_designing)
                if interactive or activity.in_use:
                    async with manager.exclusive(evict=False):
                        if activity.designing and not interactive:
                            continue
                        self.llm.gpu, self.llm.shared, self.llm.preempted = True, True, False
                        try:
                            return await make_coro()
                        except LlmPreempted:
                            logger.info("LLM stopped for the voice designer")
                            continue
                # Idle: take the background turn (a casting may hold it), unless someone comes back meanwhile.
                got = False
                while not activity.in_use:
                    try:
                        await asyncio.wait_for(self.background.acquire(), 2.0)
                        got = True
                        break
                    except asyncio.TimeoutError:
                        pass
                if not got:
                    continue
                try:
                    async with manager.exclusive():
                        if activity.in_use:
                            continue
                        self.llm.gpu, self.llm.shared, self.llm.preempted = True, False, False
                        try:
                            return await make_coro()
                        except LlmPreempted:
                            logger.info("LLM stopped: the site is in use")
                            continue
                finally:
                    self.background.release()
        if mode == "shared":
            self.llm.gpu, self.llm.shared, self.llm.preempted = True, True, False
            async with manager.exclusive(evict=False):
                return await make_coro()
        self.llm.shared = False
        use_gpu = mode == "gpu" or (mode == "auto" and manager.idle_seconds() >= GPU_IDLE_S)
        self.llm.gpu = use_gpu
        if not use_gpu:
            return await make_coro()
        async with manager.exclusive():
            try:
                return await make_coro()
            finally:
                if manager.tts_waiting:
                    await self.llm.stop()

    async def _note_designing(self) -> None:
        """Say on the book being analysed that it waits while voices are designed."""
        book_id = self.current
        if not book_id:
            return
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            if s.get("status") in ("running", "queued"):
                s["detail"] = "paused while you design voices; continues 10 minutes after the last one"
                self.store.save_script(book_id, s)

    async def on_use(self) -> None:
        """Someone started using the site ("adaptive"): an LLM on the whole GPU is stopped (its step goes on
        next to the speech model) and the speech model loads."""
        from .app import CONFIG_KEY, ENGINES_KEY

        if self.llm.cfg.llm_mode != "adaptive":
            return
        if self.llm.running_gpu and not self.llm.shared:
            await self.llm.preempt()
        manager = self.app[ENGINES_KEY]
        try:
            await manager.preload(self.app[CONFIG_KEY].default_engine)
        except Exception as exc:  # noqa: BLE001 - the first request will report it
            logger.warning("could not load the speech model: %s", exc)

    async def on_design(self) -> None:
        """Someone started designing voices ("adaptive"): the LLM stops at once to make room for the designer."""
        if self.llm.cfg.llm_mode == "adaptive" and self.llm.proc is not None:
            await self.llm.preempt()

    async def analyse(self, session, book_id: str) -> None:
        book = self.store.book(book_id)
        if book is None:
            raise LlmError("book text not uploaded")
        s = self.store.script(book_id)
        if not s["chunks_done"] and not s.get("dossier_state") and not s.get("series_seeded"):
            from . import series

            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                s["series_seeded"] = series.seed(self.app, book_id, s)
                self.store.save_script(book_id, s)
        await self._read_for_dossier(session, book, book_id)
        has_dossier = bool(self.store.script(book_id).get("dossier_state"))
        if has_dossier:
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                if not s["chunks_done"]:
                    self.narrators_from_dossier(s)
                    self.store.save_script(book_id, s)
        base, span = (0.5, 0.5) if has_dossier else (0.0, 1.0)
        chunks = make_chunks(book, runs=mixed_tense_runs(book))
        t0 = time.perf_counter()
        for i, chunk in enumerate(chunks):
            s = self.store.script(book_id)
            if chunk.key in s["chunks_done"]:
                continue
            s.update(status="running", progress=round(base + span * i / max(1, len(chunks)), 3),
                     detail=f"chapter {chunk.chapter + 1}: {len(s['lines'])} lines so far")
            self.store.save_script(book_id, s)
            result, local = await self._label(session, book, s, chunk)
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                by_tense = s.setdefault("narrator_by_tense", {})
                if chunk.tense and by_tense.get(chunk.tense):
                    # "I" in this passage is the narrator of its tense, whatever the model says.
                    result["narrator"] = by_tense[chunk.tense]
                apply_chunk_result(s, result, local)
                if chunk.tense:
                    s["last_tense"] = chunk.tense
                    other = "past" if chunk.tense == "present" else "present"
                    # The first narrator seen in each tense sticks, and the two tenses have different
                    # narrators: one confused passage must not re-label every later one.
                    if (chunk.tense not in by_tense and find_character(s, s.get("narrator") or "")
                            and find_character(s, s["narrator"]) != find_character(s, by_tense.get(other) or "")):
                        by_tense[chunk.tense] = s["narrator"]
                s["chunks_done"].append(chunk.key)
                s["model"] = self.llm.model_name
                s.setdefault("stats", {}).setdefault("tokens", 0)
                s["stats"]["tokens"] += int(result.get("_usage", {}).get("total_tokens", 0))
                self.store.save_script(book_id, s)
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            renamed = canonical_names(book, s)
            if renamed:
                logger.info("%s: named by the text: %s", book_id, ", ".join(f"{a} -> {b}" for a, b in renamed))
            s.update(detail="describing the characters")
            self.store.save_script(book_id, s)
        s = self.store.script(book_id)
        await self._gpu(lambda: describe_cast(self.llm, session, book, s))
        from . import roles

        from .profile import compact as compact_profiles

        await self._gpu(lambda: compact_profiles(self.llm, session, s))
        s["narration"] = compute_narration(book, s)  # so narrators count as main
        try:
            await self._gpu(lambda: roles.classify(self.llm, session, book, s))
        except LlmError as exc:
            logger.warning("sorting the cast failed (%s); using the line counts", exc)
            roles.apply_rules(book, s)
        if self.llm.running_gpu:
            await self.llm.stop()  # give the GPU back to TTS
        async with self.store.lock(book_id):
            fresh = self.store.script(book_id)
            for cid, c in s["characters"].items():
                if cid in fresh["characters"]:
                    for k in ("gender", "age", "accent", "temperament", "appearance", "description",
                              "role_auto", "role_source", "role_note", "role_flag", "profile"):
                        fresh["characters"][cid][k] = c.get(k, fresh["characters"][cid].get(k))
                    f = fresh["characters"][cid]
                    f["role"] = f.get("role_user") if f.get("role_user") in roles.ROLES else f.get("role_auto")
            fresh["roles_state"] = s.get("roles_state")
            fresh.update(status="done", progress=1.0,
                         detail=f"{len(fresh['lines'])} lines in {time.perf_counter() - t0:.0f}s")
            fresh.setdefault("stats", {})["seconds"] = round(time.perf_counter() - t0, 1)
            reconcile_narrators(book, fresh)
            fresh["narration"] = compute_narration(book, fresh)
            prune_characters(fresh)
            self.store.save_script(book_id, fresh)
        logger.info("analysed %s: %d lines", book_id, len(fresh["lines"]))
        await self._cast_new_speakers(book_id, fresh)

    async def _cast_new_speakers(self, book_id: str, script: dict) -> None:
        """A re-analysis can bring new speakers ("TV Announcer"); if the book was cast before, give them
        voices too, so their lines are not read in the narrator's voice."""
        public = ScriptStore.public(script)
        if not any(c.get("voice") for c in public["characters"]):
            return
        if not any(c["lines"] and not c.get("voice") for c in public["characters"]):
            return
        from .engines.manager import EngineError

        try:
            result = await cast_voices(self.app, book_id, self.llm.cfg.voice_design_engine, background=True)
            logger.info("voices for new speakers in %s: %s", book_id, result["created"])
        except (EngineError, OSError) as exc:
            logger.warning("could not create voices for new speakers: %s", exc)


def prune_characters(script: dict) -> list[str]:
    """Drop characters nobody hears: no lines, not narrating, not edited by the user, no chosen voice.

    The dossier pass records everyone who is mentioned; only speakers belong in the cast.
    """
    public = ScriptStore.public(script)
    lines = {c["id"]: c["lines"] for c in public["characters"]}
    narrating = {who for _, _, who in script.get("narration", []) if who}
    narrating |= {find_character(script, n) for n in (script.get("narrator_by_tense") or {}).values()}
    narrating.add(find_character(script, script.get("narrator") or ""))
    gone = []
    for cid, c in list(script["characters"].items()):
        voice = c.get("voice") or ""
        generated = not voice or voice.startswith(("ch-", "pool-"))
        if lines.get(cid, 0) == 0 and cid not in narrating and not c.get("user") and generated:
            del script["characters"][cid]
            gone.append(cid)
    return gone


# ----------------------------------------------------------------- casting

POOL = {
    ("male", "adult"): "An adult man with an ordinary, mid-pitched voice, natural and clear.",
    ("female", "adult"): "An adult woman with an ordinary, mid-pitched voice, natural and clear.",
    ("male", "elderly"): "An elderly man with a low, slightly rough voice, speaking unhurriedly.",
    ("female", "elderly"): "An elderly woman with a thin, gentle voice, speaking slowly.",
    ("male", "young"): "A young man in his twenties with a bright, energetic voice.",
    ("female", "young"): "A young woman in her twenties with a light, lively voice.",
    ("unknown", "child"): "A child of about ten with a high, clear voice.",
}


def pool_key(gender: str, age: str) -> tuple[str, str]:
    if age == "child":
        return ("unknown", "child")
    g = gender if gender in ("male", "female") else "male"
    if age in ("elderly",):
        return (g, "elderly")
    if age in ("teen", "young adult"):
        return (g, "young")
    return (g, "adult")


def follow_voice(c: dict, voices) -> bool:
    """A character with a voice of its own is described by that voice (what it was designed from or last
    edited to), so a voice chosen or redesigned later doesn't leave a stale description behind. Shared pool
    voices say nothing about the character."""
    vid = c.get("voice") or ""
    v = voices.get(vid) if vid and not vid.startswith("pool-") else None
    text = ((v.description or v.instruct) if v else "").strip()
    if text and text != c.get("description"):
        c["description"] = text
        return True
    return False


async def sync_voice_descriptions(app: web.Application, voice_id: str) -> None:
    """After a voice is edited: every character it voices (in any book) takes its new description."""
    from .app import VOICES_KEY

    store = app.get(DRAMA_STORE_KEY)
    if store is None:
        return
    voices = app[VOICES_KEY]
    for book_id in store.ids():
        async with store.lock(book_id):
            s = store.script(book_id)
            changed = [follow_voice(c, voices) for c in s["characters"].values() if c.get("voice") == voice_id]
            if any(changed):
                store.save_script(book_id, s)


def character_description(c: dict) -> str:
    if c.get("description"):
        return c["description"]
    from .profile import effective as profile_of, voice_description

    if profile_of(c):
        return voice_description(c)
    parts = [c.get("gender"), c.get("age"), c.get("accent") and f"{c['accent']} accent", c.get("temperament")]
    return ", ".join(p for p in parts if p and p != "unknown") or "an adult speaker with a natural voice"


def own_voice_plan(s: dict, supporting_budget: int, main_lines: int = 5) -> set[str]:
    """Who gets a designed voice of their own: main characters and narrators always, supporting characters
    (most lines first) while the budget lasts, extras never. Books without tiers: anyone with
    `main_lines` lines or more."""
    public = ScriptStore.public(s)
    lines_by = {c["id"]: c["lines"] for c in public["characters"]}
    narrators = {who for _, _, who in s.get("narration", []) if who}
    own = set()
    supporting = sorted((cid for cid, c in s["characters"].items() if c.get("role") == "supporting"), key=lambda k: -lines_by.get(k, 0))
    budget = {cid for cid in supporting[:max(0, supporting_budget)] if lines_by.get(cid, 0) > 0}
    for cid, c in s["characters"].items():
        role = c.get("role")
        if cid in narrators or role == "main" or cid in budget or (not role and lines_by.get(cid, 0) >= main_lines):
            own.add(cid)
    return own


LISTEN_QUIET_S = 60.0  # automatic casting waits until nobody has asked for speech this long


async def _wait_for_background_turn(app: web.Application, book_id: str) -> None:
    """Automatic casting gives way to people: in "adaptive" mode it waits until the site is idle,
    otherwise while the speech model is being read with."""
    from .activity import ACTIVITY_KEY
    from .app import CONFIG_KEY, ENGINES_KEY

    store = app[DRAMA_STORE_KEY]

    async def note(text: str) -> None:
        async with store.lock(book_id):
            s = store.script(book_id)
            if s.get("status") == "casting" and s.get("detail") != text:
                s["detail"] = text
                store.save_script(book_id, s)

    if app[CONFIG_KEY].llm_mode == "adaptive" and ACTIVITY_KEY in app:
        activity = app[ACTIVITY_KEY]
        if activity.in_use:
            await activity.wait_for(lambda: not activity.in_use,
                                    on_wait=lambda: note("creating voices when the site is idle"))
            await note("creating character voices")
        return
    manager = app[ENGINES_KEY]
    noted = False
    while manager.idle_seconds() < LISTEN_QUIET_S:
        if not noted:
            await note("creating voices when you stop listening")
            noted = True
        await asyncio.sleep(5)
    if noted:
        await note("creating character voices")


async def cast_voices(app: web.Application, book_id: str, engine: str, main_lines: int = 5,
                      upgrade_roles: set[str] | None = None, background: bool = False) -> dict:
    """Design voices for characters without one (see own_voice_plan); the others get pool voices.
    `upgrade_roles` also gives characters of those tiers who share a pool voice one of their own.
    `background` (casting nobody asked for just now) designs only while nobody is listening."""
    from .app import CONFIG_KEY, VOICES_KEY, design_voice

    store: ScriptStore = app[DRAMA_STORE_KEY]
    voices = app[VOICES_KEY]
    s = store.script(book_id)
    own = own_voice_plan(s, app[CONFIG_KEY].cast_supporting_voices, main_lines)
    created = []

    async def design(*args, **kwargs):
        if background:
            await _wait_for_background_turn(app, book_id)
        else:
            from .activity import touch

            touch(app, design=True)  # casting someone asked for counts as designing while it runs
        return await design_voice(*args, **kwargs)

    # Automatic casting takes the background turn (one background thing at a time while the site is idle).
    turn = (app[ANALYZER_KEY].background if background and app[CONFIG_KEY].llm_mode == "adaptive"
            and ANALYZER_KEY in app else None)
    if turn is not None:
        await _wait_for_background_turn(app, book_id)
        await turn.acquire()
    try:
        return await _cast(app, book_id, engine, s, own, voices, upgrade_roles, design, created)
    finally:
        if turn is not None:
            turn.release()


async def _cast(app, book_id, engine, s, own, voices, upgrade_roles, design, created) -> dict:
    store: ScriptStore = app[DRAMA_STORE_KEY]
    for cid, c in s["characters"].items():
        pooled = (c.get("voice") or "").startswith("pool-")
        upgrade = bool(upgrade_roles) and c.get("role") in upgrade_roles and pooled
        if c.get("voice") and voices.get(c["voice"]) and not upgrade:
            continue
        if cid in own or upgrade:
            vid = f"ch-{book_id[:8]}-{cid}"[:64]
            v = voices.get(vid) or await design(
                app, engine, character_description(c), c["name"], source="character", voice_id=vid,
                seed=zlib.crc32(cid.encode()) % 100000,
            )
            c["voice"] = v.id
            created.append(v.id)
        else:
            g, a = pool_key(c.get("gender", "unknown"), c.get("age", "unknown"))
            vid = f"pool-{g}-{a}".replace(" ", "")
            v = voices.get(vid) or await design(
                app, engine, POOL[(g, a)], f"Pool: {g} {a}", source="character", voice_id=vid, seed=7,
            )
            c["voice"] = v.id
    async with store.lock(book_id):
        fresh = store.script(book_id)
        for cid, c in s["characters"].items():
            f = fresh["characters"].get(cid)
            if f is not None and (not f.get("voice") or (f["voice"].startswith("pool-") and c.get("voice", "").startswith("ch-"))):
                f["voice"] = c.get("voice")
        store.save_script(book_id, fresh)
    return {"created": created}


# ----------------------------------------------------------------- routes

DRAMA_STORE_KEY = web.AppKey("drama_store", ScriptStore)
ANALYZER_KEY = web.AppKey("analyzer", Analyzer)
CAST_TASKS_KEY = web.AppKey("cast_tasks", set)


async def gpu_status(request: web.Request) -> web.Response:
    """What runs on the GPU right now, for the website's banner: the site's state (reading, designing, idle),
    the speech model and voice designer (loading progress), and the analysis LLM."""
    from .activity import ACTIVITY_KEY
    from .app import ENGINES_KEY

    app = request.app
    analyzer = app[ANALYZER_KEY]
    book = analyzer.current
    title = ((app[DRAMA_STORE_KEY].summary(book) or {}).get("title", "") if book else "")
    return web.json_response({
        "mode": analyzer.llm.cfg.llm_mode,
        "activity": app[ACTIVITY_KEY].json() if ACTIVITY_KEY in app else None,
        "engines": app[ENGINES_KEY].status(),
        "llm": {**analyzer.llm.info(), "book": book, "title": title},
    })


async def list_books(request: web.Request) -> web.Response:
    from .series import series_name as _series_name, volume_key

    store = request.app[DRAMA_STORE_KEY]
    books = []
    for book_id in store.ids():
        summary = store.summary(book_id) or {}
        s = store.script(book_id)
        public = ScriptStore.public(s)
        books.append({
            **summary,
            "id": book_id,
            "series": summary.get("series") or _series_name(summary.get("title", "")),
            "series_auto": not summary.get("series"),
            "volume": volume_key(summary.get("title", "")),
            "status": s.get("status", "none"),
            "progress": s.get("progress", 0.0),
            "detail": s.get("detail", ""),
            "lines": len(public["lines"]),
            "characters": len(public["characters"]),
            "voiced": sum(1 for c in public["characters"] if c.get("voice")),
            "cover": store.cover(book_id) is not None,
            "updated": s.get("updated") or summary.get("uploaded", 0),
        })
    books.sort(key=lambda b: -(b.get("updated") or 0))
    return web.json_response({"books": books})


async def get_book(request: web.Request) -> web.Response:
    store = request.app[DRAMA_STORE_KEY]
    try:
        book = store.book(request.match_info["book"])
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if book is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    return web.json_response(book)


async def list_pictures(request: web.Request) -> web.Response:
    """The book's pictures the server has (`have`) and still needs uploaded (`need`)."""
    store = request.app[DRAMA_STORE_KEY]
    try:
        book = store.book(request.match_info["book"])
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if book is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    have, need = [], []
    for path in picture_refs(book):
        f = store.picture_file(request.match_info["book"], path)
        if f is not None:
            (have if f.exists() else need).append(path)
    return web.json_response({"have": have, "need": need})


async def _picture_target(request: web.Request) -> tuple[Path | None, web.Response | None]:
    store = request.app[DRAMA_STORE_KEY]
    path = request.query.get("path", "")
    try:
        book = store.book(request.match_info["book"])
    except ValueError as exc:
        return None, web.json_response({"error": str(exc)}, status=400)
    if book is None or path not in picture_refs(book):
        return None, web.json_response({"error": "not a picture of this book"}, status=404)
    f = store.picture_file(request.match_info["book"], path)
    if f is None:
        return None, web.json_response({"error": "unsupported picture type"}, status=415)
    return f, None


async def get_picture(request: web.Request) -> web.StreamResponse:
    f, err = await _picture_target(request)
    if err is not None:
        return err
    if not f.exists():
        return web.json_response({"error": "not uploaded yet"}, status=404)
    # A path always holds the same picture, so browsers may keep it.
    return web.FileResponse(f, headers={"Content-Type": PICTURE_TYPES[f.suffix], "Cache-Control": "private, max-age=86400"})


async def get_cover(request: web.Request) -> web.StreamResponse:
    try:
        path = request.app[DRAMA_STORE_KEY].cover(request.match_info["book"])
    except ValueError:
        path = None
    if path is None:
        return web.json_response({"error": "no cover"}, status=404)
    return web.FileResponse(path, headers={"Content-Type": "image/jpeg"})


async def upload_book(request: web.Request) -> web.Response:
    book_id = request.match_info["book"]
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"error": "Body must be JSON"}, status=400)
    if not isinstance(body.get("chapters"), list):
        return web.json_response({"error": "chapters missing"}, status=400)
    store = request.app[DRAMA_STORE_KEY]
    try:
        old_book = store.book(book_id)
        store.save_book(book_id, body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if old_book is not None:
        async with store.lock(book_id):
            s = store.script(book_id)
            if migrate_ids(old_book, store.book(book_id), s):
                store.save_script(book_id, s)
                logger.info("book %s re-split into %d chapters; analysis moved to the new ids",
                            book_id, len(body["chapters"]))
    if body.get("reanalyze") or body.get("relabel"):
        async with store.lock(book_id):
            s = store.script(book_id)
            if s.get("status") not in ("queued", "running", "casting"):
                reset_for_analysis(s, keep_dossier=not body.get("reanalyze"))
                store.save_script(book_id, s)
    if body.get("analyze"):
        await request.app[ANALYZER_KEY].enqueue(book_id)
    return web.json_response(ScriptStore.public(store.script(book_id)))


_QID_RE = re.compile(r"^c(\d+)p(\d+)q(\d+)$")


def position_map(old: dict, new: dict) -> dict[tuple[int, int], tuple[int, int]] | None:
    """(chapter, paragraph) in `old` -> in `new`, when both have the same paragraphs in the same order
    and only the chapter boundaries differ (a newer parser split the book differently)."""
    flat_old = [(c, p, t) for c, ch in enumerate(old.get("chapters", [])) for p, t in enumerate(ch.get("paragraphs", []))]
    flat_new = [(c, p, t) for c, ch in enumerate(new.get("chapters", [])) for p, t in enumerate(ch.get("paragraphs", []))]
    if len(flat_old) != len(flat_new) or any(a[2] != b[2] for a, b in zip(flat_old, flat_new)):
        return None
    if [len(ch.get("paragraphs", [])) for ch in old.get("chapters", [])] == \
            [len(ch.get("paragraphs", [])) for ch in new.get("chapters", [])]:
        return None  # same split: nothing to do
    return {(a[0], a[1]): (b[0], b[1]) for a, b in zip(flat_old, flat_new)}


def migrate_ids(old: dict, new: dict, script: dict) -> bool:
    """Move a script's line labels, corrections and positions to a book's new chapter split."""
    pos = position_map(old, new)
    if pos is None:
        return False

    def qid(q: str) -> str:
        m = _QID_RE.match(q)
        if not m:
            return q
        c, p = pos.get((int(m.group(1)), int(m.group(2))), (int(m.group(1)), int(m.group(2))))
        return f"c{c}p{p}q{m.group(3)}"

    for table in ("lines", "corrections"):
        script[table] = {qid(k): v for k, v in script.get(table, {}).items()}
    script["chunks_done"] = []  # chunk keys follow the old split; a finished analysis does not need them
    for c in script.get("characters", {}).values():
        for f in (c.get("dossier") or {}).get("facts", []):
            if isinstance(f.get("at"), list) and len(f["at"]) == 2:
                f["at"] = list(pos.get(tuple(f["at"]), tuple(f["at"])))
    (script.get("dossier_state") or {}).pop("scenes", None)
    if script.get("status") == "done":
        script["narration"] = compute_narration(new, script)
    else:
        script.pop("narration", None)
    return True


def reset_for_analysis(s: dict, keep_dossier: bool) -> None:
    """Start the analysis over. Corrections, names, merges, voices and dossier edits are always kept;
    with keep_dossier only pass 2 (line labels) runs again, using the dossier as it is now."""
    s.update(status="none", progress=0.0, lines={}, chunks_done=[], narrator="", narrator_by_tense={},
             last_tense=None, series_seeded=0)
    if not keep_dossier:
        s.pop("dossier_state", None)
        for c in s["characters"].values():
            c.pop("dossier", None)


async def get_script(request: web.Request) -> web.Response:
    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    try:
        if store.book(book_id) is None:
            return web.json_response({"error": "book not uploaded"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    s = store.script(book_id)
    if s.get("status") == "done" and ("narration" not in s or s.get("mentions_v") != 1
                                      or (s["characters"] and not any(c.get("role") for c in s["characters"].values()))
                                      or s.get("names_v") != 2 or s.get("voice_desc_v") != 1
                                      or s.get("profile_v") != 2):
        async with store.lock(book_id):
            s = store.script(book_id)
            book = store.book(book_id)
            if "narration" not in s:  # analysed before narration existed
                reconcile_narrators(book, s)
                refresh_narration(store, book_id, s)
            if s.get("mentions_v") != 1:  # analysed before quoted terms were told apart from dialogue
                n = mark_mentions(book, s)
                logger.info("%s: %d quoted terms given back to the narration", book_id, n)
            if s.get("names_v") != 2:  # "a prisoner" -> "Prisoner"; "Urano" -> "Myne" (the name the book uses)
                normalize_names(s)
                canonical_names(book, s)
            if s.get("profile_v") != 2:  # dossiers -> voice profiles
                from .profile import migrate as migrate_profiles

                migrate_profiles(s)
            if s.get("voice_desc_v") != 1:  # descriptions left behind by voices chosen or redesigned later
                from .app import VOICES_KEY

                for c in s["characters"].values():
                    follow_voice(c, request.app[VOICES_KEY])
                s["voice_desc_v"] = 1
            if not any(c.get("role") for c in s["characters"].values()):  # analysed before tiers existed
                from . import roles

                roles.apply_rules(book, s)
            store.save_script(book_id, s)
    return web.json_response(ScriptStore.public(s))


async def analyze(request: web.Request) -> web.Response:
    """Queue an analysis. {"relabel": true} re-labels the lines with the current dossier (after
    dossier edits); {"reanalyze": true} also reads the book again for a fresh dossier."""
    book_id = request.match_info["book"]
    store = request.app[DRAMA_STORE_KEY]
    if store.book(book_id) is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    body = await request.json() if request.can_read_body else {}
    if isinstance(body, dict) and (body.get("relabel") or body.get("reanalyze")):
        async with store.lock(book_id):
            s = store.script(book_id)
            if s.get("status") in ("queued", "running", "casting"):
                return web.json_response({"error": "Busy; try again when it is done"}, status=409)
            reset_for_analysis(s, keep_dossier=not body.get("reanalyze"))
            store.save_script(book_id, s)
    await request.app[ANALYZER_KEY].enqueue(book_id)
    return web.json_response(ScriptStore.public(store.script(book_id)))


async def patch_line(request: web.Request) -> web.Response:
    store = request.app[DRAMA_STORE_KEY]
    book_id, qid = request.match_info["book"], request.match_info["line"]
    body = await request.json()
    async with store.lock(book_id):
        s = store.script(book_id)
        fix = s["corrections"].setdefault(qid, {})
        if "speaker" in body:
            sp = body["speaker"]
            if sp and sp not in s["characters"]:
                sp = ensure_character(s, str(sp), exact=True)  # typed by the reader: kept as typed
                if sp in s["characters"] and s["characters"][sp]["name"] == " ".join(str(body["speaker"]).split()):
                    s["characters"][sp]["name_user"] = True
            fix["speaker"] = sp or None
        if "delivery" in body:
            fix["delivery"] = (body.get("delivery") or "").strip() or None
        if s.get("status") == "done":
            refresh_narration(store, book_id, s)
        store.save_script(book_id, s)
    return web.json_response(ScriptStore.public(s))


async def patch_character(request: web.Request) -> web.Response:
    store = request.app[DRAMA_STORE_KEY]
    book_id, cid = request.match_info["book"], request.match_info["character"]
    body = await request.json()
    async with store.lock(book_id):
        s = store.script(book_id)
        c = s["characters"].get(cid)
        if c is None:
            return web.json_response({"error": "no such character"}, status=404)
        if body.get("name") and body["name"] != c["name"]:
            c.setdefault("aliases", []).append(c["name"])
            c["name"] = body["name"]
            c["name_user"] = True  # typed by the reader: never re-cased
        if "voice" in body:
            from .app import VOICES_KEY

            old_voice = c.get("voice")
            c["voice"] = body["voice"] or None
            c["voice_user"] = bool(c["voice"])  # the reader's choice: a series applied later keeps it
            follow_voice(c, request.app[VOICES_KEY])
        if isinstance(body.get("profile"), dict) or isinstance(body.get("dossier"), dict):
            from . import profile

            fields = body.get("profile") if isinstance(body.get("profile"), dict) else profile.from_dossier(body["dossier"])
            profile.set_user_fields(c, fields)
        if isinstance(body.get("aliases"), list):
            c["aliases"] = [a.strip() for a in body["aliases"] if isinstance(a, str) and a.strip()]
        if "role" in body:
            from .roles import set_user_role

            set_user_role(c, body["role"])
        c["user"] = True
        target = body.get("merge_into")
        if target and target in s["characters"] and target != cid:
            t = s["characters"][target]
            t.setdefault("aliases", []).extend([c["name"], *c.get("aliases", [])])
            for table in (s["lines"], s["corrections"]):
                for line in table.values():
                    if line.get("speaker") == cid:
                        line["speaker"] = target
            del s["characters"][cid]
            if s.get("status") == "done":
                refresh_narration(store, book_id, s)
        store.save_script(book_id, s)
    if "merge_into" not in body:
        from .series import propagate_voice, remember

        if "voice" in body:  # the same character in the series' other volumes follows
            await propagate_voice(request.app, book_id, cid, old_voice, body["voice"] or None)
        remember(request.app, book_id, only=cid)
    return web.json_response(ScriptStore.public(s))


async def voice_from_dossier(request: web.Request) -> web.Response:
    """Design a new voice for a character from its (edited) dossier and assign it.
    The old voice is left untouched, so it can be reassigned."""
    from .app import design_voice
    from .engines.manager import EngineError
    from .series import remember

    store = request.app[DRAMA_STORE_KEY]
    book_id, cid = request.match_info["book"], request.match_info["character"]
    body = await request.json() if request.can_read_body else {}
    from .activity import touch

    touch(request.app, design=True)
    s = store.script(book_id)
    c = s["characters"].get(cid)
    if c is None:
        return web.json_response({"error": "no such character"}, status=404)
    engine = (body or {}).get("engine") or request.app[CONFIG_KEY_REF].voice_design_engine
    from .profile import effective as profile_of, voice_description

    description = voice_description(c) if profile_of(c) else character_description(c)
    try:
        v = await design_voice(request.app, engine, description, c["name"], source="character",
                               voice_id=f"ch-{book_id[:8]}-{cid}"[:55] + f"-{int(time.time()) % 100000}",
                               seed=zlib.crc32(cid.encode()) % 100000)
    except EngineError as exc:
        return web.json_response({"error": exc.detail}, status=exc.status)
    async with store.lock(book_id):
        s = store.script(book_id)
        old_voice = None
        if cid in s["characters"]:
            old_voice = s["characters"][cid].get("voice")
            s["characters"][cid].update(voice=v.id, description=description, user=True, voice_user=True)
        store.save_script(book_id, s)
    from .series import propagate_voice

    await propagate_voice(request.app, book_id, cid, old_voice, v.id)
    remember(request.app, book_id, only=cid)
    return web.json_response(ScriptStore.public(s))


async def run_cast(app: web.Application, book_id: str, engine: str, upgrade: set[str] | None = None,
                   previous: str = "done", background: bool = False) -> None:
    """Creates the cast's voices (status "casting" while it runs) and stores the cast in the series.
    `background`: casting after an analysis, which waits while someone is listening."""
    from .engines.manager import EngineError
    from .series import remember

    store = app[DRAMA_STORE_KEY]
    try:
        result = await cast_voices(app, book_id, engine, upgrade_roles=upgrade, background=background)
        detail = f"{len(result['created'])} voices created"
        remember(app, book_id)
    except EngineError as exc:
        detail = "voice creation failed: " + exc.detail.strip().splitlines()[0][:200]
        logger.warning("casting %s failed: %s", book_id, exc.detail)
    except Exception as exc:  # noqa: BLE001
        logger.exception("casting failed")
        detail = f"voice creation failed: {str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}"
    async with store.lock(book_id):
        fresh = store.script(book_id)
        fresh.update(status=previous if previous != "casting" else "done", detail=detail)
        fresh.pop("auto_cast", None)
        store.save_script(book_id, fresh)


async def cast(request: web.Request) -> web.Response:
    book_id = request.match_info["book"]
    store = request.app[DRAMA_STORE_KEY]
    body = await request.json() if request.can_read_body else {}
    engine = (body or {}).get("engine") or request.app[CONFIG_KEY_REF].voice_design_engine
    # {"own_voices_for": ["supporting"]}: those tiers' characters that share a pool voice get their own.
    upgrade = {r for r in (body or {}).get("own_voices_for") or [] if r in ("main", "supporting", "extra")}
    async with store.lock(book_id):
        s = store.script(book_id)
        if s.get("status") == "casting":
            return web.json_response(ScriptStore.public(s), status=202)
        if s.get("status") in ("queued", "running"):
            return web.json_response({"error": "Wait for the analysis to finish"}, status=409)
        previous = s.get("status", "none")
        s.update(status="casting", detail="creating character voices")
        store.save_script(book_id, s)

    task = asyncio.ensure_future(run_cast(request.app, book_id, engine, upgrade or None, previous))
    request.app[CAST_TASKS_KEY].add(task)
    task.add_done_callback(request.app[CAST_TASKS_KEY].discard)
    return web.json_response(ScriptStore.public(store.script(book_id)), status=202)


CONFIG_KEY_REF = None  # set in setup()


def setup(app: web.Application) -> None:
    global CONFIG_KEY_REF
    from .app import CONFIG_KEY

    CONFIG_KEY_REF = CONFIG_KEY
    cfg = app[CONFIG_KEY]
    store = ScriptStore(cfg.data_path / "books")
    app[DRAMA_STORE_KEY] = store
    analyzer = Analyzer(app, store, LocalLlm(cfg, None))
    app[ANALYZER_KEY] = analyzer
    app[CAST_TASKS_KEY] = set()

    async def _ctx(app: web.Application):
        analyzer.start()
        # Resume analyses interrupted by a restart.
        for d in store.root.iterdir() if store.root.exists() else []:
            s = store.script(d.name) if (d / "script.json").exists() else None
            if s and s.get("status") in ("queued", "running"):
                s["status"] = "none"
                store.save_script(d.name, s)
                await analyzer.enqueue(d.name)
            elif s and s.get("status") == "casting":  # automatic casting cut short by a restart: carry on
                analyzer._spawn_cast(d.name, s.get("auto_cast_engine") or cfg.voice_design_engine)
        yield
        for t in app[CAST_TASKS_KEY]:
            t.cancel()
        await analyzer.stop()

    app.cleanup_ctx.append(_ctx)
    r = app.router
    r.add_get("/api/gpu", gpu_status)
    r.add_get("/api/books", list_books)
    r.add_post("/api/books/{book}/characters/{character}/voice-from-dossier", voice_from_dossier)
    from . import series

    series.setup(app)
    r.add_get("/api/books/{book}", get_book)
    r.add_get("/api/books/{book}/cover", get_cover)
    r.add_get("/api/books/{book}/pictures", list_pictures)
    r.add_get("/api/books/{book}/picture", get_picture)
    r.add_post("/api/books/{book}", upload_book)
    r.add_get("/api/books/{book}/script", get_script)
    r.add_post("/api/books/{book}/analyze", analyze)
    r.add_patch("/api/books/{book}/lines/{line}", patch_line)
    r.add_patch("/api/books/{book}/characters/{character}", patch_character)
    r.add_post("/api/books/{book}/cast", cast)
    from .roles import classify_route

    r.add_post("/api/books/{book}/roles", classify_route)
