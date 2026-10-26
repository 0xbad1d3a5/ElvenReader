"""Dramatization: who says each quotation in a book, and how.

The phone uploads a book's paragraphs together with the quotations it found
(stable IDs and offsets, so highlighting always matches the EPUB). A local LLM
then labels each quotation, chunk by chunk, with a speaker and a short delivery
("shouting, urgent"). The LLM never rewrites text: it only returns labels for
the IDs it was given, constrained by a JSON schema. Results are saved after
every chunk, so playback can use them while the analysis is still running.

A roster pass then describes the main characters (gender, age, accent,
temperament) and voices are created for them with a voice-design model; minor
characters share a small pool of voices. Users can rename, merge and re-voice
characters, and correct any line; corrections always win.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

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
    """Python twin of the app's Quotes.regions (used by the evaluation tools)."""
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
    """A short heading such as "(The Neighbor's POV)" that switches the first-person narrator."""
    return len(text) <= 60 and bool(_POV_RE.search(text))


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
- speaker: who says it, as one name without parentheses. Use the exact roster name for known characters, even when the passage calls them by a nickname or title the roster lists under "also called". If a new character speaks, use the name the text uses for them (e.g. "the Dormouse", "Mrs. Long"); if they are known by a real name and a nickname, give the real name and put the nickname in their aliases. If an unnamed person speaks, describe them briefly ("a footman"). A character's quoted thoughts ("..." thought Alice) belong to that character. Use "NARRATOR" only for quotations nobody speaks or thinks (titles, signs, written notes, quoted words or phrases).
- delivery: how an actor should perform the line, 1-4 lowercase words such as "shouting, urgent", "whispering", "sarcastic", "tearful", "excited", "nervous", "cold, angry", "teasing", "calm". Use "neutral" when nothing special is called for. Base it on speech verbs and adverbs near the quote ("she shrieked", "he said softly"), punctuation and context.
Rules: in a back-and-forth conversation without speech tags, speakers usually alternate. A number appearing twice marks one quotation interrupted by narration ("Yes," [3] he said, [3] "I will."): label it once. Never change the text; label each number once, in order, giving the first two or three words of its quotation in "starts" so you keep your place. If several characters speak together, give the main one.
First-person narration: when the story is told by "I", every line the narrator speaks ("...," I said / I asked / I replied, or "I" answering in a conversation) belongs to the narrating character, so give that character's name (never "I", and never another character). Work out who "I" is from the text and the narrator given for the previous passage; chapter titles such as "(Mary's POV)" mean that character narrates. Report the narrator in "narrator" (their name, or "third person" if the passage is not told by "I").
Also list every character who speaks in this passage with their gender, age group and any other names, titles or nicknames the passage uses for them (aliases)."""


def chunk_schema(local_ids: list[str]) -> dict:
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
        """What the phone sees: lines with corrections applied, characters with line counts."""
        lines = {}
        for qid, line in script.get("lines", {}).items():
            lines[qid] = {"speaker": line.get("speaker"), "delivery": line.get("delivery"), "corrected": False}
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


def tense_runs(book: dict, window: int = 4, min_run: int = 12) -> list[tuple[int, int, str]]:
    """Where the narration switches between past and present tense: [(chapter, paragraph, tense)].

    Scores each paragraph by present- minus past-tense markers (dialogue removed),
    smooths the score over neighbouring paragraphs, absorbs runs shorter than
    `min_run` paragraphs (inner thoughts, system messages) and snaps switches to
    nearby point-of-view headings.
    """
    labels: list[tuple[int, int, str | None]] = []
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
    for r in runs[1:]:
        c, p, _ = labels[r[0]]
        near = [position[m] for m in markers if m[0] == c and abs(m[1] - p) <= 8]
        if near:
            r[0] = min(near, key=lambda i: abs(i - r[0]))
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


def ensure_character(script: dict, name: str, gender: str = "unknown", age: str = "unknown",
                     aliases: list[str] | tuple = ()) -> str | None:
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
        script["characters"][cid] = {"name": name.strip(), "aliases": [], "gender": gender, "age": age,
                                     "accent": "", "temperament": "", "description": "", "voice": None}
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
            n += 1
    return n


def split_group(name: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*(?:,|&|\band\b)\s*", name or "") if p.strip()]
    return parts or [name or ""]


# ----------------------------------------------------------------- LLM


class LlmError(Exception):
    pass


class LocalLlm:
    """OpenAI-compatible chat endpoint (llama.cpp's llama-server), started on demand."""

    def __init__(self, cfg, session_factory=None):
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self.model_name = Path(cfg.llm_model).stem if cfg.llm_model else ""
        # Where the next request should run; the server is restarted when this changes.
        self.gpu = cfg.llm_mode == "gpu"
        self.running_gpu: bool | None = None
        self.running: tuple | None = None  # (model file, context) of the running server

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
        if self.proc is not None and self.proc.returncode is None and (
                self.running_gpu != self.gpu or self.running != (model_file, context)):
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
        if self.gpu:
            args += ["-ngl", "99", "--n-cpu-moe", str(self.cfg.llm_cpu_moe)]
        else:
            args += ["-ngl", "0"]
        logger.info("starting LLM: %s", " ".join(args))
        log = open(self.cfg.data_path / "llm.log", "ab")
        self.proc = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log,
                                                         creationflags=0x08000000)
        self.running_gpu = self.gpu
        self.running = (model_file, context)
        for _ in range(600):
            await asyncio.sleep(1)
            if self.proc.returncode is not None:
                raise LlmError("llama-server exited; see data/llm.log")
            if await self._alive(session):
                return
        raise LlmError("llama-server did not start in 10 minutes")

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
        try:
            async with session.post(f"{self.url}/chat/completions", json=body, timeout=1800) as r:
                data = await r.json(content_type=None)
                if r.status >= 400:
                    raise LlmError(f"LLM HTTP {r.status}: {str(data)[:300]}")
        except asyncio.TimeoutError as exc:
            raise LlmError("LLM timed out") from exc
        content = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage", {})
        try:
            parsed = json.loads(content[content.find("{"):])
        except ValueError as exc:
            raise LlmError(f"LLM returned invalid JSON: {content[:200]}") from exc
        parsed["_usage"] = usage
        parsed["_timings"] = data.get("timings", {})
        return parsed


async def label_chunk(llm: LocalLlm, session, book: dict, script: dict, chunk: Chunk,
                      thinking: bool = False, model: str | None = None) -> tuple[dict, dict[str, str]]:
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
    if script.get("dossier_state") or any(c.get("dossier_user") for c in script["characters"].values()):
        from .dossier import render_dossier, scene_cast

        present = [script["characters"][c]["name"] for c in scene_cast(script, chunk.chapter, chunk.paragraphs)]
        known = ("Characters (from a first read of the whole book: use their speech styles, verbal tics and "
                 "relationships to recognise who speaks in untagged dialogue, and \"first appears as\" to give a "
                 "character's real name even before the book reveals it):\n"
                 f"{render_dossier(script, limit=30, width=110)}"
                 + (f"\n\nPresent in this part of the story: {', '.join(present)}" if present else ""))
    else:
        known = f"Known characters:\n{roster_text(script)}"
    user = (
        f"Book: {book.get('title', '')} by {book.get('author', '')}\n"
        f"Chapter: {chapter.get('title', '')}\n\n"
        f"{known}\n\n"
        f"Narrator of the previous passage: {narrator}\n\n"
        f"Most recent lines before this passage:\n{recent_text(script, book, chunk)}\n\n"
        f"Passage:\n{text}\n\n"
        f"Label every number from 1 to {len(local)} once."
    )
    # Generous budget: reasoning models (gpt-oss) spend tokens before the JSON, and the
    # schema grammar ends the output as soon as the JSON is complete anyway.
    budget = 1024 + 60 * len(local)
    extra = {"thinking": True} if thinking else {}
    if model:
        extra["model"] = model
    if thinking:
        budget += 8000
    try:
        result = await llm.complete(session, SYSTEM_PROMPT, user, chunk_schema(list(local)), max_tokens=budget,
                                    **extra)
    except LlmError as exc:
        if "invalid JSON" not in str(exc):
            raise
        result = await llm.complete(session, SYSTEM_PROMPT, user, chunk_schema(list(local)), max_tokens=budget * 3,
                                    **extra)
    return result, local


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
        if raw.get("dossier") or raw.get("dossier_user"):
            from .dossier import effective

            d = effective(raw)
            about = "\n".join(f"- {k.replace('_', ' ')}: {d[k]}" for k in
                              ("gender", "apparent_age", "form", "true_nature", "personality", "speech_style")
                              if d.get(k))
            blocks.append(f"## {c['name']}" + (f" (also called: {', '.join(aliases)})" if aliases else "") + "\n"
                          + "Dossier (from reading the whole book; follow its gender and apparent age):\n" + about
                          + "\nLines:\n" + "\n".join(f"- {t}" for t in texts))
            continue
        looks = appearance_evidence(book, [c["name"], *aliases])
        blocks.append(
            f"## {c['name']}" + (f" (also called: {', '.join(aliases)})" if aliases else "") + "\n"
            + "Described as:\n" + ("\n".join(f"- {t}" for t in looks) or "- (nothing found)") + "\n"
            + "Lines:\n" + "\n".join(f"- {t}" for t in texts))
    user = f"Book: {book.get('title', '')} by {book.get('author', '')}\n\n" + "\n\n".join(blocks)
    result = await llm.complete(session, ROSTER_PROMPT, user, ROSTER_SCHEMA, max_tokens=1024 + 300 * len(main))
    for d in result.get("characters", []):
        cid = find_character(script, d.get("name", ""))
        if cid is None:
            continue
        c = script["characters"][cid]
        for key in ("gender", "age", "accent", "temperament", "appearance"):
            if d.get(key):
                c[key] = d[key]
        c["description"] = d.get("voice", "")
        if c.get("dossier") or c.get("dossier_user"):
            from .dossier import effective, sync_character

            sync_character(c)  # the dossier's apparent gender and age win
            if effective(c).get("voice"):
                c["description"] = effective(c)["voice"]


# ----------------------------------------------------------------- analysis job


class Analyzer:
    def __init__(self, app: web.Application, store: ScriptStore, llm: LocalLlm):
        self.app = app
        self.store = store
        self.llm = llm
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.current: str | None = None

    def start(self) -> None:
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

    async def _label(self, session, book: dict, script: dict, chunk: Chunk):
        thinking = self.llm.cfg.llm_label_thinking
        return await self._gpu(lambda: label_chunk(self.llm, session, book, script, chunk, thinking=thinking))

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

    async def _gpu(self, make_coro):
        """Runs one LLM step on the GPU when allowed and the voice model has been idle,
        otherwise on the CPU alongside speech synthesis."""
        from .app import ENGINES_KEY

        manager = self.app[ENGINES_KEY]
        mode = self.llm.cfg.llm_mode
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
            s.update(detail="describing the characters")
            self.store.save_script(book_id, s)
        s = self.store.script(book_id)
        await self._gpu(lambda: describe_cast(self.llm, session, book, s))
        if self.llm.running_gpu:
            await self.llm.stop()  # give the GPU back to TTS
        async with self.store.lock(book_id):
            fresh = self.store.script(book_id)
            for cid, c in s["characters"].items():
                if cid in fresh["characters"]:
                    for k in ("gender", "age", "accent", "temperament", "appearance", "description"):
                        fresh["characters"][cid][k] = c.get(k, fresh["characters"][cid].get(k))
            fresh.update(status="done", progress=1.0,
                         detail=f"{len(fresh['lines'])} lines in {time.perf_counter() - t0:.0f}s")
            fresh.setdefault("stats", {})["seconds"] = round(time.perf_counter() - t0, 1)
            reconcile_narrators(book, fresh)
            fresh["narration"] = compute_narration(book, fresh)
            prune_characters(fresh)
            self.store.save_script(book_id, fresh)
        logger.info("analysed %s: %d lines", book_id, len(fresh["lines"]))


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


def character_description(c: dict) -> str:
    user_voice = (c.get("dossier_user") or {}).get("voice")
    if user_voice:
        return user_voice
    if c.get("description"):
        return c["description"]
    if c.get("dossier"):
        from .dossier import voice_description

        return voice_description(c)
    parts = [c.get("gender"), c.get("age"), c.get("accent") and f"{c['accent']} accent", c.get("temperament")]
    return ", ".join(p for p in parts if p and p != "unknown") or "an adult speaker with a natural voice"


async def cast_voices(app: web.Application, book_id: str, engine: str, main_lines: int = 5) -> dict:
    """Design voices for main characters without one; minor ones get pool voices."""
    from .app import VOICES_KEY, design_voice

    store: ScriptStore = app[DRAMA_STORE_KEY]
    voices = app[VOICES_KEY]
    s = store.script(book_id)
    public = ScriptStore.public(s)
    lines_by = {c["id"]: c["lines"] for c in public["characters"]}
    narrators = {who for _, _, who in s.get("narration", []) if who}
    created = []
    for cid, c in s["characters"].items():
        if c.get("voice") and voices.get(c["voice"]):
            continue
        if lines_by.get(cid, 0) >= main_lines or cid in narrators:
            vid = f"ch-{book_id[:8]}-{cid}"[:64]
            v = voices.get(vid) or await design_voice(
                app, engine, character_description(c), c["name"], source="character", voice_id=vid,
                seed=zlib.crc32(cid.encode()) % 100000,
            )
            c["voice"] = v.id
            created.append(v.id)
        else:
            g, a = pool_key(c.get("gender", "unknown"), c.get("age", "unknown"))
            vid = f"pool-{g}-{a}".replace(" ", "")
            v = voices.get(vid) or await design_voice(
                app, engine, POOL[(g, a)], f"Pool: {g} {a}", source="character", voice_id=vid, seed=7,
            )
            c["voice"] = v.id
    async with store.lock(book_id):
        fresh = store.script(book_id)
        for cid, c in s["characters"].items():
            if cid in fresh["characters"] and not fresh["characters"][cid].get("voice"):
                fresh["characters"][cid]["voice"] = c.get("voice")
        store.save_script(book_id, fresh)
    return {"created": created}


# ----------------------------------------------------------------- routes

DRAMA_STORE_KEY = web.AppKey("drama_store", ScriptStore)
ANALYZER_KEY = web.AppKey("analyzer", Analyzer)
CAST_TASKS_KEY = web.AppKey("cast_tasks", set)


async def list_books(request: web.Request) -> web.Response:
    from .series import series_name as _series_name

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
        store.save_book(book_id, body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if body.get("reanalyze") or body.get("relabel"):
        async with store.lock(book_id):
            s = store.script(book_id)
            if s.get("status") not in ("queued", "running", "casting"):
                reset_for_analysis(s, keep_dossier=not body.get("reanalyze"))
                store.save_script(book_id, s)
    if body.get("analyze"):
        await request.app[ANALYZER_KEY].enqueue(book_id)
    return web.json_response(ScriptStore.public(store.script(book_id)))


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
    if s.get("status") == "done" and "narration" not in s:  # analysed before narration existed
        async with store.lock(book_id):
            s = store.script(book_id)
            book = store.book(book_id)
            reconcile_narrators(book, s)
            refresh_narration(store, book_id, s)
            store.save_script(book_id, s)
    return web.json_response(with_voice_versions(request.app, ScriptStore.public(s)))


def with_voice_versions(app: web.Application, public: dict) -> dict:
    """Add each cast voice's fingerprint, so clients notice edited voices and drop prepared audio."""
    from .app import VOICES_KEY

    voices = app[VOICES_KEY]
    versions = {}
    for c in public.get("characters", []):
        v = voices.get(c.get("voice"))
        if v is not None:
            versions[v.id] = v.fingerprint
    return {**public, "voice_versions": versions}


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
                sp = ensure_character(s, str(sp))
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
        if "voice" in body:
            c["voice"] = body["voice"] or None
        if isinstance(body.get("dossier"), dict):
            from .dossier import FIELDS, effective, sync_character

            mine = c.setdefault("dossier_user", {})
            for key, value in body["dossier"].items():
                if key not in FIELDS and key != "narrates":
                    continue
                value = (value or "").strip() if isinstance(value, str) else ""
                if value and value != (c.get("dossier") or {}).get(key):
                    mine[key] = value
                elif not value or value == (c.get("dossier") or {}).get(key):
                    mine.pop(key, None)
            sync_character(c)
            if mine.get("voice"):
                c["description"] = effective(c)["voice"]
        if isinstance(body.get("aliases"), list):
            c["aliases"] = [a.strip() for a in body["aliases"] if isinstance(a, str) and a.strip()]
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
        from .series import remember

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
    s = store.script(book_id)
    c = s["characters"].get(cid)
    if c is None:
        return web.json_response({"error": "no such character"}, status=404)
    engine = (body or {}).get("engine") or request.app[CONFIG_KEY_REF].voice_design_engine
    description = character_description(c)
    try:
        v = await design_voice(request.app, engine, description, c["name"], source="character",
                               voice_id=f"ch-{book_id[:8]}-{cid}"[:55] + f"-{int(time.time()) % 100000}",
                               seed=zlib.crc32(cid.encode()) % 100000)
    except EngineError as exc:
        return web.json_response({"error": exc.detail}, status=exc.status)
    async with store.lock(book_id):
        s = store.script(book_id)
        if cid in s["characters"]:
            s["characters"][cid].update(voice=v.id, description=description, user=True)
        store.save_script(book_id, s)
    remember(request.app, book_id, only=cid)
    return web.json_response(ScriptStore.public(s))


async def cast(request: web.Request) -> web.Response:
    book_id = request.match_info["book"]
    store = request.app[DRAMA_STORE_KEY]
    body = await request.json() if request.can_read_body else {}
    engine = (body or {}).get("engine") or request.app[CONFIG_KEY_REF].voice_design_engine
    async with store.lock(book_id):
        s = store.script(book_id)
        if s.get("status") == "casting":
            return web.json_response(ScriptStore.public(s), status=202)
        if s.get("status") in ("queued", "running"):
            return web.json_response({"error": "Wait for the analysis to finish"}, status=409)
        previous = s.get("status", "none")
        s.update(status="casting", detail="creating character voices")
        store.save_script(book_id, s)

    async def run():
        from .engines.manager import EngineError

        try:
            result = await cast_voices(request.app, book_id, engine)
            detail = f"{len(result['created'])} voices created"
            from .series import remember

            remember(request.app, book_id)
        except EngineError as exc:
            detail = f"voice creation failed: {exc.detail}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("casting failed")
            detail = f"voice creation failed: {exc}"
        async with store.lock(book_id):
            fresh = store.script(book_id)
            fresh.update(status=previous if previous != "casting" else "done", detail=detail)
            store.save_script(book_id, fresh)

    task = asyncio.ensure_future(run())
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
        yield
        for t in app[CAST_TASKS_KEY]:
            t.cancel()
        await analyzer.stop()

    app.cleanup_ctx.append(_ctx)
    r = app.router
    r.add_get("/api/books", list_books)
    r.add_post("/api/books/{book}/characters/{character}/voice-from-dossier", voice_from_dossier)
    from . import series

    series.setup(app)
    r.add_get("/api/books/{book}", get_book)
    r.add_get("/api/books/{book}/cover", get_cover)
    r.add_post("/api/books/{book}", upload_book)
    r.add_get("/api/books/{book}/script", get_script)
    r.add_post("/api/books/{book}/analyze", analyze)
    r.add_patch("/api/books/{book}/lines/{line}", patch_line)
    r.add_patch("/api/books/{book}/characters/{character}", patch_character)
    r.add_post("/api/books/{book}/cast", cast)
