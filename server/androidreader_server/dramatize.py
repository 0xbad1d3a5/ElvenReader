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


_POV_RE = re.compile(r"\bPOV\b|point of view|['\u2019]s side\b|\bside\s*[:\u2014-]", re.IGNORECASE)


def is_pov_marker(text: str) -> bool:
    """A short heading such as "(The Neighbor's POV)" that switches the first-person narrator."""
    return len(text) <= 60 and bool(_POV_RE.search(text))


def make_chunks(book: dict, max_words: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ci, ch in enumerate(book.get("chapters", [])):
        paras: list[str] = ch.get("paragraphs", [])
        by_para: dict[int, list] = {}
        for q in ch.get("quotes", []):
            qid, p, s, e = q
            by_para.setdefault(int(p), []).append((str(qid), int(p), int(s), int(e)))
        current: list[int] = []
        words = 0
        k = 0

        def flush():
            nonlocal current, words, k
            if any(p in by_para for p in current):
                chunks.append(Chunk(f"c{ci}k{k}", ci, current, [q for p in current for q in by_para.get(p, [])]))
                k += 1
            current, words = [], 0

        for pi, text in enumerate(paras):
            # A point-of-view change starts a new chunk, so each chunk has one narrator.
            if current and is_pov_marker(text):
                flush()
            current.append(pi)
            words += len(text.split())
            if words >= max_words:
                flush()
        if current:
            flush()
    return chunks


def pov_change(book: dict, chunk: Chunk) -> str | None:
    """The point-of-view heading this chunk starts with (or its chapter title), if any."""
    chapter = book["chapters"][chunk.chapter]
    first = chapter["paragraphs"][chunk.paragraphs[0]]
    if is_pov_marker(first):
        return first.strip()
    if chunk.paragraphs[0] == 0 and is_pov_marker(chapter.get("title", "")):
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


ROSTER_PROMPT = """You are casting voices for an audiobook. For each character, using their lines and how the narration describes them, give:
gender, age group, accent (e.g. "british", "american", "scottish", "neutral"), temperament (2-4 words), and a voice description for a voice-design model: one sentence covering gender, age, pitch, pace, timbre and manner, e.g. "A middle-aged woman with a high, slightly shrill voice, talking quickly and fussily, with a refined English accent." Keep each character distinct from the others."""

ROSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "gender": {"type": "string", "enum": GENDERS},
                    "age": {"type": "string", "enum": AGES},
                    "accent": {"type": "string"},
                    "temperament": {"type": "string"},
                    "voice": {"type": "string"},
                },
                "required": ["name", "gender", "age", "accent", "temperament", "voice"],
            },
        }
    },
    "required": ["characters"],
}


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
        (d / "meta.json").write_text(json.dumps(book_summary(payload) | {"uploaded": time.time()},
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
    Third-person books (little or no evidence) get an empty list.
    """
    lines = ScriptStore.public(script)["lines"]
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

    @property
    def url(self) -> str:
        return self.cfg.llm_url.rstrip("/")

    async def _alive(self, session) -> bool:
        try:
            async with session.get(self.url.rsplit("/v1", 1)[0] + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    async def ensure(self, session) -> None:
        if self.proc is not None and self.proc.returncode is None and self.running_gpu != self.gpu:
            await self.stop()
        if await self._alive(session):
            return
        model = Path(self.cfg.llm_model) if self.cfg.llm_model else None
        if model and not model.is_absolute():
            model = self.cfg.models_path / "llm" / model
        exe = self.cfg.models_path / "llama.cpp" / ("cuda" if self.gpu else "cpu") / "llama-server.exe"
        if not model or not model.exists() or not exe.exists():
            raise LlmError(
                "No local LLM configured. Run download-llm.ps1, or point llm_url at a running "
                "OpenAI-compatible server (LM Studio, Ollama, llama-server)."
            )
        port = int(self.url.rsplit(":", 1)[1].split("/")[0])
        args = [str(exe), "-m", str(model), "--port", str(port), "--host", "127.0.0.1",
                "-c", str(self.cfg.llm_context), "-t", str(self.cfg.llm_threads), "--jinja", "-np", "1"]
        if self.gpu:
            args += ["-ngl", "99", "--n-cpu-moe", str(self.cfg.llm_cpu_moe)]
        else:
            args += ["-ngl", "0"]
        logger.info("starting LLM: %s", " ".join(args))
        log = open(self.cfg.data_path / "llm.log", "ab")
        self.proc = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log,
                                                         creationflags=0x08000000)
        self.running_gpu = self.gpu
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

    async def complete(self, session, system: str, user: str, schema: dict, max_tokens: int = 2048) -> dict:
        await self.ensure(session)
        body = {
            "model": self.model_name or "local",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
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


async def label_chunk(llm: LocalLlm, session, book: dict, script: dict, chunk: Chunk) -> tuple[dict, dict[str, str]]:
    text, local = render_chunk(book, chunk)
    chapter = book["chapters"][chunk.chapter]
    narrator = script.get("narrator") or "unknown (work it out from the passage)"
    pov = pov_change(book, chunk)
    if pov:
        narrator = (f"{narrator}. But this passage starts with a point-of-view change, \"{pov}\": from here "
                    f"\"I\" is that character, not the previous narrator. Lines \"I\" says belong to them.")
    user = (
        f"Book: {book.get('title', '')} by {book.get('author', '')}\n"
        f"Chapter: {chapter.get('title', '')}\n\n"
        f"Known characters:\n{roster_text(script)}\n\n"
        f"Narrator of the previous passage: {narrator}\n\n"
        f"Most recent lines before this passage:\n{recent_text(script, book, chunk)}\n\n"
        f"Passage:\n{text}\n\n"
        f"Label every number from 1 to {len(local)} once."
    )
    # Generous budget: reasoning models (gpt-oss) spend tokens before the JSON, and the
    # schema grammar ends the output as soon as the JSON is complete anyway.
    budget = 1024 + 60 * len(local)
    try:
        result = await llm.complete(session, SYSTEM_PROMPT, user, chunk_schema(list(local)), max_tokens=budget)
    except LlmError as exc:
        if "invalid JSON" not in str(exc):
            raise
        result = await llm.complete(session, SYSTEM_PROMPT, user, chunk_schema(list(local)), max_tokens=budget * 3)
    return result, local


async def describe_cast(llm: LocalLlm, session, book: dict, script: dict, max_chars: int = 12) -> None:
    public = ScriptStore.public(script)
    main = [c for c in public["characters"] if c["lines"] >= 2][:max_chars]
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
        blocks.append(f"## {c['name']} ({c.get('gender')}, {c.get('age')})\n" + "\n".join(f"- {t}" for t in texts))
    user = f"Book: {book.get('title', '')} by {book.get('author', '')}\n\n" + "\n\n".join(blocks)
    result = await llm.complete(session, ROSTER_PROMPT, user, ROSTER_SCHEMA, max_tokens=1024 + 200 * len(main))
    for d in result.get("characters", []):
        cid = find_character(script, d.get("name", ""))
        if cid is None:
            continue
        c = script["characters"][cid]
        for key in ("gender", "age", "accent", "temperament"):
            if d.get(key):
                c[key] = d[key]
        c["description"] = d.get("voice", "")


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
        return await self._gpu(lambda: label_chunk(self.llm, session, book, script, chunk))

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
        chunks = make_chunks(book)
        t0 = time.perf_counter()
        for i, chunk in enumerate(chunks):
            s = self.store.script(book_id)
            if chunk.key in s["chunks_done"]:
                continue
            s.update(status="running", progress=round(i / max(1, len(chunks)), 3),
                     detail=f"chapter {chunk.chapter + 1}: {len(s['lines'])} lines so far")
            self.store.save_script(book_id, s)
            result, local = await self._label(session, book, s, chunk)
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                apply_chunk_result(s, result, local)
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
                    for k in ("gender", "age", "accent", "temperament", "description"):
                        fresh["characters"][cid][k] = c.get(k, fresh["characters"][cid].get(k))
            fresh.update(status="done", progress=1.0,
                         detail=f"{len(fresh['lines'])} lines in {time.perf_counter() - t0:.0f}s")
            fresh.setdefault("stats", {})["seconds"] = round(time.perf_counter() - t0, 1)
            fresh["narration"] = compute_narration(book, fresh)
            self.store.save_script(book_id, fresh)
        logger.info("analysed %s: %d lines", book_id, len(fresh["lines"]))


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
    if c.get("description"):
        return c["description"]
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
    store = request.app[DRAMA_STORE_KEY]
    books = []
    for book_id in store.ids():
        summary = store.summary(book_id) or {}
        s = store.script(book_id)
        public = ScriptStore.public(s)
        books.append({
            **summary,
            "id": book_id,
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
    if body.get("reanalyze"):
        async with store.lock(book_id):
            s = store.script(book_id)
            if s.get("status") not in ("queued", "running", "casting"):
                # Start over; user corrections, names and voices are kept.
                s.update(status="none", progress=0.0, lines={}, chunks_done=[], narrator="")
                store.save_script(book_id, s)
    if body.get("analyze"):
        await request.app[ANALYZER_KEY].enqueue(book_id)
    return web.json_response(ScriptStore.public(store.script(book_id)))


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
            refresh_narration(store, book_id, s)
            store.save_script(book_id, s)
    return web.json_response(ScriptStore.public(s))


async def analyze(request: web.Request) -> web.Response:
    book_id = request.match_info["book"]
    if request.app[DRAMA_STORE_KEY].book(book_id) is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    await request.app[ANALYZER_KEY].enqueue(book_id)
    return web.json_response(ScriptStore.public(request.app[DRAMA_STORE_KEY].script(book_id)))


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
    r.add_get("/api/books/{book}", get_book)
    r.add_get("/api/books/{book}/cover", get_cover)
    r.add_post("/api/books/{book}", upload_book)
    r.add_get("/api/books/{book}/script", get_script)
    r.add_post("/api/books/{book}/analyze", analyze)
    r.add_patch("/api/books/{book}/lines/{line}", patch_line)
    r.add_patch("/api/books/{book}/characters/{character}", patch_character)
    r.add_post("/api/books/{book}/cast", cast)
