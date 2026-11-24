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
import math
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from aiohttp import ClientConnectionError, web

from . import delivery as delivery_tags
from . import sounds as sound_marks

logger = logging.getLogger("elvenreader.drama")

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
    script["mentions_v"] = MENTIONS_V
    return n


MENTIONS_V = 1  # script["mentions_v"]: quoted terms told apart from dialogue


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
# A sentence ends in the narration between two quotations ("…now?” Asuna whispered. Right behind her ear, Kirito
# said, “Sounds good.”): they are two turns, even when the narration ends on a comma. Not after a title ("Mr.").
_SENTENCE_BREAK = re.compile(r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bMt)(?<!\bJr)(?<!\bSr)(?<!\bLt)"
                             r"(?<!\bProf)(?<!\bCapt)(?<!\bSgt)(?<!\bGen)(?<!\bCol)"
                             r"[.!?\u2026][\"'\u201d\u2019)]*\s+[\"'\u201c\u2018(]*[A-Z]")
# Narration that brings in someone else by name ("…went "Sh! sh!" and the Dormouse sulkily remarked, "If you
# can't be civil…"", "and Leafa commented,", "then the Rabbit's voice along—"): a new turn although it ends on a
# comma. A pronoun ("then she asked,") or a named subordinate clause ("Before Miyuki's unease could rear its head
# again, he added,") is still the same speaker.
_NEW_SPEAKER = re.compile(r"^\W*(?:and|but|then)\s+(?:the\s+|a\s+|an\s+)?(?!I\b|Then\b)[A-Z]", re.IGNORECASE)


def interrupts(between: str | None) -> bool:
    """Whether the narration between two quotations of a paragraph only interrupts one quotation ("Yes," he said,
    "I will."): short, ending on a comma or dash, without a sentence ending inside it or another speaker
    brought in by name."""
    return (between is not None and len(between) <= 80 and bool(_CONTINUES.search(between))
            and not _SENTENCE_BREAK.search(between) and not _new_speaker(between))


def _new_speaker(between: str) -> bool:
    m = _NEW_SPEAKER.match(between)
    return bool(m) and between[m.end() - 1].isupper()


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
            if not interrupts(between):
                n += 1
                local[str(n)] = []
            local[str(n)].append(qid)
            pieces.append(text[last:s])
            pieces.append(f"[{n}]{text[s:e]}")
            last = prev_end = e
        pieces.append(text[last:])
        if pi in scene_breaks(book, chunk.chapter):
            out.append(SCENE_BREAK)
        out.extend(silent_lines(book, chunk.chapter, pi))
        out.append("".join(pieces))
    return "\n\n".join(out), local


# The EPUB's scene breaks (an ornament, or a blank line the publisher marks), shown to the LLM on a line of their own.
SCENE_BREAK = "~~~ SCENE BREAK: a new scene starts here ~~~"


def scene_breaks(book: dict, ci: int) -> set[int]:
    """The paragraphs of chapter `ci` that start a new scene (book.json "breaks", from the EPUB)."""
    return set(book["chapters"][ci].get("breaks") or [])


def with_breaks(book: dict, ci: int, paragraphs) -> str:
    """Paragraphs of chapter `ci` as text for the LLM, with SCENE_BREAK before each that starts a new scene and the
    wordless lines ("…?") the book has before them."""
    breaks, paras = scene_breaks(book, ci), book["chapters"][ci]["paragraphs"]
    out = []
    for p in paragraphs:
        if p in breaks:
            out.append(SCENE_BREAK)
        out.extend(silent_lines(book, ci, p))
        out.append(paras[p])
    return "\n\n".join(out)


def silent_lines(book: dict, ci: int, pi: int) -> list[str]:
    """Wordless lines ("“…?”", "“……”") the book has just before paragraph `pi` (book.json "silent"): nothing is said,
    but someone reacts. Not numbered: they have no speaker to label or speech to make."""
    return list((book["chapters"][ci].get("silent") or {}).get(str(pi), []))


# ----------------------------------------------------------------- names


def norm_name(name: str) -> str:
    s = name.lower().replace(".", "").strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return " ".join(s.split())


def slug(name: str) -> str:
    return _SLUG_RE.sub("-", norm_name(name)).strip("-")[:40] or "someone"


# ----------------------------------------------------------------- prompts

# How the LLM chooses a line's delivery tags: part of SYSTEM_PROMPT, and of DELIVERY_PROMPT (re-scoring deliveries).
DELIVERY_RULE = """- delivery: how an actor should perform the line, as tags from the approved list below (nothing else). Give tags only when the situation and context call for the voice to carry emotion or a special manner (a quarrel, fear, grief, a shout across a room, a whisper, an inner thought), and then only the one or two that matter most, the most important first; never more than two. Ordinary conversation, explanations, questions and plain remarks get [], even in a tense scene: the words carry the feeling, and every tag changes how the whole line is voiced. A question is already voiced as a question: "confused" only for real bewilderment ("Wha...? Where am I?"). Concern or worry for someone is "tender", not "afraid". "firm" only for orders and stern warnings. When in doubt, give []. Clear signs: a speech verb or adverb near the quote ("she shrieked" -> ["afraid", "shout"]; "he said softly" -> ["soft"]; "she laughed" -> ["amused", "laugh"]; "he sighed" -> ["sigh"]; quoted thoughts -> ["thought"]). A sound tag only when the narration says the speaker makes that sound. At most one feeling (it may start with "slightly" or "very"), one volume, one pace/manner, one sound, one situation. Approved tags (each covers the words in brackets):
{delivery_tags}""".replace("{delivery_tags}", delivery_tags.prompt_list())

SYSTEM_PROMPT = """You label dialogue in novels for a multi-voice audiobook.
For every numbered quotation [n] in the passage, decide:
- speaker: who says it, as one name without parentheses. Use the exact roster name for known characters, even when the passage calls them by a nickname or title the roster lists under "also called". But a title or role can belong to several people (two commanders, many knights, several priests): when the passage names or clearly identifies a different person, use that person's name, even if the roster lists the title for someone else. If a new character speaks, use the name the text uses for them (e.g. "the Dormouse", "Mrs. Long"); if they are known by a real name and a nickname, give the real name and put the nickname in their aliases. If an unnamed person speaks, describe them briefly ("a footman"). A character's quoted thoughts ("..." thought Alice) belong to that character. Use "NARRATOR" only for quotations nobody speaks or thinks aloud at that moment: titles, signs, written notes, and words or phrases set in quotation marks inside a sentence (a term, a nickname, scare quotes: the so-called "experts"; what she calls "fun"; French for "dog."; his "plan" failed). Such quoted terms are part of the narration and are NARRATOR even in first-person passages and even when the text says who coined the word.
{delivery_rule}
- respell: [] for almost every quotation. Only when a word is spelled the way it sounds, so a speech engine reading its letters would garble it (drawn-out letters: "whaaaaaaat", "Nooooo", "pleeease", "so-o-o"), list that one word (just the word, no punctuation) as written and as the plain word ({"written": "whaaaaaaat", "said": "what"}), and put the drawing-out in the delivery (["shout", "slow"]). Stutters ("Y-yeah", "Wh… what"), ellipses, and sounds that are words of their own (Hmm, Shh, Zzz, Grr, Kyaa, growls, animal noises) stay as they are.
- sounds: [] for most quotations. When the quotation itself writes out a non-verbal sound (laughter, a sigh, a scream, a hesitation, a groan: "Ahaha, that's fine.", "Haaah... Really.", "Kyaaah!"), list each such word (just the word, as written, no punctuation) with its kind ({"written": "Ahaha", "sound": "laugh"}). Ordinary words are not sounds ("Wow", "Yes", "Oh no", names); "Uh-huh" and "Mm-hmm" meaning yes are words. A sound only the narration mentions ("she laughed") goes in the delivery instead, never here. Kinds:
{sound_kinds}
Rules: in a back-and-forth conversation without speech tags, speakers usually alternate. A number appearing twice marks one quotation interrupted by narration ("Yes," [3] he said, [3] "I will."): label it once. Never change the text; label each number once, in order, giving the first two or three words of its quotation in "starts" so you keep your place. If several characters speak together, give the main one.
First-person narration: when the story is told by "I", every line the narrator speaks ("...," I said / I asked / I replied, or "I" answering in a conversation) belongs to the narrating character, so give that character's name (never "I", and never another character). Work out who "I" is from the text and the narrator given for the previous passage; chapter titles such as "(Mary's POV)" mean that character narrates. A line "~~~ SCENE BREAK ~~~" marks where the book moves to a new scene. Work out from the new scene itself who narrates it (who "I" is, if anyone) and who is present. Report the narrator in "narrator" (their name, or "third person" if the passage is not told by "I"; for a passage with a scene break, the narrator of its last scene).
Also list every character who speaks in this passage with their gender, age group and any other names, titles or nicknames the passage uses for them (aliases). Only give an alias when the passage shows it is the same person; never another character's name.""".replace(
    "{delivery_rule}", DELIVERY_RULE).replace("{sound_kinds}", sound_marks.prompt_list())


RESPELL_SCHEMA = {"type": "array", "maxItems": 4, "items": {
    "type": "object", "properties": {"written": {"type": "string", "maxLength": 40},
                                     "said": {"type": "string", "maxLength": 40}},
    "required": ["written", "said"]}}
SOUNDS_SCHEMA = {"type": "array", "maxItems": 4, "items": {
    "type": "object", "properties": {"written": {"type": "string", "maxLength": 40},
                                     "sound": {"type": "string", "enum": list(sound_marks.KINDS)}},
    "required": ["written", "sound"]}}


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
                        # Approved tags only (delivery.py); [] = plainly spoken.
                        "delivery": {"type": "array", "maxItems": delivery_tags.MAX_TAGS,
                                     "items": {"type": "string", "enum": delivery_tags.all_tags()}},
                        # Words spelled the way they sound ("whaaaaaaat" -> "what"): [] for the others.
                        "respell": RESPELL_SCHEMA,
                        # Non-verbal sounds written in the quotation ("Ahaha" -> laugh): [] for the others.
                        "sounds": SOUNDS_SCHEMA,
                    },
                    "required": ["id", "starts", "speaker", "delivery", "respell", "sounds"],
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

    def save_book(self, book_id: str, payload: dict, added_by: str | None = None) -> None:
        """Saves an uploaded book. `added_by`: the user adding it, whose library it joins (the book's `users`;
        permissions.py); the same book added by someone else is in both libraries."""
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
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                keep = {k: v for k, v in old.items() if k in ("series", "users")
                        or (old.get("title_user") and k in ("title", "title_user"))}
            except ValueError:
                pass
        if added_by and added_by not in keep.get("users", []):
            keep["users"] = [*keep.get("users", []), added_by]
        if keep.get("title_user"):  # the reader renamed it: uploading the file again keeps their title
            payload["title"] = keep["title"]
            (d / "book.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(book_summary(payload) | {"uploaded": time.time()} | keep,
                                        ensure_ascii=False), encoding="utf-8")

    def add_user(self, book_id: str, user_id: str) -> None:
        """The book joins a user's library (meta.json `users`)."""
        meta_path = self.dir(book_id) / "meta.json"
        meta = self.summary(book_id) or {}
        if user_id and user_id not in meta.get("users", []):
            meta["users"] = [*meta.get("users", []), user_id]
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

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
            if line.get("conf") is not None:
                lines[qid]["conf"] = line["conf"]  # how sure the model was
                lines[qid]["alts"] = line.get("alts") or []  # [[character id or None (narrator), probability]]
            if line.get("dodds"):
                lines[qid]["dodds"] = line["dodds"]  # [[delivery tag, probability]]: the unsure ones left out
            if line.get("mention") and line.get("speaker") is None:
                lines[qid]["mention"] = True  # a quoted term inside narration, not dialogue
            if line.get("revised"):
                lines[qid]["revised"] = True  # changed by a later passage's evidence
            if line.get("say"):
                lines[qid]["say"] = line["say"]  # spoken as (the quotation spells a sound out)
        for qid, fix in script.get("corrections", {}).items():
            base = lines.get(qid, {"speaker": None, "delivery": None})
            mention = fix.get("mention", base.get("mention") and "speaker" not in fix)
            lines[qid] = {
                "speaker": None if mention else fix.get("speaker", base["speaker"]),
                "delivery": None if mention else fix.get("delivery", base["delivery"]),
                "corrected": True,
            }
            if mention:
                lines[qid]["mention"] = True  # read in the flow of its sentence, by the narration
            if base.get("say"):
                lines[qid]["say"] = base["say"]
            if qid in script.get("lines", {}):  # what the analysis said, for the line's dialog
                lines[qid]["analysis"] = {k: base[k] for k in ("speaker", "delivery", "conf", "alts", "dodds", "mention")
                                          if k in base}
        for name in ("guidance", "temperature"):  # the reader's own speech settings (the line dialog's sliders)
            for qid, v in (script.get(name) or {}).items():
                if qid in lines:
                    lines[qid][name] = v
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
            # The reader's corrections of it: {"chapter:paragraph": character id or None} (fix_narration).
            "narration_fixed": script.get("narration_fixed") or {},
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
    """Who narrates each section: as the analysis finds it (analysed_narration), with the reader's corrections
    (script["narration_fixed"]) over it."""
    return fix_narration(book, analysed_narration(book, script), script.get("narration_fixed") or {})


def section_end(book: dict, ci: int, pi: int) -> tuple[int, int] | None:
    """Where a narration section starting at (ci, pi) ends: the next point-of-view heading or scene break in its
    chapter, or the next chapter; None at the end of the book."""
    ch = book["chapters"][ci]
    paras = ch.get("paragraphs", [])
    breaks = set(ch.get("breaks") or [])
    for p in range(pi + 1, len(paras)):
        if p in breaks or is_pov_marker(paras[p]):
            return ci, p
    return (ci + 1, 0) if ci + 1 < len(book["chapters"]) else None


def fix_narration(book: dict, runs: list[list], fixed: dict) -> list[list]:
    """`runs` with the reader's corrections: {"chapter:paragraph": character id, or None for the book's narrator
    voice}, each holding from that paragraph to the end of its section (section_end), where the narration goes back
    to what it was."""
    if not fixed:
        return runs

    def who_at(pos):
        who = None
        for c, p, w in runs:
            if (c, p) > pos:
                break
            who = w
        return who

    events = {(c, p): w for c, p, w in runs}
    chapters = book.get("chapters", [])
    for key, who in sorted(fixed.items(), key=lambda kv: tuple(map(int, kv[0].split(":")))):
        ci, pi = map(int, key.split(":"))
        if ci >= len(chapters) or pi >= len(chapters[ci].get("paragraphs", [])):
            continue  # the book's text changed
        end = section_end(book, ci, pi)
        if end is not None and end not in events:
            events[end] = who_at(end)
        for pos in [pos for pos in events if (ci, pi) < pos < (end or (len(chapters), 0))]:
            del events[pos]
        events[(ci, pi)] = who
    out: list[list] = []
    for (c, p), w in sorted(events.items()):
        if (out[-1][2] if out else None) != w:
            out.append([c, p, w])
    return out


def analysed_narration(book: dict, script: dict) -> list[list]:
    """Who narrates each section, for first-person books.

    Evidence is deterministic: the character that quotations tagged "…," I said
    are attributed to. Sections are split at chapters, point-of-view headings and
    scene breaks (the EPUB's "breaks"). A section without evidence is left to the
    book's narrator voice when its narration is in the third person (hardly an "I"
    outside the quotations) or it starts with a point-of-view change; a scene after
    a break goes on with the narrator of the scene before (a flashback, a time skip);
    otherwise it keeps the book's main narrator.
    Third-person books (little or no evidence) get an empty list. Books with two
    first-person narrators in different tenses are split by narration tense.
    """
    lines = ScriptStore.public(script)["lines"]
    narrator = narrators_by_tense(book, lines)
    if narrator is not None:
        return tense_narration(book, narrator)
    # (chapter, first paragraph, votes, how it starts: "chapter" / "pov" / "break", narration words, first-person words
    # (with "we"), first-person singular words)
    sections: list[list] = []
    for ci, ch in enumerate(book.get("chapters", [])):
        paras = ch.get("paragraphs", [])
        breaks = set(ch.get("breaks") or [])
        by_para: dict[int, list] = {}
        for qid, p, s, e in ch.get("quotes", []):
            by_para.setdefault(int(p), []).append((qid, int(s), int(e)))
        cur = [ci, 0, {}, "pov" if is_pov_marker(ch.get("title", "")) else "chapter", 0, 0, 0]
        for pi, text in enumerate(paras):
            if pi > 0 and (is_pov_marker(text) or pi in breaks):
                sections.append(cur)
                cur = [ci, pi, {}, "pov" if is_pov_marker(text) else "break", 0, 0, 0]
            votes, last = cur[2], 0
            narration = []
            for qid, s, e in by_para.get(pi, []):
                narration.append(text[last:s])
                last = e
                speaker = (lines.get(qid) or {}).get("speaker")
                if speaker and (_I_SAID_AFTER.match(text[e:e + 60]) or _I_SAID_BEFORE.search(text[max(0, s - 60):s])):
                    votes[speaker] = votes.get(speaker, 0) + 1
            narration.append(text[last:])
            said = " ".join(narration)
            cur[4] += len(said.split())
            cur[5] += len(_FIRST_PERSON_WORDS.findall(said))
            cur[6] += len(_FIRST_PERSON_SINGULAR.findall(said))
        sections.append(cur)
    totals: dict[str, int] = {}
    for sec in sections:
        for k, v in sec[2].items():
            totals[k] = totals.get(k, 0) + v
    if sum(totals.values()) < 3:
        return []
    main = max(totals, key=totals.get)
    out: list[list] = []
    previous = main
    for ci, start, votes, kind, words, first, singular in sections:
        if votes:
            who = max(votes, key=votes.get)
        elif words >= THIRD_PERSON_MIN_WORDS and (not singular or first * 100 < THIRD_PERSON_MAX_PER_100 * words):
            who = None  # told in the third person: the book's narrator voice
        elif kind == "pov":
            who = None
        elif kind == "break" and (words < THIRD_PERSON_MIN_WORDS or previous is not None):
            who = previous  # a new scene told by the same "I" (or too short to tell): the narrator goes on
        else:
            who = main
        previous = who
        if not out or out[-1][2] != who:
            out.append([ci, start, who])
    return out


# A section's narration is in the third person when "I", "me", "we", "our"... make up less than this many of every 100
# of its words outside the quotations (first-person narration has several, though "we" and "us" can carry a scene: 0.89
# in Unital Ring I's ch. 8 opening; a third-person scene only a stray thought: at most 0.74 across the library),
# judged from this many words.
_FIRST_PERSON_WORDS = re.compile(r"\b(?:I|me|my|mine|myself|we|us|our|ours|ourselves|I'm|I've|I'd|I'll|we're|we've|"
                                 r"we'd|we'll|I\u2019m|I\u2019ve|I\u2019d|I\u2019ll|we\u2019re|we\u2019ve|we\u2019d|"
                                 r"we\u2019ll)\b")
# ...and with no "I" at all it is third person whatever its "we" ("…our city, the elder said": reported speech).
_FIRST_PERSON_SINGULAR = re.compile(r"\b(?:I|me|my|mine|myself|I'm|I've|I'd|I'll|I\u2019m|I\u2019ve|I\u2019d|I\u2019ll)\b")
THIRD_PERSON_MAX_PER_100 = 0.8
THIRD_PERSON_MIN_WORDS = 120


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


def detach_alias(script: dict, name: str) -> dict[str, list[str]]:
    """The reader names someone new with a name that is only another character's alias (the analysis filed
    "Fanatio" under Asuna): that alias and longer forms of it ("Fanatio Synthesis Two") leave the other
    character, so the name can become a character of its own. Nothing changes when it is a character's own
    name. Returns {character id: aliases removed}."""
    key = norm_name(name)
    if not key or any(norm_name(c["name"]) == key for c in script["characters"].values()):
        return {}
    removed: dict[str, list[str]] = {}
    for cid, c in script["characters"].items():
        gone = [a for a in c.get("aliases", []) if norm_name(a) == key or norm_name(a).startswith(key + " ")]
        if gone:
            c["aliases"] = [a for a in c["aliases"] if a not in gone]
            removed[cid] = gone
    return removed


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
                   and a.lower() not in PRONOUNS and find_character(script, a) == cid and count(a) >= 3]
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
                     aliases: list[str] | tuple = (), exact: bool = False, vote: str | None = None) -> str | None:
    """The one place characters are created. Names from the model go through normalize_character_name
    ("a prisoner" -> "Prisoner"); `exact` keeps a name the reader typed as it is.

    A new name only joins an existing character when one of its aliases is that character's own name
    ("Kazuto Kirigaya", also called "Kirito"), never through another alias: a title two people share
    ("Commander") once folded Fanatio into Asuna. `vote` (the passage reporting it) makes an alias unrelated to
    the character's name ("Commander", "Urano") wait until a second passage reports it too; None (the series
    memory, the reader) takes aliases at once."""
    if not name or name.strip().upper() == NARRATOR:
        return None
    cid = find_character(script, name)
    if cid is None:
        for alias in aliases:
            cid = _named(script, alias) if is_proper_alias(alias) else None
            if cid:
                break
    joined = None  # the pending alias this name joined through
    if cid is None and not exact:
        # A name one passage already gave as someone's alias, now speaking on its own ("Sasaki", also called
        # "Kenta Nakano"; later just "Kenta Nakano"): it is them, and a new passage counts as the second report.
        pending = _pending_owner(script, name)
        if pending:
            cid, joined = pending
            seen = script["characters"][cid]["alias_votes"][joined]
            if vote is None or vote not in seen:
                seen.append(vote or f"label-{time.time_ns()}")
            if len(seen) >= ALIAS_VOTES:
                script["characters"][cid].setdefault("aliases", []).append(joined)
                del script["characters"][cid]["alias_votes"][joined]
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
        if not is_proper_alias(alias) or norm_name(alias) == norm_name(c["name"]) or alias == joined:
            continue
        if norm_name(alias) in (norm_name(a) for a in c.setdefault("aliases", [])):
            continue
        owner = find_character(script, alias)
        if owner is not None:  # never steal another character's name
            continue
        if vote is None or alias == name or related_names(alias, c["name"]):
            c["aliases"].append(alias)
            c.get("alias_votes", {}).pop(alias, None)
            continue
        seen = c.setdefault("alias_votes", {}).setdefault(alias, [])
        if vote not in seen:
            seen.append(vote)
        if len(seen) >= ALIAS_VOTES:
            c["aliases"].append(alias)
            del c["alias_votes"][alias]
    if c.get("gender", "unknown") == "unknown" and gender in GENDERS:
        c["gender"] = gender
    if c.get("age", "unknown") == "unknown" and age in AGES:
        c["age"] = age
    return cid


ALIAS_VOTES = 2  # passages that must report an alias unrelated to a character's name before it is taken
PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
            "we", "us", "our", "they", "them", "their", "it", "its"}


def _named(script: dict, name: str) -> str | None:
    """The character whose own name (not an alias) this is."""
    key = norm_name(name)
    return next((cid for cid, c in script["characters"].items() if norm_name(c["name"]) == key), None)


def _pending_owner(script: dict, name: str) -> tuple[str, str] | None:
    """(character id, alias) when exactly one character has `name` as an alias still waiting for its second
    report."""
    key = norm_name(name)
    hits = [(cid, a) for cid, c in script["characters"].items() for a in (c.get("alias_votes") or {})
            if norm_name(a) == key]
    return hits[0] if len({cid for cid, _ in hits}) == 1 else None


def _name_words(name: str) -> set[str]:
    words = re.findall(r"[^\W\d_][\w'\u2019-]*", name.lower())
    return {w for w in words if w not in _DESCRIPTOR_WORDS and w not in _SMALL_WORDS and len(w) > 1}


def related_names(a: str, b: str) -> bool:
    """Whether two names share a word that isn't a title ("Lady Asuna" and "Asuna Yuuki"; not "Lady Fizel"
    and "Lady Sheyta")."""
    return bool(_name_words(a) & _name_words(b))


def is_proper_alias(alias: str) -> bool:
    """Nicknames and names ("Makeup", "Robot Girl", "Zhang Lee"), not descriptions ("the man")."""
    alias = alias.strip()
    return (0 < len(alias) <= 40 and any(ch.isupper() for ch in alias) and alias.upper() != NARRATOR
            and alias.lower() not in PRONOUNS)


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
    for cid, c in ranked:
        extra = ", ".join(x for x in (c.get("gender"), c.get("age")) if x and x != "unknown")
        # An alias another character also answers to would steer that character's lines here.
        aliases = [a for a in c.get("aliases", []) if find_character(script, a) == cid and a.lower() not in PRONOUNS]
        alias = f"; also called {', '.join(aliases)}" if aliases else ""
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


def apply_chunk_result(script: dict, result: dict, local: dict[str, list[str]], vote: str | None = None) -> int:
    """Folds one passage's labels into the script. `vote` names the passage (its chunk key): a new alias
    unrelated to a character's name needs reports from two passages (ensure_character)."""
    vote = vote or f"call-{time.time_ns()}"
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
                             [a for a in sp.get("aliases") or [] if isinstance(a, str)], vote=vote)
    # Aliases this passage reported but not yet confirmed by a second one still resolve its own labels ("Nick",
    # also called "Zhang Lee", then "man (Zhang Lee)" says a line); they are taken back afterwards.
    passage_only = []
    for cid, c in script["characters"].items():
        for alias, seen in (c.get("alias_votes") or {}).items():
            if vote in seen and find_character(script, alias) is None:
                c.setdefault("aliases", []).append(alias)
                passage_only.append((c, alias))
    try:
        n = _apply_quotes(script, result, local, narrator)
    finally:
        for c, alias in passage_only:
            if alias in (c.get("alias_votes") or {}) and alias in c.get("aliases", []):
                c["aliases"].remove(alias)
    if result.get("revisions"):
        apply_revisions(script, result)
    if result.get("profile_updates"):
        from .profile import apply_updates

        apply_updates(script, result["profile_updates"])
    return n


_QUOTE_MARKS = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\u300c\u300d\u300e\u300f "
# What the model names must be one word ("whaaaaaaat", "so-o-o"): given "Wh…whaaaaaaat?!" it said "What?",
# losing the stutter and the "!".
_ONE_WORD = re.compile(r"[^\W\d_]+(?:[-'\u2019][^\W\d_]+)*")


def _destutters(written: str, said: str) -> bool:
    """"Y-yeah" -> "Yeah": a stutter the model flattened although told to keep it (they read fine)."""
    parts = written.lower().split("-")
    return len(parts) > 1 and said.lower() == parts[-1] and all(parts[-1].startswith(p) for p in parts[:-1])


def respelled(text: str | None, pairs) -> str:
    """How a quotation is read aloud ("say") when the model found words in it spelled the way they sound
    ([{"written": "whaaaaaaat", "said": "what"}]): the quotation with just those words swapped, or "" when there
    are none. The model names words rather than rewriting the quotation: asked for the whole quotation, it
    "tidied" ordinary lines (stutters, ellipses) and dropped the end of a long one. A word not in the
    quotation is ignored."""
    if not text or not isinstance(pairs, list):
        return ""
    out = text
    for p in pairs:
        if not isinstance(p, dict):
            continue
        written = str(p.get("written") or "").strip(_QUOTE_MARKS)
        said = " ".join(str(p.get("said") or "").split()).strip(_QUOTE_MARKS)
        if not _ONE_WORD.fullmatch(written) or not any(ch.isalpha() for ch in said) or \
                said.lower() == written.lower() or _destutters(written, said) or len(said) > len(written):
            # (Plain words are never longer than drawn-out ones: "plea—" cut off mid-word became "please".)
            continue
        m = re.search(r"(?<!\w)" + re.escape(written) + r"(?!\w)", out, re.IGNORECASE)
        if not m:
            continue
        if m.group(0)[0].isupper() and said[0].islower():
            said = said[0].upper() + said[1:]
        out = out[:m.start()] + said + out[m.end():]
    return out if out != text else ""


def spoken_form(text: str | None, respell=None, sounds=None, said: str | None = None) -> str:
    """A quotation's spoken form ("say"): its respelled words swapped (respelled) and the sounds written in it
    marked ({laugh|Ahaha}, sounds.py), from `said` (an earlier spoken form) or the text; "" when that changes
    nothing."""
    if not text:
        return ""
    base = respelled(text, respell) or said or text
    out = sound_marks.mark(base, sounds)
    return out if out != text else ""


def sure_delivery(tags: list, odds=None) -> tuple[str | None, list | None]:
    """(delivery, odds) from the model's tags and how likely it was to give each (delivery_odds): only tags it was
    at least delivery.MIN_CONF sure of are kept; `odds` ([[tag, probability], ...], every approved tag it gave;
    [] for none) shows what was left out. Without probabilities (a reasoning model) every tag is kept, and odds
    is None."""
    if odds is None:
        return delivery_tags.normalize(tags), None
    probs = {" ".join(str(t).lower().split()): p for t, p in odds}
    keep = [t for t in tags if probs.get(" ".join(str(t).lower().split()), 1.0) >= delivery_tags.MIN_CONF]
    listed = [[" ".join(str(t).lower().split()), round(p, 3)] for t, p in odds if delivery_tags.is_approved(str(t))]
    return delivery_tags.normalize(keep), listed


def _apply_quotes(script: dict, result: dict, local: dict, narrator: str) -> int:
    n = 0
    for q in result.get("quotes", []):
        qids = local.get(str(q.get("id")))
        if not qids:
            continue
        if isinstance(qids, str):
            qids = [qids]
        raw = q.get("delivery")
        dodds = None
        if isinstance(raw, list):  # approved tags (the labelling schema)
            delivery, dodds = sure_delivery(raw, q.get("_dodds"))
        else:  # a free-text delivery (a cached answer from before the approved list)
            delivery = delivery_tags.from_text(raw)
        # "the Hatter and the March Hare" in chorus: voiced by the first.
        speaker = resolve_speaker(script, q.get("speaker", ""), narrator)
        odds = line_odds(script, q, speaker, narrator)
        # A quotation split by narration ("Yes," [3] he said, [3] "I will.") has one label for two pieces: a
        # spoken form can't be shared out between them.
        say = spoken_form(q.get("_text"), q.get("respell"), q.get("sounds")) if len(qids) == 1 else ""
        for qid in qids:
            script["lines"][qid] = {"speaker": speaker, "delivery": delivery, **odds,
                                    **({"dodds": dodds} if dodds is not None else {})}
            if say:
                script["lines"][qid]["say"] = say
            if q.get("mention"):
                script["lines"][qid]["mention"] = True
            n += 1
    return n


def split_group(name: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*(?:,|&|\band\b)\s*", name or "") if p.strip()]
    return parts or [name or ""]


MIN_ALTERNATIVE = 0.03  # alternatives less likely than this are not kept


def guess_speaker(script: dict, start: str, narrator: str = ""):
    """The speaker a name starting with `start` (an alternative first token) would be: a character id, None for
    narration, or False when no known character or several start that way."""
    s = start.strip()
    if not s:
        return False
    found: set = set()
    if NARRATOR.startswith(s) and s.isupper():
        found.add(None)
    if s == "I" and narrator and narrator.lower() not in ("third person", "unknown", "none"):
        found.add(find_character(script, narrator))
    for cid, c in script["characters"].items():
        if any(n.startswith(s) for n in [c["name"], *c.get("aliases", [])]):
            found.add(cid)
    return found.pop() if len(found) == 1 else False


def line_odds(script: dict, q: dict, speaker: str | None, narrator: str = "") -> dict:
    """{"conf": how sure the model was of its speaker, "alts": [[other speaker, probability], ...]} from the
    probabilities label_chunk read off its tokens; {} without them."""
    if q.get("_p") is None:
        return {}
    others: dict = {}
    for start, p in q.get("_alts") or []:
        who = guess_speaker(script, start, narrator)
        if who is not False:
            others[who] = others.get(who, 0.0) + p
    conf = min(1.0, q["_p"] + others.pop(speaker, 0.0))  # another spelling of the same speaker counts for it
    alts = [[who, round(p, 3)] for who, p in sorted(others.items(), key=lambda kv: -kv[1]) if p >= MIN_ALTERNATIVE]
    return {"conf": round(conf, 3), "alts": alts[:3]}


# ----------------------------------------------------------------- LLM


class LlmError(Exception):
    pass


class LlmPreempted(Exception):
    """The LLM was stopped because someone started using the site (llm_mode "adaptive"); the step is retried later."""


SHARED_BASE_VRAM_GB = 4.0  # the speech model llm_cpu_moe_shared was tuned next to (OmniVoice)
MOE_LAYER_GB = 0.5  # one layer's experts of the default LLM (Qwen3.5-35B-A3B Q4_K_M: 20.5 GB, 40 layers)


def speech_model_extra_layers(cfg) -> int:
    """More layers' experts kept in RAM in shared mode when the speech models need more VRAM than OmniVoice alone
    (VoxCPM2 ~7 GB: 6 more; with OmniVoice reading the narration next to it, 11 GB: 14 more), so they all still
    fit on the GPU together."""
    from .engines.registry import ENGINES

    spec = ENGINES.get(cfg.default_engine)
    speech_gb = spec.vram_gb if spec else SHARED_BASE_VRAM_GB
    narration = ENGINES.get(cfg.narration_engine) if cfg.narration_engine != cfg.default_engine else None
    extra = speech_gb + (narration.vram_gb if narration else 0.0) - SHARED_BASE_VRAM_GB
    return max(0, math.ceil(extra / MOE_LAYER_GB - 1e-9))


LLM_SHARED_MIN_GB = 3.0  # the LLM next to the speech models with every expert in RAM: attention, cache, buffers
DESKTOP_VRAM_GB = 1.0  # Windows and the desktop
_GPU_TOTAL_GB: list = []


def gpu_total_gb() -> float | None:
    """The GPU's memory (nvidia-smi), once; None without an NVIDIA GPU."""
    if not _GPU_TOTAL_GB:
        import shutil
        import subprocess

        total = None
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                out = subprocess.run([smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.split()
                total = float(out[0]) / 1024 if out else None
            except (OSError, ValueError, subprocess.SubprocessError):
                total = None
        _GPU_TOTAL_GB.append(total)
    return _GPU_TOTAL_GB[0]


def speech_models_gb(cfg) -> float:
    from .engines.registry import ENGINES

    spec = ENGINES.get(cfg.default_engine)
    narration = ENGINES.get(cfg.narration_engine) if cfg.narration_engine != cfg.default_engine else None
    return (spec.vram_gb if spec else SHARED_BASE_VRAM_GB) + (narration.vram_gb if narration else 0.0)


def shared_fits(cfg) -> bool:
    """Whether the analysis LLM fits on the GPU next to the speech models. Breeze TTS 2 (9 GB) with OmniVoice
    narrating (4 GB) took 14 GB with Windows, measured: the LLM's 3 GB more would spill into system memory, which
    Windows does without an error and which slows the speech and the LLM alike. Then analysis waits while someone
    reads, as with two readers. VoxCPM2 (7 GB) with OmniVoice: 14.1 GB with the LLM, it fits."""
    total = gpu_total_gb()
    return total is None or speech_models_gb(cfg) + LLM_SHARED_MIN_GB + DESKTOP_VRAM_GB <= total


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
        # VRAM to leave free on "the whole GPU" (a speech model parked in system RAM still holds some).
        self.reserve_gb = 0.0
        # The inference server, when Settings choose it for analysis (llm_backend "remote").
        self.remote = RemoteLlm(cfg)

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
        cpu_moe = self.cfg.llm_cpu_moe_shared + speech_model_extra_layers(self.cfg) if shared else self.cfg.llm_cpu_moe
        if self.gpu and not shared and self.reserve_gb:
            # The whole GPU but for what the parked speech model still holds: that many more layers in RAM.
            cpu_moe += math.ceil(self.reserve_gb / MOE_LAYER_GB - 1e-9)
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
        if self.cfg.llm_backend == "remote":
            busy = self.remote.busy_since is not None
            return {"state": "running" if busy else "stopped", "where": "remote" if busy else None}
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
                       model: str | None = None, thinking: bool = False, effort: str = "high",
                       logprobs: int = 0) -> dict:
        """One chat completion constrained to `schema`. With `thinking`, reasoning models reason
        first (llama.cpp returns the reasoning separately and applies the schema to the answer). `logprobs`
        (the top alternatives per token) adds the answer's tokens with their probabilities as "_tokens"."""
        if self.cfg.llm_backend == "remote":  # Settings: analysis runs on the inference server
            return await self.remote.complete(session, system, user, schema, max_tokens=max_tokens, thinking=thinking,
                                              effort=effort, logprobs=logprobs)
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
        if logprobs:
            # The model's own distribution (before sampling), so a 60/30 split between two speakers shows.
            body.update(logprobs=True, top_logprobs=logprobs)
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
        if logprobs:
            parsed["_tokens"] = ((data["choices"][0].get("logprobs") or {}).get("content")) or []
        return parsed


REMOTE_BUSY_WAITS = 120  # tries while the analysis server is at its request limit (about an hour in all)
REMOTE_BUSY_DELAY_S = 2.0  # the first wait; each one longer, up to 30 s


class RemoteLlm:
    """Analysis on an OpenAI-compatible server elsewhere (llm_remote_url, model llm_remote_model): llama-swap on the
    inference PC, which loads the model with the first request (a few minutes) and serves several at once.

    The same requests as the local llama-server (json_schema, logprobs, thinking on or off), except: vLLM's grammar
    and its speculative decoding don't go together (every schema-constrained request came back HTTP 500), so after
    one such answer the schema is written into the prompt instead, and the JSON taken from the answer. When the
    server is at its request limit (HTTP 429, shared with everyone using it) the request waits and tries again."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.schema_in_prompt = False
        self.busy_since: float | None = None  # a request in flight (the banner)
        self.inflight = 0

    @property
    def url(self) -> str:
        return self.cfg.llm_remote_url.rstrip("/")

    async def complete(self, session, system: str, user: str, schema: dict, max_tokens: int = 2048,
                       thinking: bool = False, effort: str = "high", logprobs: int = 0) -> dict:
        if not self.url or not self.cfg.llm_remote_model:
            raise LlmError("No analysis server is set up (llm_remote_url and llm_remote_model)")
        self.inflight += 1
        self.busy_since = self.busy_since or time.monotonic()
        try:
            busy_waits = 0
            while True:
                in_prompt = self.schema_in_prompt
                body = {
                    "model": self.cfg.llm_remote_model,
                    "messages": [{"role": "system", "content": system + (
                        "\n\nReply with only one JSON object that follows this JSON Schema exactly (no prose, no code "
                        "fence; copy every enum value exactly):\n" + json.dumps(schema, ensure_ascii=False)
                        if in_prompt else "")},
                                 {"role": "user", "content": user}],
                    "temperature": 0.6 if thinking else 0.1,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": ({"enable_thinking": True, "reasoning_effort": effort} if thinking
                                             else {"enable_thinking": False}),
                }
                if not in_prompt:
                    body["response_format"] = {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}}
                if logprobs:
                    body.update(logprobs=True, top_logprobs=logprobs)
                try:
                    # Long: the first request waits while the server loads the model.
                    async with session.post(f"{self.url}/chat/completions", json=body, timeout=1800) as r:
                        data = await r.json(content_type=None)
                        status = r.status
                except asyncio.TimeoutError as exc:
                    raise LlmError("The analysis server timed out") from exc
                except (OSError, ClientConnectionError) as exc:
                    raise LlmError(f"The analysis server can't be reached ({self.url}): {exc}") from exc
                if status == 429 and busy_waits < REMOTE_BUSY_WAITS:
                    # Its request limit is shared with everyone else using the server: wait for a free slot.
                    busy_waits += 1
                    await asyncio.sleep(min(30.0, REMOTE_BUSY_DELAY_S * busy_waits))
                    continue
                if status >= 500 and not in_prompt:
                    logger.warning("the analysis server refused a JSON-schema request (HTTP %s); the schema goes in "
                                   "the prompt from now on", status)
                    self.schema_in_prompt = True
                    continue
                if status >= 400 or not isinstance(data, dict) or "choices" not in data:
                    raise LlmError(f"Analysis server HTTP {status}: {str(data)[:300]}")
                break
            choice = data["choices"][0]
            content = re.sub(r"^\s*```(?:json)?|```\s*$", "", choice["message"].get("content") or "").strip()
            try:
                parsed, _ = json.JSONDecoder().raw_decode(content[content.find("{"):])
            except ValueError as exc:
                raise LlmError(f"LLM returned invalid JSON: {content[:200]}") from exc
            if not isinstance(parsed, dict):
                raise LlmError(f"LLM returned invalid JSON: {content[:200]}")
            parsed["_usage"] = data.get("usage", {})
            parsed["_timings"] = data.get("timings", {})
            parsed["_model"] = self.cfg.llm_remote_model
            if logprobs:
                parsed["_tokens"] = (choice.get("logprobs") or {}).get("content") or []
            return parsed
        finally:
            self.inflight -= 1
            if not self.inflight:
                self.busy_since = None


def preceding_text(book: dict, chunk: Chunk, words: int) -> str:
    """Up to `words` words of the chapter just before this chunk (unnumbered, for context)."""
    paras = book["chapters"][chunk.chapter]["paragraphs"]
    picked, n, k = [], 0, chunk.paragraphs[0] - 1
    while k >= 0 and n < words:
        picked.insert(0, k)
        n += len(paras[k].split())
        k -= 1
    text = with_breaks(book, chunk.chapter, picked)
    if chunk.paragraphs[0] in scene_breaks(book, chunk.chapter):
        text += "\n\n" + SCENE_BREAK  # (the passage itself starts with it too)
    return text


def following_text(book: dict, chunk: Chunk, words: int) -> str:
    """Up to `words` words of the chapter just after this chunk (unnumbered, for context): a speech tag or a
    reply in the next paragraph often shows who said the passage's last lines."""
    paras = book["chapters"][chunk.chapter]["paragraphs"]
    picked, n, k = [], 0, chunk.paragraphs[-1] + 1
    while k < len(paras) and n < words:
        picked.append(k)
        n += len(paras[k].split())
        k += 1
    return with_breaks(book, chunk.chapter, picked)


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
        script["lines"][qid] = {**{k: v for k, v in line.items() if k not in ("conf", "alts")},
                                "speaker": new, "revised": True}
        script.setdefault("revisions", []).append(event)
        done.append(event)
    return done


async def label_chunk(llm: LocalLlm, session, book: dict, script: dict, chunk: Chunk,
                      thinking: bool = False, model: str | None = None, effort: str = "high",
                      context_words: int = 0, recent: int = 6, after_words: int = 0,
                      proposed: dict[str, str] | None = None, revise: int = 0,
                      revise_unnamed: bool = False, profiles: bool = False) -> tuple[dict, dict[str, str]]:
    """Label one chunk. `context_words` adds the preceding text, `after_words` the text after the passage,
    `recent` sets how many earlier attributions are shown, and `proposed` (number -> speaker) asks the model to
    review a first pass."""
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
    elif chunk.paragraphs[0] in scene_breaks(book, chunk.chapter):
        narrator = f"{narrator}. This passage starts a new scene (after a scene break): work out who narrates it."
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
    after = following_text(book, chunk, after_words) if after_words else ""
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
        + (f"Story just after this passage (for context only: its quotations are not numbered and are "
           f"labelled later; use it for speech tags and replies that show who said the passage's lines):\n"
           f"{after}\n\n" if after else "")
        + f"{review}{revise_block}"
        f"Label every number from 1 to {len(local)} once."
    )
    # Generous budget: reasoning models (gpt-oss) spend tokens before the JSON, and the
    # schema grammar ends the output as soon as the JSON is complete anyway.
    budget = 1024 + 70 * len(local)
    extra = {"thinking": True, "effort": effort} if thinking else {"logprobs": TOP_ALTERNATIVES}
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
    tokens = result.pop("_tokens", None) or []
    for q, odds in zip(result.get("quotes", []), speaker_odds(tokens)):
        q["_p"], q["_alts"] = odds
    for q, odds in zip(result.get("quotes", []), delivery_odds(tokens)):
        q["_dodds"] = odds
    # Backstop for the prompt rule: a quoted term inside a sentence is narration.
    paras = book["chapters"][chunk.chapter]["paragraphs"]
    spans = {q[0]: q for q in chunk.quotes}
    for q in result.get("quotes", []):
        ids = local.get(str(q.get("id"))) or []
        if len(ids) == 1 and ids[0] in spans:  # the words to respell are swapped in this (respelled)
            s = spans[ids[0]]
            q["_text"] = paras[s[1]][s[2]:s[3]]
        if ids and all(is_mention(paras[spans[i][1]], spans[i][2], spans[i][3]) for i in ids):
            q.update(speaker=NARRATOR, delivery=[], mention=True)
            q.pop("_p", None)
            q.pop("_alts", None)
            q.pop("_dodds", None)
    return result, local


TOP_ALTERNATIVES = 8  # tokens per position the LLM reports with their probabilities
_SPEAKER_FIELD = re.compile(r'"speaker"\s*:\s*"((?:[^"\\]|\\.)*)"')
_DELIVERY_FIELD = re.compile(r'"delivery"\s*:\s*(\[)([^\]]*)\]')
_DELIVERY_TAG = re.compile(r'"((?:[^"\\]|\\.)*)"')


def delivery_odds(tokens: list[dict]) -> list[list[tuple[str, float]]]:
    """From the answer's tokens with their probabilities, for each "delivery" field in order: each tag the model
    gave with how likely it was to give it, from the choice to add a tag at all (another tag rather than closing
    the list) through the tag's last token."""
    if not tokens:
        return []
    text, starts = "", []
    for t in tokens:
        starts.append(len(text))
        text += t.get("token", "")
    out = []
    for m in _DELIVERY_FIELD.finditer(text):
        tags, decided = [], m.start(1)  # where the choice to add this tag is made: "[" or the previous tag's end
        for t in _DELIVERY_TAG.finditer(m.group(2)):
            a, b = m.start(2) + t.start(1), m.start(2) + t.end(1)
            inside = [i for i, s in enumerate(starts) if s < b and s + len(tokens[i].get("token", "")) > decided]
            logp = sum(tokens[i].get("logprob", 0.0) for i in inside)
            tags.append((t.group(1), math.exp(logp) if inside else 0.0))
            decided = b
        out.append(tags)
    return out


def speaker_odds(tokens: list[dict]) -> list[tuple[float, list[tuple[str, float]]]]:
    """From the answer's tokens with their probabilities, for each "speaker" field in order: how likely the
    model's speaker was (the product over its tokens) and the other ways the name could have started, with
    their probabilities ("Fan" 0.28 when it wrote "Asuna")."""
    if not tokens:
        return []
    text, starts = "", []
    for t in tokens:
        starts.append(len(text))
        text += t.get("token", "")
    out = []
    for m in _SPEAKER_FIELD.finditer(text):
        a, b = m.span(1)
        if a == b:
            out.append((0.0, []))
            continue
        inside = [i for i, s in enumerate(starts) if s < b and s + len(tokens[i].get("token", "")) > a]
        if not inside:
            out.append((0.0, []))
            continue
        logp = sum(tokens[i].get("logprob", 0.0) for i in inside)
        first = tokens[inside[0]]
        cut = a - starts[inside[0]]  # the first token may carry the opening quote
        prefix = first.get("token", "")[:cut]
        alts = []
        for alt in first.get("top_logprobs") or []:
            tok = alt.get("token", "")
            if tok == first.get("token") or not tok.startswith(prefix):
                continue
            rest = tok[cut:].lstrip()
            if rest and not rest.startswith('"'):
                alts.append((rest, math.exp(alt.get("logprob", -99.0))))
        out.append((math.exp(logp), alts))
    return out


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


SPOKEN_FORMS = 2  # script["spoken_forms"]: every line of the book has been checked for words to respell and
# written sounds to mark ("say"); 2 added the sounds
SPOKEN_PROMPT = """You prepare a novel's dialogue for a speech engine, which reads each word's letters exactly as written. Below are numbered quotations. Find two things; most quotations have neither, so empty lists are the usual answer.
1. "spoken": words spelled the way they sound, so the engine would garble them: drawn-out letters ("whaaaaaaat", "Nooooo", "pleeease", "so-o-o"). For each, give the quotation's number, that one word exactly as written (just the word: no punctuation, nothing around it) and the plain word ({"id": "3", "written": "whaaaaaaat", "said": "what"}). Stutters ("Y-yeah", "Wh… what"), ellipses, and sounds that are words of their own (Hmm, Shh, Zzz, Grr, Kyaa, growls, animal noises) stay as they are.
2. "sounds": non-verbal sounds the quotation writes out (laughter, a sigh, a scream, a hesitation, a groan: "Ahaha, that's fine.", "Haaah... Really.", "Kyaaah!"). For each, give the quotation's number, the word as written (no punctuation) and its kind ({"id": "5", "written": "Ahaha", "sound": "laugh"}). Ordinary words are not sounds ("Wow", "Yes", "Oh no", names); "Uh-huh" and "Mm-hmm" meaning yes are words. Kinds:
""" + sound_marks.prompt_list()
SPOKEN_BATCH_CHARS = 8000


def spoken_batches(book: dict, script: dict) -> list[list[tuple[str, str]]]:
    """The book's quotations (id, text) in batches for find_spoken_forms; quoted terms inside narration are
    left out."""
    lines = script.get("lines", {})
    batches, cur, size = [], [], 0
    for ch in book.get("chapters", []):
        paras = ch.get("paragraphs", [])
        for qid, pi, s, e in ch.get("quotes", []):
            line = lines.get(qid)
            if line is None or line.get("mention"):
                continue
            text = paras[pi][s:e]
            if cur and size + len(text) > SPOKEN_BATCH_CHARS:
                batches.append(cur)
                cur, size = [], 0
            cur.append((qid, text))
            size += len(text) + 8
    if cur:
        batches.append(cur)
    return batches


async def find_spoken_forms(llm: LocalLlm, session, batch: list[tuple[str, str]],
                            said: dict[str, str] | None = None) -> dict[str, str]:
    """For a batch of quotations: how those with words spelled the way they sound or sounds written out are read
    ({id: say}), building on their earlier spoken forms (`said`)."""
    ids = [str(i) for i in range(1, len(batch) + 1)]
    item = lambda props: {"type": "array", "maxItems": 2 * len(batch), "items": {  # noqa: E731
        "type": "object", "properties": {"id": {"type": "string", "enum": ids}, **props},
        "required": ["id", *props]}}
    schema = {"type": "object", "properties": {"spoken": item(RESPELL_SCHEMA["items"]["properties"]),
                                               "sounds": item(SOUNDS_SCHEMA["items"]["properties"])},
              "required": ["spoken", "sounds"]}
    user = "\n".join(f"[{n}] {text}" for n, (_, text) in zip(ids, batch))
    result = await llm.complete(session, SPOKEN_PROMPT, user, schema, max_tokens=2500)
    found: dict[int, dict[str, list]] = {}
    for key in ("spoken", "sounds"):
        for it in result.get(key) or []:
            try:
                n = int(it.get("id")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= n < len(batch):
                found.setdefault(n, {"spoken": [], "sounds": []})[key].append(it)
    out = {}
    for n, items in found.items():
        qid, text = batch[n]
        if say := spoken_form(text, items["spoken"], items["sounds"], (said or {}).get(qid)):
            out[qid] = say
    return out


DELIVERY_SCORED = 1  # script["delivery_scored"]: every line's delivery came with the model's confidence ("dodds")
DELIVERY_PROMPT = """You direct the voice actors of a multi-voice audiobook. Below is a passage of a novel; each quotation is numbered, and you are told who says it. For every number, decide:
""" + DELIVERY_RULE + """
A quotation nobody says aloud (a quoted term, a sign, a written note) gets []."""


def needs_scoring(script: dict, chunk: Chunk) -> bool:
    """Whether some line of the passage has a delivery without the model's confidence."""
    lines = script.get("lines", {})
    return any(qid in lines and not lines[qid].get("mention") and "dodds" not in lines[qid]
               for qid, *_ in chunk.quotes)


async def score_deliveries(llm: LocalLlm, session, book: dict, script: dict, chunk: Chunk,
                           context_words: int = 0) -> dict[str, tuple[str | None, list]]:
    """The deliveries of a passage's lines again, with how sure the model is of each tag ({qid: (delivery,
    dodds)}, sure_delivery): for lines labelled before the confidence was read. The speakers are given, not
    asked for."""
    text, local = render_chunk(book, chunk)
    lines, chars = script.get("lines", {}), script.get("characters", {})

    def who(qids: list[str]) -> str:
        line = lines.get(qids[0]) or {}
        if line.get("mention"):
            return "nobody (a quoted term)"
        sp = line.get("speaker")
        return (chars.get(sp) or {}).get("name", sp) if sp else "the narrator, or nobody (written text)"

    ids = list(local)
    chapter = book["chapters"][chunk.chapter]
    before = preceding_text(book, chunk, context_words) if context_words else ""
    user = (f"Book: {book.get('title', '')} by {book.get('author', '')}\nChapter: {chapter.get('title', '')}\n\n"
            + (f"Story just before this passage (for context only):\n{before}\n\n" if before else "")
            + f"Passage:\n{text}\n\nWho says each quotation:\n"
            + "\n".join(f"[{n}] {who(local[n])}" for n in ids)
            + f"\n\nGive the delivery of every number from 1 to {len(ids)} once.")
    schema = {"type": "object", "properties": {"quotes": {
        "type": "array", "minItems": len(ids), "maxItems": len(ids),
        "items": {"type": "object", "properties": {
            "id": {"type": "string", "enum": ids},
            "delivery": {"type": "array", "maxItems": delivery_tags.MAX_TAGS,
                         "items": {"type": "string", "enum": delivery_tags.all_tags()}}},
                  "required": ["id", "delivery"]}}}, "required": ["quotes"]}
    result = await llm.complete(session, DELIVERY_PROMPT, user, schema, max_tokens=512 + 40 * len(ids),
                                logprobs=TOP_ALTERNATIVES)
    out = {}
    for q, odds in zip(result.get("quotes", []), delivery_odds(result.pop("_tokens", None) or [])):
        delivery, dodds = sure_delivery(q.get("delivery") or [], odds)
        for qid in local.get(str(q.get("id"))) or []:
            out[qid] = (delivery, dodds)
    return out


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
        prof = c["profile"] = c.get("profile") or {}  # (stored as null for some characters: setdefault keeps the None)
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


class BookQueue:
    """The books waiting for analysis, in order. It can be reordered (the Queue page) and is kept in a file, so
    a restart resumes in the same order."""

    def __init__(self, path: Path | None = None):
        self.items: list[str] = []
        self.path = path
        self.current = None  # the book being analysed (saved first, so a restart resumes it first)
        self._changed = asyncio.Event()

    def _save(self) -> None:
        if self.path is None:
            return
        order = ([self.current] if self.current else []) + self.items
        try:
            self.path.write_text(json.dumps({"order": order}), encoding="utf-8")
        except OSError:
            pass

    def saved_order(self) -> list[str]:
        try:
            return list(json.loads(self.path.read_text(encoding="utf-8")).get("order", [])) if self.path else []
        except (OSError, ValueError):
            return []

    async def put(self, book_id: str, before: str | None = None, at: int | None = None) -> None:
        if book_id not in self.items:
            if at is not None:
                self.items.insert(at, book_id)
            elif before in self.items:
                self.items.insert(self.items.index(before), book_id)
            else:
                self.items.append(book_id)
            self._save()
        self._changed.set()

    async def get(self) -> str:
        while not self.items:
            self._changed.clear()
            await self._changed.wait()
        book_id = self.items.pop(0)
        self._save()
        return book_id

    def empty(self) -> bool:
        return not self.items

    def remove(self, book_id: str) -> bool:
        if book_id not in self.items:
            return False
        self.items.remove(book_id)
        self._save()
        return True

    def reorder(self, ids: list[str]) -> None:
        """`ids` first, in that order; books not named keep their order after them."""
        wanted = [i for i in dict.fromkeys(ids) if i in self.items]
        self.items = wanted + [i for i in self.items if i not in wanted]
        self._save()

    def to_bottom(self, book_id: str) -> bool:
        if book_id not in self.items:
            return False
        self.items.remove(book_id)
        self.items.append(book_id)
        self._save()
        return True

    def set_current(self, book_id: str | None) -> None:
        self.current = book_id
        self._save()


class Analyzer:
    def __init__(self, app: web.Application, store: ScriptStore, llm: LocalLlm):
        self.app = app
        self.store = store
        self.llm = llm
        self.queue = BookQueue(Path(store.root).parent / "analysis-queue.json")
        self.task: asyncio.Task | None = None
        self.current: str | None = None
        self.job: asyncio.Task | None = None  # the current book's analysis
        self._interrupt: str | None = None  # "next" or "bottom": where the interrupted book goes
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
            self.app[ACTIVITY_KEY].on_crowd.append(self.on_crowd)
        self.task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
        await self.llm.stop()

    async def enqueue(self, book_id: str, keep_order: bool = False) -> None:
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            if s["status"] in ("queued", "running"):
                return
            s.update(status="queued", detail="waiting for the analyser")
            self.store.save_script(book_id, s)
        await self.queue.put(book_id, before=None if keep_order else self._later_volume(book_id))
        # Volumes are analysed in order: a later volume of its series being analysed now steps aside (keeping
        # what it has done; it continues right after).
        if not keep_order and self.current and self._is_earlier(book_id, self.current):
            self.analyse_now(book_id)

    def _volume(self, book_id: str) -> tuple[str, list[float]]:
        """(series, volume number) of a book; ("", []) when it has no series or no number."""
        from .series import book_series, series_slug, volume_key

        series = book_series(self.app, book_id) or ""
        vol = volume_key((self.store.summary(book_id) or {}).get("title", "")) if series else []
        return (series_slug(series) if series else ""), vol

    def _is_earlier(self, a: str, b: str) -> bool:
        """Whether book `a` is an earlier volume of book `b`'s series."""
        (sa, va), (sb, vb) = self._volume(a), self._volume(b)
        return bool(sa and va and vb and sa == sb and va < vb)

    def _later_volume(self, book_id: str) -> str | None:
        """The first waiting book that is a later volume of the same series: a volume is analysed before the
        ones after it, so they inherit its cast and voices."""
        return next((b for b in self.queue.items if self._is_earlier(book_id, b)), None)

    def _first_volume_waiting(self, book_id: str) -> str:
        """The earliest waiting volume of `book_id`'s series that comes before it, or `book_id` itself: however the
        queue was ordered (by hand, "analyse now", a restart), a series' volumes are analysed in order, so each
        one is labelled knowing the characters of the volumes before it."""
        earlier = [b for b in self.queue.items if self._is_earlier(b, book_id)]
        return min(earlier, key=lambda b: self._volume(b)[1]) if earlier else book_id

    async def _run(self) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Characters still on the old voice pool move to the new one first (minutes), before a resumed analysis.
            await self._backfill_pool(session, finish=True)
            while True:
                if self.queue.empty():
                    await self._backfill_pool(session)
                    await self._backfill_spoken_forms(session)
                    await self._backfill_deliveries(session)
                book_id = await self.queue.get()
                first = self._first_volume_waiting(book_id)
                if first != book_id:  # an earlier volume of its series is waiting: that one first
                    self.queue.remove(first)
                    await self.queue.put(book_id, at=0)
                    book_id = first
                self.current = book_id
                self.queue.set_current(book_id)
                self._interrupt = None
                try:
                    self.job = asyncio.ensure_future(self.analyse(session, book_id))
                    await self.job
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
                    if not self._interrupt or asyncio.current_task().cancelling():
                        self.current = None  # the server stops: the book stays first in the saved order, so it resumes first
                        raise
                    self.queue.current = None
                    await self._requeue(book_id, self._interrupt)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("analysis of %s failed", book_id)
                    async with self.store.lock(book_id):
                        s = self.store.script(book_id)
                        s.update(status="error", detail=str(exc)[:300])
                        self.store.save_script(book_id, s)
                self.current = None  # finished or failed
                self.queue.set_current(None)
                # Between analyses (nothing to do once every book has them; the book that was interrupted
                # by a restart finishes first).
                await self._backfill_pool(session)
                await self._backfill_spoken_forms(session)
                await self._backfill_deliveries(session)
                if self.queue.empty():
                    await self.llm.stop()  # free RAM/VRAM when idle

    async def _backfill_spoken_forms(self, session) -> None:
        """Finished books labelled before spoken forms existed get them between analyses (a few minutes each).
        Books waiting for (or in) an analysis get them when it ends. The books listened to last go first."""

        def last_heard(book_id: str) -> float:
            try:
                return (self.store.dir(book_id) / "book.elvenbook").stat().st_mtime
            except (OSError, ValueError):
                return 0.0

        for book_id in sorted(self.store.ids(), key=last_heard, reverse=True):
            s = self.store.script(book_id)
            if s.get("status") == "done" and s.get("lines") and s.get("spoken_forms") != SPOKEN_FORMS:
                try:
                    await self._spoken_forms(session, book_id)
                except Exception as exc:  # noqa: BLE001 - tried again before the next book
                    logger.warning("%s: no spoken forms (%s)", book_id, exc)
                    return

    async def _backfill_deliveries(self, session) -> None:
        """Finished books labelled before the delivery confidence existed get their deliveries scored again
        (score_deliveries: only tags the model is sure of are kept; speakers and corrections stay), between
        analyses, about 9 s a passage. The books listened to last go first. A new book's queued analysis waits
        for the book being scored (at most ~25 minutes)."""

        def last_heard(book_id: str) -> float:
            try:
                return (self.store.dir(book_id) / "book.elvenbook").stat().st_mtime
            except (OSError, ValueError):
                return 0.0

        for book_id in sorted(self.store.ids(), key=last_heard, reverse=True):
            s = self.store.script(book_id)
            if s.get("status") == "done" and s.get("lines") and s.get("delivery_scored") != DELIVERY_SCORED:
                try:
                    await self._score_deliveries(session, book_id)
                except Exception as exc:  # noqa: BLE001 - tried again before the next book
                    logger.warning("%s: deliveries not scored (%s)", book_id, exc)
                    return
                if not self.queue.empty():
                    return  # a book is waiting for its analysis

    async def _score_deliveries(self, session, book_id: str) -> None:
        """Scores the deliveries of every passage that has lines without the model's confidence, then marks the
        book done. Each passage is saved as it is scored, so a restart continues where it stopped."""
        book = self.store.book(book_id)
        if not book:
            return
        cfg, changed, done = self.llm.cfg, 0, 0
        for chunk in make_chunks(book, runs=mixed_tense_runs(book)):
            if not needs_scoring(self.store.script(book_id), chunk):
                continue
            s = self.store.script(book_id)
            scored = await self._gpu(lambda c=chunk, sc=s: score_deliveries(
                self.llm, session, book, sc, c, context_words=cfg.llm_label_context_words))
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                for qid, (delivery, dodds) in scored.items():
                    line = s["lines"].get(qid)
                    if line is None or line.get("mention"):
                        continue
                    changed += line.get("delivery") != delivery
                    line["delivery"] = delivery
                    line["dodds"] = dodds if dodds is not None else []
                self.store.save_script(book_id, s)
            done += 1
            if done % 25 == 0:
                logger.info("%s: deliveries scored in %d passages so far", book_id, done)
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            s["delivery_scored"] = DELIVERY_SCORED
            self.store.save_script(book_id, s)
        logger.info("%s: deliveries scored (%d passages, %d lines changed)", book_id, done, changed)

    async def _pool_picks(self, session, book_id: str, book: dict, s: dict, cids: list[str]) -> int:
        """Sets `pool_pick` on the characters `cids` of script `s` (in memory): the same character's pick in the
        series' other volumes when there is one (by name or alias), so they sound the same throughout; the LLM's
        otherwise."""
        from .series import book_series, series_slug

        series = series_slug(book_series(self.app, book_id) or "")
        known: dict[str, str] = {}
        if book_series(self.app, book_id):
            for other in self.store.ids():
                if other == book_id or series_slug(book_series(self.app, other) or "") != series:
                    continue
                for oc in self.store.script(other).get("characters", {}).values():
                    if oc.get("pool_pick"):
                        for n in (oc.get("name", ""), *oc.get("aliases", [])):
                            known.setdefault(norm_name(n), oc["pool_pick"])
        ask = []
        for cid in cids:
            c = s["characters"][cid]
            earlier = next((known[k] for k in map(norm_name, (c.get("name", ""), *c.get("aliases", [])))
                            if k in known and known[k] in pool_options(c)), None)
            if earlier:
                c["pool_pick"] = earlier
            else:
                ask.append(cid)
        picks = await self._gpu(lambda: pick_pool_voices(self.llm, session, book, s, ask)) if ask else {}
        for cid, vid in picks.items():
            s["characters"][cid]["pool_pick"] = vid
        return len(picks)

    async def _ensure_pool(self) -> bool:
        """Designs the pool's voices that don't exist yet, and again those whose description in POOL changed
        since (in place, same id: characters keep them; one the reader edited is left alone, known by its
        description no longer being the one it was designed from, kept in data/voice-pool.json). Up to
        POOL_DESIGN_TRIES seeds each, about 6 s a seed. False when the designer failed."""
        from .app import DESIGN_TEXT, VOICES_KEY, design_clip, design_voice, wav_bytes

        voices = self.app[VOICES_KEY]
        engine = self.llm.cfg.voice_design_engine
        # What each pool voice was designed from: one whose description no longer matches was edited by the reader
        made_path = Path(self.store.root).parent / "voice-pool.json"
        try:
            made = json.loads(made_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            made = {}
        for vid, (name, description) in POOL.items():
            v = voices.get(vid)
            if v and v.instruct == description:
                continue
            edited = v is not None and vid in made and (v.instruct != made[vid] or v.description != made[vid])
            if edited:
                continue  # the reader's own edit
            try:
                if v is None:
                    await design_voice(self.app, engine, description, f"Pool: {name}", source="character",
                                       voice_id=vid, seed=7, tries=POOL_DESIGN_TRIES)
                else:
                    rate, pcm, seed = await design_clip(self.app, engine, description, DESIGN_TEXT, 7, POOL_DESIGN_TRIES)
                    voices.replace_audio(vid, wav_bytes(pcm, rate), DESIGN_TEXT, instruct=description,
                                         description=description, seed=seed)
            except Exception as exc:  # noqa: BLE001 - tried again before the next analysis
                logger.warning("generic voice %s not designed (%s)", name, exc)
                return False
            made[vid] = description
            made_path.write_text(json.dumps(made, indent=1), encoding="utf-8")
            logger.info("designed generic voice %s", name)
        return True

    async def _backfill_pool(self, session, finish: bool = False) -> None:
        try:
            await self._move_to_pool(session, finish)
        except Exception:  # noqa: BLE001 - tried again between analyses
            logger.exception("moving characters to the new voice pool failed")

    async def _move_to_pool(self, session, finish: bool) -> None:
        """Characters on a voice of an earlier pool get the best fit from this one (POOL), book by book in volume
        order (later volumes reuse the picks), then the old pool voices are deleted. Each book is one or a few
        LLM calls; a book waiting for its analysis goes first unless `finish`."""
        from .app import VOICES_KEY
        from .series import SERIES_KEY, book_series, remember, volume_key

        def todo() -> list[str]:  # every book not being analysed or cast right now (one waiting in the queue is not)
            return [b for b in self.store.ids() if self.store.script(b).get("pool_v") != POOL_V
                    and self.store.script(b).get("status") not in ("running", "casting")]

        voices_have = self.app[VOICES_KEY].get  # the pool is in use: keep its voices up to date

        books = todo()
        moving = any(is_pool(c.get("voice")) for b in books for c in self.store.script(b).get("characters", {}).values())
        if (moving or any(voices_have(vid) for vid in POOL)) and not await self._ensure_pool():
            return
        books.sort(key=lambda b: ((book_series(self.app, b) or "").lower(),
                                  volume_key((self.store.summary(b) or {}).get("title", "")) or []))
        for book_id in books:
            book = self.store.book(book_id)
            s = self.store.script(book_id)
            cids = [cid for cid, c in s["characters"].items() if is_pool(c.get("voice"))]
            try:
                if cids and book:
                    await self._pool_picks(session, book_id, book, s, cids)
            except Exception as exc:  # noqa: BLE001 - tried again before the next book
                logger.warning("%s: generic voices not picked (%s)", book_id, exc)
                return
            async with self.store.lock(book_id):
                fresh = self.store.script(book_id)
                moved = 0
                for cid in cids:
                    f = fresh["characters"].get(cid)
                    if f is None or not is_pool(f.get("voice")):
                        continue
                    if s["characters"][cid].get("pool_pick"):
                        f["pool_pick"] = s["characters"][cid]["pool_pick"]
                    vid = pool_voice(f)
                    moved += vid != f["voice"]
                    f["voice"] = vid
                fresh["pool_v"] = POOL_V
                self.store.save_script(book_id, fresh)
            if cids:
                remember(self.app, book_id)
                logger.info("%s: %d characters on generic voices, %d moved to another", book_id, len(cids), moved)
            if not finish and not self.queue.empty():
                return  # a book is waiting for its analysis
        if todo():
            return
        # Every book is on the new pool: the series' memories too, then the old voices go.
        series = self.app[SERIES_KEY]
        for data in series.all():
            data = series.get(data["name"])
            stale = [e for e in data["characters"].values() if is_pool(e.get("voice")) and e["voice"] not in POOL]
            for e in stale:
                e["voice"] = pool_voice(e)
            if stale:
                series.save(data)
        used = {c.get("voice") for b in self.store.ids() for c in self.store.script(b).get("characters", {}).values()}
        voices = self.app[VOICES_KEY]
        for v in voices.list():
            if is_pool(v.id) and v.id not in POOL and v.id not in used:
                voices.delete(v.id)
                logger.info("deleted the old generic voice %s", v.name)

    async def _spoken_forms(self, session, book_id: str) -> None:
        """Checks every line of the book for a spoken form ("say": find_spoken_forms) and marks it done."""
        book = self.store.book(book_id)
        found = 0
        for batch in spoken_batches(book, self.store.script(book_id)) if book else []:
            lines = self.store.script(book_id).get("lines", {})
            said = {qid: lines[qid]["say"] for qid, _ in batch if (lines.get(qid) or {}).get("say")}
            says = await self._gpu(lambda b=batch, s=said: find_spoken_forms(self.llm, session, b, s))
            if says:
                async with self.store.lock(book_id):
                    s = self.store.script(book_id)
                    for qid, say in says.items():
                        if qid in s["lines"]:
                            s["lines"][qid]["say"] = say
                    self.store.save_script(book_id, s)
                found += len(says)
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            s["spoken_forms"] = SPOKEN_FORMS
            self.store.save_script(book_id, s)
        logger.info("%s: %d lines spelled the way they sound get a spoken form", book_id, found)

    async def _requeue(self, book_id: str, where: str) -> None:
        """The current book was set aside (another one is analysed now, or it went to the bottom): its finished
        passages are kept, and analysing it again continues from there."""
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            s.update(status="queued", detail="set aside; continues where it stopped")
            self.store.save_script(book_id, s)
        await self.queue.put(book_id, at=min(1, len(self.queue.items)) if where == "next" else None)

    def _set_aside(self, where: str) -> bool:
        if self.current is None or self.job is None or self.job.done():
            return False
        self._interrupt = where
        self.job.cancel()
        return True

    def analyse_now(self, book_id: str) -> bool:
        """Analyse a waiting book right away: the current one stops (keeping what it has done) and goes next.
        Not when the current one is an earlier volume of its series (volumes go in order): it goes next instead."""
        if book_id not in self.queue.items:
            return False
        self.queue.reorder([book_id])
        if not (self.current and self._is_earlier(self.current, book_id)):
            self._set_aside("next")
        return True

    def send_to_bottom(self, book_id: str) -> bool:
        if book_id == self.current:
            return self._set_aside("bottom")
        return self.queue.to_bottom(book_id)

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
                                                   after_words=cfg.llm_label_after_words,
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

        if self.llm.cfg.llm_backend == "remote":
            # On the inference server: nothing here waits for (or takes) the GPU.
            return await make_coro()
        manager = self.app[ENGINES_KEY]
        mode = self.llm.cfg.llm_mode
        if mode == "adaptive" and ACTIVITY_KEY in self.app:
            activity = self.app[ACTIVITY_KEY]
            while True:
                if self.llm.cfg.llm_backend == "remote":  # switched to the inference server meanwhile
                    return await make_coro()
                if not interactive:
                    await activity.wait_for(lambda: not activity.designing, on_wait=self._note_designing)
                    # Two or more people listening to new speech: the speech models get the GPU to themselves.
                    if activity.crowded:
                        await activity.wait_for(lambda: not activity.crowded, on_wait=lambda: self._note_paused(
                            "paused while several people are listening; continues a few minutes after"))
                        continue
                    # The speech models leave no room for it next to them: it waits until nobody is using the site.
                    if activity.in_use and not shared_fits(self.llm.cfg):
                        await activity.wait_for(lambda: not activity.in_use or shared_fits(self.llm.cfg),
                                                on_wait=lambda: self._note_paused(
                                                    "paused while someone reads: the speech models fill the GPU"))
                        continue
                if interactive or activity.in_use:
                    if not (self.llm.running_gpu and self.llm.shared):
                        # Starting next to the speech model: let it come back and answer first.
                        await manager.wait_settled()
                    async with manager.exclusive(evict=False):
                        if (activity.designing or activity.crowded) and not interactive:
                            continue
                        self.llm.gpu, self.llm.shared, self.llm.preempted = True, True, False
                        try:
                            return await make_coro()
                        except LlmPreempted:
                            logger.info("LLM stopped for the voice designer, or for several people listening")
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
                        self.llm.reserve_gb = manager.parked_vram_gb
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
            self.llm.reserve_gb = manager.parked_vram_gb
            try:
                return await make_coro()
            finally:
                if manager.tts_waiting:
                    await self.llm.stop()

    async def _note_designing(self) -> None:
        """Say on the book being analysed that it waits while voices are designed."""
        await self._note_paused("paused while you design voices; continues 10 minutes after the last one")

    async def _note_paused(self, why: str) -> None:
        """Say on the book being analysed why it waits."""
        book_id = self.current
        if not book_id:
            return
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            if s.get("status") in ("running", "queued"):
                s["detail"] = why
                self.store.save_script(book_id, s)

    async def on_crowd(self) -> None:
        """A second person started listening to new speech ("adaptive"): the LLM leaves the GPU at once (its step is
        done again later); the speech models get all of it."""
        if self.llm.cfg.llm_mode == "adaptive" and self.llm.proc is not None:
            await self.llm.preempt()

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
            if self.app[CONFIG_KEY].narration_engine:
                await manager.preload_narration(self.app[CONFIG_KEY].narration_engine)
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
        if not self.store.script(book_id)["chunks_done"]:
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                s["spoken_forms"] = SPOKEN_FORMS  # every passage gets them from labelling
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
                apply_chunk_result(s, result, local, vote=chunk.key)
                model = result.get("_model") or self.llm.model_name
                log_answer(self.store, book_id, chunk.key, model, result, local)
                if chunk.tense:
                    s["last_tense"] = chunk.tense
                    other = "past" if chunk.tense == "present" else "present"
                    # The first narrator seen in each tense sticks, and the two tenses have different
                    # narrators: one confused passage must not re-label every later one.
                    if (chunk.tense not in by_tense and find_character(s, s.get("narrator") or "")
                            and find_character(s, s["narrator"]) != find_character(s, by_tense.get(other) or "")):
                        by_tense[chunk.tense] = s["narrator"]
                s["chunks_done"].append(chunk.key)
                s["model"] = model
                s.setdefault("stats", {}).setdefault("tokens", 0)
                s["stats"]["tokens"] += int(result.get("_usage", {}).get("total_tokens", 0))
                self.store.save_script(book_id, s)
        if self.store.script(book_id).get("spoken_forms") != SPOKEN_FORMS:
            # Some passages were labelled before spoken forms existed.
            async with self.store.lock(book_id):
                s = self.store.script(book_id)
                s.update(detail="finding lines spelled the way they sound")
                self.store.save_script(book_id, s)
            await self._spoken_forms(session, book_id)
        if self.store.script(book_id).get("delivery_scored") != DELIVERY_SCORED:
            # Passages labelled before the delivery confidence existed (a resumed analysis); quick otherwise.
            await self._score_deliveries(session, book_id)
        async with self.store.lock(book_id):
            s = self.store.script(book_id)
            renamed = canonical_names(book, s)
            if renamed:
                logger.info("%s: named by the text: %s", book_id, ", ".join(f"{a} -> {b}" for a, b in renamed))
            s.update(detail="describing the characters")
            self.store.save_script(book_id, s)
        if renamed:
            name_voices(self.app)
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
        try:  # which generic voice fits each character without a voice of their own (main ones get their own)
            await self._pool_picks(session, book_id, book, s,
                                   [cid for cid, c in s["characters"].items() if c.get("role") != "main"])
        except LlmError as exc:
            logger.warning("picking generic voices failed (%s); the plain ones are used", exc)
        if self.llm.running_gpu:
            await self.llm.stop()  # give the GPU back to TTS
        async with self.store.lock(book_id):
            fresh = self.store.script(book_id)
            for cid, c in s["characters"].items():
                if cid in fresh["characters"]:
                    for k in ("gender", "age", "accent", "temperament", "appearance", "description",
                              "role_auto", "role_source", "role_note", "role_flag", "profile", "pool_pick"):
                        fresh["characters"][cid][k] = c.get(k, fresh["characters"][cid].get(k))
                    f = fresh["characters"][cid]
                    f["role"] = f.get("role_user") if f.get("role_user") in roles.ROLES else f.get("role_auto")
            fresh["roles_state"] = s.get("roles_state")
            fresh.update(status="done", progress=1.0,
                         detail=f"{len(fresh['lines'])} lines in {time.perf_counter() - t0:.0f}s")
            fresh.setdefault("stats", {})["seconds"] = round(time.perf_counter() - t0, 1)
            reconcile_narrators(book, fresh)
            fresh["narration"] = compute_narration(book, fresh)
            # Characters the series knows from any volume (later ones too) who speak here take its voice and
            # profile; the labeller was only told of earlier volumes' characters. The others go again (prune).
            from . import series as series_memory

            series_memory.seed(self.app, book_id, fresh, earlier_only=False)
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


def log_answer(store: "ScriptStore", book_id: str, chunk_key: str, model: str, result: dict, local: dict) -> None:
    """Keeps what the model answered for each passage (data/books/<id>/llm-answers.jsonl), so a wrong label
    can be traced to the answer, and the cast, it came from."""
    entry = {"t": round(time.time()), "chunk": chunk_key, "model": model, "numbers": local,
             "answer": {k: v for k, v in result.items() if not k.startswith("_") or k in ("_editable",)},
             "tokens": result.get("_usage", {}).get("total_tokens")}
    try:
        with open(store.dir(book_id) / "llm-answers.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("%s: could not keep the LLM's answer: %s", book_id, exc)


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

# The voice pool: generic voices for characters without a voice of their own, by gender, age and how strongly
# they speak. The LLM picks the best fit for each character (pick_pool_voices, `pool_pick`) from the voices that
# suit their gender and age; without a pick they get the plain one. Voices of an earlier pool (any other "pool-"
# voice) are replaced: every character on one gets a pick from this pool, then the old voices are deleted.
POOL = {
    "pool-f-elderly": ("Female Elderly", "A grandmother in her sixties with a warm, mellow, clear voice."),
    "pool-f-adult": ("Female Adult", "An adult woman in her thirties or forties with an ordinary, mid-pitched voice, natural and clear."),
    "pool-f-adult-strong": ("Female Adult - Strong", "An adult woman in her thirties or forties with a strong, full, confident voice that carries, firm and resonant."),
    "pool-f-adult-soft": ("Female Adult - Soft", "An adult woman in her thirties or forties with a soft, gentle, quiet voice, warm and light."),
    "pool-f-adult-young": ("Female Adult - Young", "A young woman in her early twenties with a light, lively, bright voice."),
    "pool-f-teen": ("Female Teen", "A teenage girl of about fifteen with a clear, youthful voice, natural and expressive."),
    "pool-f-teen-strong": ("Female Teen - Strong", "A teenage girl of about fifteen with a strong, bold, energetic voice, loud and spirited."),
    "pool-f-teen-soft": ("Female Teen - Soft", "A teenage girl of about fifteen with a soft, shy, gentle voice, quiet and light."),
    "pool-f-child": ("Female Child", "A little girl of about eight with a high, clear, childlike voice."),
    "pool-m-elderly": ("Male Elderly", "An elderly man in his seventies with a low, slightly rough voice, speaking unhurriedly."),
    "pool-m-adult": ("Male Adult", "An adult man in his thirties or forties with an ordinary, mid-pitched voice, natural and clear."),
    "pool-m-adult-strong": ("Male Adult - Strong", "An adult man in his thirties or forties with a strong, deep, commanding voice that carries, firm and resonant."),
    "pool-m-adult-soft": ("Male Adult - Soft", "An adult man in his thirties or forties with a soft, gentle, quiet voice, mild and warm."),
    "pool-m-adult-young": ("Male Adult - Young", "A young man in his early twenties with a bright, energetic, light voice."),
    "pool-m-teen": ("Male Teen", "A teenage boy of about fifteen with a clear, youthful voice that has just broken."),
    "pool-m-teen-strong": ("Male Teen - Strong", "A teenage boy of about fifteen with a strong, loud, brash voice, full of energy."),
    "pool-m-teen-soft": ("Male Teen - Soft", "A teenage boy of about fifteen with a soft, quiet, gentle voice, a little shy."),
    "pool-m-child": ("Male Child", "A little boy of about eight with a high, clear, childlike voice."),
}
POOL_V = 2  # books whose characters on generic voices were given picks from this pool
POOL_DESIGN_TRIES = 12  # seeds tried for a clean design (no hum in the pauses): hundreds of characters share each
_POOL_KINDS = {  # a character's age -> the kinds of pool voice that can fit them
    "child": ("child",),
    "teen": ("teen", "teen-strong", "teen-soft"),
    "young adult": ("adult-young", "adult", "adult-strong", "adult-soft"),
    "adult": ("adult", "adult-strong", "adult-soft"),
    "middle-aged": ("adult", "adult-strong", "adult-soft"),
    "elderly": ("elderly",),
}
_POOL_PLAIN = {"child": "child", "teen": "teen", "young adult": "adult-young", "elderly": "elderly"}


def is_pool(voice_id: str | None) -> bool:
    return (voice_id or "").startswith("pool-")


def _gender_age(c: dict) -> tuple[str, str]:
    from .profile import effective

    p = effective(c)
    return (p.get("gender") or c.get("gender") or "unknown"), (p.get("apparent_age") or c.get("age") or "unknown")


def pool_options(c: dict) -> list[str]:
    """The pool voices that suit a character's gender and age (all of them when those are unknown)."""
    gender, age = _gender_age(c)
    genders = {"female": ("f",), "male": ("m",)}.get(gender, ("f", "m"))
    kinds = _POOL_KINDS.get(age) or tuple(dict.fromkeys(k for ks in _POOL_KINDS.values() for k in ks))
    return [vid for vid in (f"pool-{g}-{k}" for g in genders for k in kinds) if vid in POOL]


def pool_voice(c: dict) -> str:
    """A character's generic voice: the LLM's pick while it still suits their gender and age, otherwise the plain
    voice for them."""
    if c.get("pool_pick") in pool_options(c):
        return c["pool_pick"]
    gender, age = _gender_age(c)
    return f"pool-{'f' if gender == 'female' else 'm'}-{_POOL_PLAIN.get(age, 'adult')}"


POOL_PROMPT = """You cast generic voices for the minor characters of a novel's audiobook. For each character, pick the voice that fits best from the ones listed for them, by gender and age and by how they speak: a "Strong" voice for someone forceful (loud, commanding, bold, brash, gruff, stern, boisterous), a "Soft" one for someone gentle (quiet, timid, meek, shy, soothing, frail, mild), and the plain voice when neither stands out. "Adult - Young" is someone in their early twenties. When the gender is not given, judge it from the name and description. Return every character exactly once."""
POOL_BATCH = 40


async def pick_pool_voices(llm, session, book: dict, script: dict, cids: list[str]) -> dict[str, str]:
    """The LLM's best-fitting pool voice for each of `cids` (character id -> voice id), in batches."""
    by_name = {name: vid for vid, (name, _) in POOL.items()}
    picks: dict[str, str] = {}
    for i in range(0, len(cids), POOL_BATCH):
        batch = cids[i:i + POOL_BATCH]
        rows = []
        for cid in batch:
            c = script["characters"][cid]
            from .profile import effective

            p = effective(c)
            gender, age = _gender_age(c)
            about = "; ".join(x for x in (
                f"speaks: {p['speech_style'][:160]}" if p.get("speech_style") else "",
                (p.get("notes") or "")[:160], (p.get("appearance") or c.get("description") or "")[:120]) if x)
            rows.append(f"- {cid}: {c['name']} ({gender}, {age}). {about}\n  voices: "
                        + ", ".join(POOL[v][0] for v in pool_options(c)))
        schema = {"type": "object", "properties": {"characters": {
            "type": "array", "minItems": len(batch), "maxItems": len(batch),
            "items": {"type": "object", "properties": {
                "id": {"type": "string", "enum": batch},
                "voice": {"type": "string", "enum": list(by_name)}},
                "required": ["id", "voice"]}}}, "required": ["characters"]}
        user = f"Book: {book.get('title', '')}\n\nCharacters:\n" + "\n".join(rows)
        result = await llm.complete(session, POOL_PROMPT, user, schema, max_tokens=256 + 40 * len(batch))
        for a in result.get("characters", []):
            vid, cid = by_name.get(a.get("voice")), a.get("id")
            if cid in batch and cid not in picks and vid in pool_options(script["characters"][cid]):
                picks[cid] = vid
    return picks


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


def name_voices(app: web.Application) -> list[tuple[str, str]]:
    """Voices designed for a character are named after them. A character first known by a description and
    renamed later ("Dad", until the book says he is Gunther) leaves their voice under the old name: it takes the
    character's name once no character it voices is called by its name any more (the same character may still
    carry it in another book: "Lord Ortwin" there, "Ortwin" here, so the voice stays put rather than flip back
    and forth). A name the reader gave the voice is not one of the character's names, so it is kept. Returns
    (old, new) for each voice renamed."""
    from collections import Counter

    from .app import VOICES_KEY

    store = app.get(DRAMA_STORE_KEY)
    if store is None or VOICES_KEY not in app:
        return []
    users: dict[str, list[dict]] = {}
    for book_id in store.ids():
        for c in store.script(book_id).get("characters", {}).values():
            vid = c.get("voice") or ""
            if vid and not vid.startswith("pool-"):
                users.setdefault(vid, []).append(c)
    voices, renamed = app[VOICES_KEY], []
    for vid, cs in users.items():
        v = voices.get(vid)
        if v is None or v.source != "character" or any(c.get("name") == v.name for c in cs):
            continue
        names = Counter(c["name"] for c in cs if c.get("name") and v.name in c.get("aliases", []))
        if names:
            new = max(names, key=lambda n: (names[n], n))
            voices.update(vid, name=new)
            renamed.append((v.name, new))
    if renamed:
        logger.info("voices named after their characters: %s", ", ".join(f"{a} -> {b}" for a, b in renamed))
    return renamed


def delete_unused_voices(app: web.Application) -> list[str]:
    """Voices made for characters that nothing reaches any more (a character renamed, merged or dropped by a new
    analysis; an earlier design replaced without a trace) are deleted, with their earlier versions. Kept: voices any
    book's character uses, the series' memories, the default reader voice, the generic voices, and voices a book's
    reports name (a character's past voices). Your own and imported voices are never deleted."""
    from . import reports
    from .app import CONFIG_KEY, VOICES_KEY
    from .series import SERIES_KEY

    store = app.get(DRAMA_STORE_KEY)
    if store is None or VOICES_KEY not in app:
        return []
    keep = {app[CONFIG_KEY].default_voice} if CONFIG_KEY in app else set()
    for book_id in store.ids():
        keep |= {c.get("voice") for c in store.script(book_id).get("characters", {}).values()}
        for r in reports.read(app, book_id):
            for side in (r.get("before") or {}, r.get("after") or {}):
                keep |= {side.get("voice"), side.get("id")}
    if SERIES_KEY in app:
        for data in app[SERIES_KEY].all():
            keep |= {e.get("voice") for e in data.get("characters", {}).values()}
    voices, deleted = app[VOICES_KEY], []
    for v in voices.list():
        if v.source == "character" and not is_pool(v.id) and v.id not in keep and voices.delete(v.id):
            deleted.append(v.id)
    if deleted:
        logger.info("deleted %d voices nothing uses any more: %s", len(deleted), ", ".join(deleted))
    return deleted


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
            vid = pool_voice(c)
            v = voices.get(vid) or await design(
                app, engine, POOL[vid][1], f"Pool: {POOL[vid][0]}", source="character", voice_id=vid, seed=7,
                tries=POOL_DESIGN_TRIES,
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
    from .app import ENGINES_KEY, viewer_version

    app = request.app
    analyzer = app[ANALYZER_KEY]
    book = analyzer.current
    title = ((app[DRAMA_STORE_KEY].summary(book) or {}).get("title", "") if book else "")
    return web.json_response({
        "viewer": viewer_version(),
        "mode": analyzer.llm.cfg.llm_mode,
        "activity": app[ACTIVITY_KEY].json() if ACTIVITY_KEY in app else None,
        "engines": app[ENGINES_KEY].status(),
        # "backend": "remote" analysis runs on the inference server and never waits for this GPU.
        "llm": {**analyzer.llm.info(), "backend": analyzer.llm.cfg.llm_backend, "book": book, "title": title},
    })


def _queue_json(app: web.Application) -> dict:
    """The analysis queue for the Queue page: the book being analysed, the books waiting (in order) and the
    books whose voices are still to be created."""
    store = app[DRAMA_STORE_KEY]
    analyzer = app[ANALYZER_KEY]
    from .series import book_series

    def item(book_id: str) -> dict:
        summary = store.summary(book_id) or {}
        s = store.script(book_id)
        return {"id": book_id, "title": summary.get("title", book_id), "author": summary.get("author", ""),
                "words": summary.get("words", 0), "series": book_series(app, book_id),
                "status": s.get("status", "none"), "progress": s.get("progress", 0.0), "detail": s.get("detail", ""),
                "lines": len(s.get("lines", {})), "cover": store.cover(book_id) is not None,
                "cover_v": cover_version(store, book_id)}

    casting = [b for b in store.ids() if store.script(b).get("status") == "casting"]
    return {"current": item(analyzer.current) if analyzer.current else None,
            "queued": [item(b) for b in analyzer.queue.items],
            "casting": [item(b) for b in casting]}


async def queue_get(request: web.Request) -> web.Response:
    return web.json_response(_queue_json(request.app))


async def queue_put(request: web.Request) -> web.Response:
    """{"order": [book ids]}: those books first, in that order (the others keep theirs after them)."""
    body = await request.json() if request.can_read_body else {}
    order = body.get("order") if isinstance(body, dict) else None
    if not isinstance(order, list) or not all(isinstance(b, str) for b in order):
        return web.json_response({"error": "Send {\"order\": [book ids]}"}, status=400)
    request.app[ANALYZER_KEY].queue.reorder(order)
    return web.json_response(_queue_json(request.app))


async def queue_now(request: web.Request) -> web.Response:
    """Analyse a waiting book right away; the current one stops, keeps its progress and goes next."""
    if not request.app[ANALYZER_KEY].analyse_now(request.match_info["book"]):
        return web.json_response({"error": "That book isn't waiting in the queue"}, status=404)
    await asyncio.sleep(0.2)  # let the switch happen, so the answer shows it
    return web.json_response(_queue_json(request.app))


async def queue_bottom(request: web.Request) -> web.Response:
    """Send a book (waiting, or the one being analysed: it stops, keeping its progress) to the end of the queue."""
    if not request.app[ANALYZER_KEY].send_to_bottom(request.match_info["book"]):
        return web.json_response({"error": "That book isn't in the queue"}, status=404)
    await asyncio.sleep(0.2)
    return web.json_response(_queue_json(request.app))


async def queue_delete(request: web.Request) -> web.Response:
    """Take a waiting book out of the queue. Its analysis so far is kept; analysing it again continues it."""
    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    if not request.app[ANALYZER_KEY].queue.remove(book_id):
        return web.json_response({"error": "That book isn't waiting in the queue"}, status=404)
    async with store.lock(book_id):
        s = store.script(book_id)
        s.update(status="none", detail="taken out of the analysis queue")
        store.save_script(book_id, s)
    return web.json_response(_queue_json(request.app))


async def list_books(request: web.Request) -> web.Response:
    from . import bookmark
    from .permissions import can_book
    from .profiles import current
    from .series import series_name as _series_name, volume_key

    store = request.app[DRAMA_STORE_KEY]
    pid = current(request)
    books = []
    for book_id in store.ids():
        summary = store.summary(book_id) or {}
        s = store.script(book_id)
        public = ScriptStore.public(s)
        place = bookmark.summary(request.app, book_id, store.book(book_id), pid) \
            if pid and bookmark.path(request.app, book_id).exists() else None
        books.append({
            **summary,
            "id": book_id,
            # Where the requesting user (their token) is in it: {chapter, title, percent, updated}.
            **({"bookmark": place} if place else {}),
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
            "cover_v": cover_version(store, book_id),
            "updated": s.get("updated") or summary.get("uploaded", 0),
            "can_edit": can_book(request, book_id),
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


async def rename_book(request: web.Request) -> web.Response:
    """{"title": "..."} renames a book. Volume order in a series follows the title ("Vol. 19", "Book 3",
    "Part 2 Volume 1"). The book stays in its series, and uploading the file again keeps the new title."""
    from .series import book_series, series_info, series_name

    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    try:
        book = store.book(book_id)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if book is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    body = await request.json() if request.can_read_body else {}
    title = " ".join(str(body.get("title") or "").split()) if isinstance(body, dict) else ""
    if not title:
        return web.json_response({"error": "Give the book a title"}, status=400)
    async with store.lock(book_id):
        series = book_series(request.app, book_id)
        meta_path = store.dir(book_id) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else dict(store.summary(book_id) or {})
        shared = any(b != book_id and book_series(request.app, b).lower() == series.lower() for b in store.ids())
        if not meta.get("series") and shared and series_name(title).lower() != series.lower():
            meta["series"] = series  # the new title alone wouldn't say which series: keep the one it's in
        meta.update(title=title, title_user=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        book["title"] = title
        (store.dir(book_id) / "book.json").write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
    return web.json_response(series_info(request.app, book_id))


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


COVER_WIDTHS = (120, 240, 400, 800)
MAX_SPRITE = 120  # covers per sheet


def cover_thumbnail(path: Path, width: int) -> Path:
    """A small JPEG of the cover (made once, kept next to it; made again when the cover changes). Book covers
    are ~2 MB at 1400 px, which the library and queue show at 34-200 px."""
    out = path.with_name(f"cover-{width}.jpg")
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
        tmp = out.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=82, optimize=True, progressive=True)
    tmp.replace(out)
    return out


async def get_cover(request: web.Request) -> web.StreamResponse:
    """The cover; ?w=N gives a thumbnail at least N px wide (from COVER_WIDTHS). With ?v= (the cover's version,
    from the book list) the browser keeps it, so the library doesn't download covers again on every visit."""
    try:
        path = request.app[DRAMA_STORE_KEY].cover(request.match_info["book"])
    except ValueError:
        path = None
    if path is None:
        return web.json_response({"error": "no cover"}, status=404)
    try:
        want = int(request.query.get("w") or 0)
    except ValueError:
        want = 0
    if want > 0:
        width = next((w for w in COVER_WIDTHS if w >= want), COVER_WIDTHS[-1])
        try:
            path = await asyncio.to_thread(cover_thumbnail, path, width)
        except Exception:  # noqa: BLE001 - an unreadable cover is still served as it is
            logger.exception("cover thumbnail for %s failed", request.match_info["book"])
    cache = "private, max-age=31536000, immutable" if request.query.get("v") else "private, no-cache"
    return web.FileResponse(path, headers={"Content-Type": "image/jpeg", "Cache-Control": cache})


def cover_version(store: "ScriptStore", book_id: str) -> int:
    path = store.cover(book_id)
    return int(path.stat().st_mtime) if path else 0


def cover_sprite(store: "ScriptStore", ids: list[str], width: int) -> Path:
    """One JPEG with the books' covers side by side, each cropped to a 2:3 cell `width` px wide (the library's
    card shape), so a page loads a series' covers in one request. Kept on disk under its contents' hash."""
    from PIL import Image, ImageOps

    height = round(width * 1.5)
    covers = [store.cover(b) for b in ids]
    key = hashlib.sha1(json.dumps([width, [(b, int(p.stat().st_mtime) if p else 0) for b, p in zip(ids, covers)]])
                       .encode()).hexdigest()[:20]
    folder = Path(store.root).parent / "cache" / "covers"
    out = folder / f"{key}.jpg"
    if out.exists():
        return out
    folder.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (width * len(ids), height), (200, 200, 200))
    for i, path in enumerate(covers):
        if path is None:
            continue
        try:
            with Image.open(path) as im:
                sheet.paste(ImageOps.fit(im.convert("RGB"), (width, height), Image.LANCZOS), (i * width, 0))
        except Exception:  # noqa: BLE001 - one unreadable cover leaves a grey cell
            logger.exception("cover of %s unreadable", ids[i])
    for old in sorted(folder.glob("*.jpg"), key=lambda p: p.stat().st_mtime)[:-200]:  # keep the cache small
        old.unlink(missing_ok=True)
    tmp = out.with_suffix(".tmp")
    sheet.save(tmp, "JPEG", quality=80, optimize=True, progressive=True)
    tmp.replace(out)
    return out


async def get_covers(request: web.Request) -> web.StreamResponse:
    """?ids=a,b,c&w=240: the covers of several books in one image (cover_sprite), left to right in that order.
    With ?v= (any version string from the book list) the browser keeps it."""
    store = request.app[DRAMA_STORE_KEY]
    ids = [b for b in request.query.get("ids", "").split(",") if b][:MAX_SPRITE]
    if not ids or not all(_ID_RE.match(b) for b in ids):
        return web.json_response({"error": "ids: book ids, comma-separated"}, status=400)
    try:
        want = int(request.query.get("w") or 240)
    except ValueError:
        want = 240
    width = next((w for w in COVER_WIDTHS if w >= want), COVER_WIDTHS[-1])
    path = await asyncio.to_thread(cover_sprite, store, ids, width)
    cache = "private, max-age=31536000, immutable" if request.query.get("v") else "private, no-cache"
    return web.FileResponse(path, headers={"Content-Type": "image/jpeg", "Cache-Control": cache})


async def upload_book(request: web.Request) -> web.Response:
    book_id = request.match_info["book"]
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"error": "Body must be JSON"}, status=400)
    if not isinstance(body.get("chapters"), list):
        return web.json_response({"error": "chapters missing"}, status=400)
    store = request.app[DRAMA_STORE_KEY]
    from .profiles import current

    try:
        old_book = store.book(book_id)
        store.save_book(book_id, body, added_by=current(request))
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
    if s.get("status") == "done" and ("narration" not in s or s.get("mentions_v") != MENTIONS_V
                                      or (s["characters"] and not any(c.get("role") for c in s["characters"].values()))
                                      or s.get("names_v") != 2 or s.get("voice_desc_v") != 1
                                      or s.get("profile_v") != 2):
        async with store.lock(book_id):
            s = store.script(book_id)
            book = store.book(book_id)
            if "narration" not in s:  # analysed before narration existed
                reconcile_narrators(book, s)
                refresh_narration(store, book_id, s)
            if s.get("mentions_v") != MENTIONS_V:  # analysed before quoted terms were told apart from dialogue
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
    from .app import CONFIG_KEY, ENGINES_KEY, TUNING, default_tuning

    cfg = request.app[CONFIG_KEY]
    pub = ScriptStore.public(s)
    defaults = default_tuning(cfg, cfg.default_engine, request.app[ENGINES_KEY].specs.get(cfg.default_engine))
    # The per-line settings the speech model takes: the line dialog shows a slider for each.
    pub["tuning"] = {name: {"default": d, "min": TUNING[name]["range"][0], "max": TUNING[name]["range"][1],
                            "step": TUNING[name]["step"]} for name, d in defaults.items()}
    # The approved delivery tags (the line dialog suggests them): tag -> what it covers.
    pub["delivery_tags"] = {t: delivery_tags.APPROVED[delivery_tags.split(t)[1]][1] for t in delivery_tags.all_tags()}
    pub["delivery_min_conf"] = delivery_tags.MIN_CONF
    from .permissions import can_book

    pub["can_edit"] = can_book(request, book_id)  # whether this user may change the book (else it's read-only)
    return web.json_response(pub)


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
    """{speaker?, delivery?, mention?, guidance?, temperature?}: the reader's fix of a line (a speaker id, or a
    name for someone new; mention: not dialogue, read as part of the narration). Each speaker, delivery or mention
    change is kept as a report of incorrect analysis (reports.py);
    guidance and temperature (app.TUNING; null = the speech model's default) tune the speech only and are not
    corrections."""
    from . import reports
    from .app import CONFIG_KEY, ENGINES_KEY, TUNING, line_tuning

    store = request.app[DRAMA_STORE_KEY]
    book_id, qid = request.match_info["book"], request.match_info["line"]
    body = await request.json()
    detached: dict[str, list[str]] = {}
    created = False
    async with store.lock(book_id):
        s = store.script(book_id)
        before = ScriptStore.public(s)["lines"].get(qid)
        fix = s["corrections"].get(qid, {})
        if "speaker" in body or "delivery" in body or "mention" in body:
            fix = s["corrections"].setdefault(qid, {})
        cfg = request.app[CONFIG_KEY]
        spec = request.app[ENGINES_KEY].specs.get(cfg.default_engine)
        tuned = line_tuning(cfg, cfg.default_engine, spec, body)
        for name in TUNING:
            if name in body:
                per_line = s.setdefault(name, {})
                if name in tuned:
                    per_line[qid] = tuned[name]
                else:
                    per_line.pop(qid, None)  # null, or the default: forgotten
        if "speaker" in body:
            sp = body["speaker"]
            if sp and sp not in s["characters"]:
                typed = " ".join(str(sp).split())
                detached = detach_alias(s, typed)  # a new character, not the one this name was filed under
                known = set(s["characters"])
                sp = ensure_character(s, typed, exact=True)  # typed by the reader: kept as typed
                created = sp not in known
                if sp in s["characters"] and s["characters"][sp]["name"] == typed:
                    s["characters"][sp]["name_user"] = True
            fix["speaker"] = sp or None
        if "delivery" in body:
            fix["delivery"] = (body.get("delivery") or "").strip() or None
        if "mention" in body:
            # Not dialogue: the quotation is read in the flow of its sentence, by the narration (no speaker,
            # no delivery); false makes it a line of its own again (the speaker the body gives, if any).
            fix["mention"] = bool(body["mention"])
            if fix["mention"]:
                fix.update(speaker=None, delivery=None)
        if s.get("status") == "done":
            refresh_narration(store, book_id, s)
        store.save_script(book_id, s)
        names = {cid: c["name"] for cid, c in s["characters"].items()}
    if detached:
        from .series import forget_aliases

        forget_aliases(request.app, book_id, detached)
    old = (before or {}).get("analysis") or before or {}
    text = reports.quote_text(store.book(book_id), qid)
    common = {"line": qid, "text": text, "before": before}
    if "speaker" in body and (fix.get("speaker") != (before or {}).get("speaker") or before is None):
        kind = "split_alias" if detached else "new_character" if created else "speaker"
        reports.record(request.app, book_id, kind, **common,
                       after={"speaker": fix.get("speaker"), "name": names.get(fix.get("speaker"), "NARRATOR")},
                       analysis_speaker=names.get(old.get("speaker"), "NARRATOR") if old else None,
                       detached=detached or None)
    if "delivery" in body and fix.get("delivery") != (before or {}).get("delivery"):
        reports.record(request.app, book_id, "delivery", **common, after={"delivery": fix.get("delivery")})
    if "mention" in body and bool(fix.get("mention")) != bool((before or {}).get("mention")):
        reports.record(request.app, book_id, "mention", **common, after={"mention": bool(fix.get("mention"))})
    return web.json_response(ScriptStore.public(s))


async def patch_narration(request: web.Request) -> web.Response:
    """Who narrates a section, from a paragraph to the next point-of-view heading or chapter: {"chapter",
    "paragraph", "narrator": a character id, null (the book's narrator voice) or "auto" (as the analysis has it)}.
    Kept as a correction (narration_fixed) and as a report of incorrect analysis."""
    from . import reports

    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    body = await request.json()
    try:
        ci, pi = int(body["chapter"]), int(body["paragraph"])
    except (KeyError, TypeError, ValueError):
        return web.json_response({"error": "chapter and paragraph are required"}, status=400)
    who = body.get("narrator")
    async with store.lock(book_id):
        s = store.script(book_id)
        book = store.book(book_id)
        chapters = (book or {}).get("chapters", [])
        if not (0 <= ci < len(chapters)) or not (0 <= pi < len(chapters[ci].get("paragraphs", []))):
            return web.json_response({"error": "no such paragraph"}, status=400)
        if who not in (None, "auto") and who not in s["characters"]:
            return web.json_response({"error": "no such character"}, status=404)

        def at(runs):
            found = None
            for c, p, w in runs:
                if (c, p) > (ci, pi):
                    break
                found = w
            return found

        before = at(s.get("narration") or [])
        fixed = s.setdefault("narration_fixed", {})
        if who == "auto":
            fixed.pop(f"{ci}:{pi}", None)
        else:
            fixed[f"{ci}:{pi}"] = who
        s["narration"] = compute_narration(book, s)
        after = at(s["narration"])
        store.save_script(book_id, s)
        names = {cid: c["name"] for cid, c in s["characters"].items()}
    if after != before:
        reports.record(request.app, book_id, "narration", chapter=ci, paragraph=pi,
                       before={"narrator": before, "name": names.get(before, "NARRATOR")},
                       after={"narrator": after, "name": names.get(after, "NARRATOR")})
    return web.json_response(ScriptStore.public(s))


async def patch_character(request: web.Request) -> web.Response:
    """Rename, voice, profile, aliases, role or merge_into; each kept as a report of incorrect analysis."""
    from . import reports

    store = request.app[DRAMA_STORE_KEY]
    book_id, cid = request.match_info["book"], request.match_info["character"]
    body = await request.json()
    async with store.lock(book_id):
        s = store.script(book_id)
        c = s["characters"].get(cid)
        if c is None:
            return web.json_response({"error": "no such character"}, status=404)
        before = reports.character_view(c)
        before_profile = dict(c.get("profile_user") or {})
        changed = []
        if body.get("name") and body["name"] != c["name"]:
            c.setdefault("aliases", []).append(c["name"])
            c["name"] = body["name"]
            c["name_user"] = True  # typed by the reader: never re-cased
            changed.append("name")
        if "voice" in body:
            from .app import VOICES_KEY

            old_voice = c.get("voice")
            c["voice"] = body["voice"] or None
            c["voice_user"] = bool(c["voice"])  # the reader's choice: a series applied later keeps it
            follow_voice(c, request.app[VOICES_KEY])
            if c["voice"] != old_voice:
                changed.append("voice")
        if isinstance(body.get("profile"), dict) or isinstance(body.get("dossier"), dict):
            from . import profile

            fields = body.get("profile") if isinstance(body.get("profile"), dict) else profile.from_dossier(body["dossier"])
            profile.set_user_fields(c, fields)
            if (c.get("profile_user") or {}) != before_profile:
                changed.append("profile")
        if isinstance(body.get("aliases"), list):
            new = [a.strip() for a in body["aliases"] if isinstance(a, str) and a.strip()]
            if new != c.get("aliases", []):
                changed.append("aliases")
            c["aliases"] = new
        if "role" in body:
            from .roles import set_user_role

            old_role = c.get("role")
            set_user_role(c, body["role"])
            if c.get("role") != old_role:
                changed.append("role")
        c["user"] = True
        after = reports.character_view(c)
        target = body.get("merge_into")
        merged_into = None
        if target and target in s["characters"] and target != cid:
            t = s["characters"][target]
            merged_into = t["name"]
            t.setdefault("aliases", []).extend([c["name"], *c.get("aliases", [])])
            for table in (s["lines"], s["corrections"]):
                for line in table.values():
                    if line.get("speaker") == cid:
                        line["speaker"] = target
            del s["characters"][cid]
            if s.get("status") == "done":
                refresh_narration(store, book_id, s)
        store.save_script(book_id, s)
    for kind in changed:
        reports.record(request.app, book_id, kind, character=cid, before=before, after=after,
                       profile=c.get("profile_user") if kind == "profile" else None)
    if merged_into:
        reports.record(request.app, book_id, "merge", character=cid, before=before,
                       after={"merged_into": target, "name": merged_into})
    if "name" in changed or merged_into:  # a voice still named what the character was called before follows
        name_voices(request.app)
    if "merge_into" not in body:
        from .series import propagate_voice, remember

        if "voice" in body:  # the same character in the series' other volumes follows (in books this user may change)
            from .permissions import can_book

            await propagate_voice(request.app, book_id, cid, old_voice, body["voice"] or None,
                                  may=lambda b: can_book(request, b))
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
    from .permissions import can_book
    from .series import propagate_voice

    await propagate_voice(request.app, book_id, cid, old_voice, v.id, may=lambda b: can_book(request, b))
    remember(request.app, book_id, only=cid)
    if old_voice and old_voice != v.id:  # kept as a report, so the old voice is one of their past voices
        from . import reports

        reports.record(request.app, book_id, "voice", character=cid, edit="designed from the profile",
                       before={"voice": old_voice}, after={"voice": v.id})
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
        name_voices(app)  # voices left under a name their character has since lost
        delete_unused_voices(app)
        # Resume analyses interrupted by a restart, in the queue's saved order (the one being analysed first).
        waiting = {}
        for d in store.root.iterdir() if store.root.exists() else []:
            s = store.script(d.name) if (d / "script.json").exists() else None
            if s and s.get("status") in ("queued", "running"):
                waiting[d.name] = s
            elif s and s.get("status") == "casting":  # automatic casting cut short by a restart: carry on
                analyzer._spawn_cast(d.name, s.get("auto_cast_engine") or cfg.voice_design_engine)
        saved = [b for b in analyzer.queue.saved_order() if b in waiting]
        for book_id in saved + [b for b in waiting if b not in saved]:
            s = waiting[book_id]
            s["status"] = "none"
            store.save_script(book_id, s)
            await analyzer.enqueue(book_id, keep_order=book_id in saved)
        yield
        for t in app[CAST_TASKS_KEY]:
            t.cancel()
        await analyzer.stop()

    app.cleanup_ctx.append(_ctx)
    r = app.router
    r.add_get("/api/gpu", gpu_status)
    r.add_get("/api/queue", queue_get)
    r.add_put("/api/queue", queue_put)
    r.add_delete("/api/queue/{book}", queue_delete)
    r.add_post("/api/queue/{book}/now", queue_now)
    r.add_post("/api/queue/{book}/bottom", queue_bottom)
    r.add_get("/api/books", list_books)
    r.add_post("/api/books/{book}/characters/{character}/voice-from-dossier", voice_from_dossier)
    from . import series

    series.setup(app)
    r.add_get("/api/books/{book}", get_book)
    r.add_get("/api/books/{book}/cover", get_cover)
    r.add_get("/api/covers", get_covers)
    r.add_get("/api/books/{book}/pictures", list_pictures)
    r.add_get("/api/books/{book}/picture", get_picture)
    r.add_post("/api/books/{book}", upload_book)
    r.add_patch("/api/books/{book}", rename_book)
    r.add_get("/api/books/{book}/script", get_script)
    r.add_post("/api/books/{book}/analyze", analyze)
    r.add_patch("/api/books/{book}/lines/{line}", patch_line)
    r.add_patch("/api/books/{book}/narration", patch_narration)
    r.add_patch("/api/books/{book}/characters/{character}", patch_character)
    r.add_post("/api/books/{book}/cast", cast)
    from .roles import classify_route

    r.add_post("/api/books/{book}/roles", classify_route)
