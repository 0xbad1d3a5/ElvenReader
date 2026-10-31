"""Character dossier: a first read of the whole book by a reasoning model.

Pass 1 reads the book in order, a passage at a time, with the dossier so far as
context, and records for every character: names and aliases, gender, apparent
age and physical form (what a listener should hear), true age or nature when it
differs, personality, speech style and verbal tics, relationships, how they are
referred to before the book names them, whether they narrate, and the evidence
(short exact quotes) for each fact. Later passages may revise earlier beliefs.
It also records who is present in each scene and who "I" is.

Pass 2 (dramatize.label_chunk) then attributes lines using the final dossier, so
early chapters benefit from later reveals. Voices are designed from the dossier.

User edits to a dossier (``dossier_user``) always win over the model's.
"""
from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger("elvenreader.dossier")

FIELDS = ["gender", "apparent_age", "form", "true_nature", "personality", "speech_style", "relationships",
          "introduced_as", "voice"]
MAX_FACTS = 10
_SKIP_TITLES = re.compile(r"copyright|contents|newsletter|afterword|acknowledg|about the author|table of", re.I)

DOSSIER_PROMPT = """You are reading a novel in order, one passage at a time, to build a character dossier for a multi-voice audiobook. You get the dossier so far and the next passage.

Return updates for every character who appears, speaks or is described in this passage and about whom the passage says something new or different. For each give their main name (use the dossier's name for known characters) and only the fields you can now fill in or must correct:
- aliases: other names, titles, nicknames and descriptions used for them ("Makeup", "the Japanese guy", "Grandmother").
- gender.
- apparent_age: how old they LOOK and SOUND to the people around them, which is what a listener should hear: a centuries-old being in a little girl's body is "child"; a schoolgirl is "teen"; an animal or creature by how its voice comes across in the story (a wise old sage reborn as a bird sounds like an elderly man). Leave it out when the passage gives no cue about their looks or age; never guess "adult" by default.
- form: their physical form in a few words ("human", "Java sparrow", "android girl", "demon").
- true_nature: true age or nature when it differs from how they look ("centuries old", "a reincarnated sage").
- personality: 3-8 words.
- speech_style: how they talk and their verbal tics ("calls everyone 'dearie', archaic and teasing", "terse military speech", "formal, refers to herself in the third person").
- relationships: to the main characters, briefly.
- introduced_as: only if THIS character appears in the story under a description before the book reveals their name, that description ("an Asian prisoner in the yard", "the man at the door"). Never use another character's description.
- facts: new evidence, each a short fact with an exact short quote (under 120 characters) from this passage.
- revised: if this passage contradicts or clarifies something in the dossier, say what changed and why.

Also give:
- narrator: who "I" is in this passage (their name), or "third person".
- present: everyone present in this passage's scenes, by dossier name.

Do not invent: leave a field out unless the book supports it. Keep names consistent with the dossier."""

DOSSIER_SCHEMA = {
    "type": "object",
    "properties": {
        "narrator": {"type": "string", "maxLength": 60},
        "present": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
        "characters": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "apparent_age": {"type": "string",
                                     "enum": ["child", "teen", "young adult", "adult", "middle-aged", "elderly",
                                              "unknown"]},
                    "form": {"type": "string"},
                    "true_nature": {"type": "string"},
                    "personality": {"type": "string"},
                    "speech_style": {"type": "string"},
                    "relationships": {"type": "string"},
                    "introduced_as": {"type": "string"},
                    "facts": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {"type": "object",
                                  "properties": {"fact": {"type": "string"}, "quote": {"type": "string"}},
                                  "required": ["fact", "quote"]},
                    },
                    "revised": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["narrator", "present", "characters"],
}


def effective(c: dict) -> dict:
    """The dossier as used: the model's, with the user's edits on top."""
    return {**(c.get("dossier") or {}), **(c.get("dossier_user") or {})}


def dossier_chunks(book: dict, max_words: int = 1800) -> list[tuple[str, int, list[int]]]:
    """(key, chapter, paragraphs) passages covering the story, front and back matter left out."""
    out = []
    for ci, ch in enumerate(book.get("chapters", [])):
        paras = ch.get("paragraphs", [])
        if _SKIP_TITLES.search(ch.get("title", "")) or sum(len(p.split()) for p in paras) < 15:
            continue
        current, words, k = [], 0, 0
        for pi, text in enumerate(paras):
            current.append(pi)
            words += len(text.split())
            if words >= max_words:
                out.append((f"d{ci}k{k}", ci, current))
                current, words, k = [], 0, k + 1
        if current:
            out.append((f"d{ci}k{k}", ci, current))
    return out


def render_dossier(script: dict, limit: int = 40, width: int = 140) -> str:
    """The dossier as compact text for prompts, most important characters first."""
    counts: dict[str, int] = {}
    for line in script.get("lines", {}).values():
        if line.get("speaker"):
            counts[line["speaker"]] = counts.get(line["speaker"], 0) + 1
    mentions = (script.get("dossier_state") or {}).get("mentions", {})
    ranked = sorted(script["characters"].items(),
                    key=lambda kv: -(counts.get(kv[0], 0) + 3 * mentions.get(kv[0], 0)))[:limit]
    parts = []
    for cid, c in ranked:
        d = effective(c)
        if not d and not c.get("aliases"):
            continue
        head = c["name"] + (f" (also: {', '.join(c.get('aliases', [])[:6])})" if c.get("aliases") else "")
        bits = [x for x in (d.get("gender"), d.get("apparent_age"), d.get("form")) if x and x != "unknown"]
        lines = [f"- {head}: " + ", ".join(bits)]
        for key, label in (("true_nature", "really"), ("personality", "personality"), ("speech_style", "speech"),
                           ("relationships", "relations"), ("introduced_as", "first appears as"),
                           ("narrates", "narrates")):
            if d.get(key):
                lines.append(f"  {label}: {str(d[key])[:width]}")
        parts.append("\n".join(lines))
    return "\n".join(parts) or "(empty: this is the start of the book)"


def _clean(value):
    """Collapse odd whitespace (models like to emit narrow no-break spaces inside names)."""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def merge_update(script: dict, result: dict, where: tuple[int, int], tense: str | None) -> list[str]:
    """Fold one passage's findings into the dossier; returns the ids of characters present."""
    from .dramatize import ensure_character, find_character, resolve_speaker

    result = _clean({k: v for k, v in result.items() if not k.startswith("_")})
    state = script.setdefault("dossier_state", {})
    mentions = state.setdefault("mentions", {})
    for upd in result.get("characters", []):
        name = (upd.get("name") or "").strip()
        if not name or name.lower() in ("i", "narrator", "the narrator"):
            continue
        aliases = [a for a in upd.get("aliases") or [] if isinstance(a, str)]
        cid = ensure_character(script, name, upd.get("gender", "unknown"), upd.get("apparent_age", "unknown"),
                               aliases)
        if cid is None:
            continue
        c = script["characters"][cid]
        d = c.setdefault("dossier", {})
        for key in FIELDS:
            value = upd.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
                d[key] = value.strip()[:300]
        for f in upd.get("facts") or []:
            if isinstance(f, dict) and f.get("fact"):
                d.setdefault("facts", []).append(
                    {"fact": str(f["fact"])[:200], "quote": str(f.get("quote", ""))[:160], "at": list(where)})
        d["facts"] = d.get("facts", [])[-MAX_FACTS:]
        if isinstance(upd.get("revised"), str) and upd["revised"].strip():
            d.setdefault("revisions", []).append({"note": upd["revised"].strip()[:200], "at": list(where)})
            d["revisions"] = d["revisions"][-5:]
        sync_character(c)
        mentions[cid] = mentions.get(cid, 0) + 1
    narrator = (result.get("narrator") or "").strip()
    if narrator and narrator.lower() not in ("third person", "unknown", "none", "narrator", "i"):
        cid = ensure_character(script, narrator)
        if cid:
            key = tense or "any"
            votes = state.setdefault("narrator_votes", {}).setdefault(key, {})
            votes[cid] = votes.get(cid, 0) + 1
    present = []
    for name in result.get("present") or []:
        cid = find_character(script, name) if isinstance(name, str) else None
        if cid is None and isinstance(name, str) and name.strip():
            cid = resolve_speaker(script, name)
        if cid and cid not in present:
            present.append(cid)
    return present


def sync_character(c: dict) -> None:
    """Apparent gender and age from the dossier drive voice choice (pool voices, casting)."""
    d = effective(c)
    if d.get("gender") in ("male", "female"):
        c["gender"] = d["gender"]
    if d.get("apparent_age") and d["apparent_age"] != "unknown":
        c["age"] = d["apparent_age"]


def narrators(script: dict) -> dict[str, str]:
    """{tense or "any": character id} by majority over the passages read."""
    votes = (script.get("dossier_state") or {}).get("narrator_votes", {})
    out = {}
    for key, v in votes.items():
        if v:
            out[key] = max(v, key=v.get)
    return out


def scene_cast(script: dict, chapter: int, paragraphs: list[int]) -> list[str]:
    """Characters present in the passages of pass 1 overlapping these paragraphs."""
    lo, hi = min(paragraphs), max(paragraphs)
    out = []
    for c, p0, p1, cids in (script.get("dossier_state") or {}).get("scenes", []):
        if c == chapter and p0 <= hi and p1 >= lo:
            out += [x for x in cids if x not in out and x in script["characters"]]
    return out


def voice_description(c: dict) -> str:
    """A voice-design description from the dossier: apparent age and form, manner from personality and speech."""
    d = effective(c)
    if d.get("voice"):
        return d["voice"]
    who = " ".join(x for x in (d.get("apparent_age"), d.get("gender")) if x and x != "unknown") or "adult"
    form = d.get("form", "")
    lead = f"A {who} voice" + (f" ({form})" if form and form.lower() not in ("human", "person") else "")
    manner = "; ".join(x for x in (d.get("personality"), d.get("speech_style")) if x)
    return lead + (f", speaking in a {manner} manner" if manner else "") + "."


CONSOLIDATE_PROMPT = """You are given a character dossier built while reading a novel passage by passage. Clean it up for casting an audiobook:
- merges: the same person often appears under several entries (nicknames like "Makeup", titles, cover names, a fake identity, descriptions such as "Japanese Prisoner" or "the girl"). Group them: "keep" is the entry with the proper name, "same_as" the other entries' names exactly as listed. Merge only when the dossier (relationships, facts, aliases) or what you know about this book and its series supports it; say why.
- drop: entries that are not one character (groups, crowds, "unknown male", narrator placeholders like "first-person I").
- fixes: for each main character whose gender, apparent age or form is missing or doubtful, give them. Apparent age is how old they look and sound to people around them (what a listener should hear): a centuries-old being in a little girl's body is "child", a middle-school girl is "teen", a wise sage reborn as a bird sounds "elderly"."""

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {"type": "array", "items": {"type": "object", "properties": {
            "keep": {"type": "string"}, "same_as": {"type": "array", "items": {"type": "string"}},
            "why": {"type": "string"}}, "required": ["keep", "same_as", "why"]}},
        "drop": {"type": "array", "items": {"type": "string"}},
        "fixes": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
            "apparent_age": {"type": "string", "enum": ["child", "teen", "young adult", "adult", "middle-aged",
                                                        "elderly", "unknown"]},
            "form": {"type": "string"}, "true_nature": {"type": "string"}}, "required": ["name"]}},
    },
    "required": ["merges", "drop", "fixes"],
}


def merge_characters(script: dict, keep: str, other: str) -> bool:
    """Fold character `other` into `keep` (aliases, dossier, lines, scenes, votes). Never folds a
    character the user has edited into another."""
    chars = script["characters"]
    if keep == other or keep not in chars or other not in chars or chars[other].get("user"):
        return False
    k, o = chars[keep], chars[other]
    names = [o["name"], *o.get("aliases", [])]
    k["aliases"] = list(dict.fromkeys([*k.get("aliases", []), *(n for n in names if n != k["name"])]))
    kd, od = k.setdefault("dossier", {}), o.get("dossier") or {}
    for key, value in od.items():
        if key == "facts" or key == "revisions":
            kd[key] = (kd.get(key, []) + value)[-MAX_FACTS:]
        elif not kd.get(key):
            kd[key] = value
    if not k.get("voice") and o.get("voice"):
        k["voice"] = o["voice"]
    for table in (script.get("lines", {}), script.get("corrections", {})):
        for line in table.values():
            if line.get("speaker") == other:
                line["speaker"] = keep
    state = script.get("dossier_state") or {}
    m = state.get("mentions", {})
    if other in m:
        m[keep] = m.get(keep, 0) + m.pop(other)
    for scene in state.get("scenes", []):
        scene[3] = list(dict.fromkeys(keep if x == other else x for x in scene[3]))
    for votes in state.get("narrator_votes", {}).values():
        if other in votes:
            votes[keep] = votes.get(keep, 0) + votes.pop(other)
    del chars[other]
    sync_character(k)
    return True


async def consolidate(llm, session, book: dict, script: dict, model: str, gpu_step) -> dict:
    """One reasoning call over the whole dossier: merge duplicates, drop non-characters, fill gaps."""
    from .dramatize import find_character

    user = (f"Book: {book.get('title', '')} by {book.get('author', '')}\n\n"
            f"Dossier:\n{render_dossier(script, limit=80, width=160)}")
    result = await gpu_step(lambda: llm.complete(session, CONSOLIDATE_PROMPT, user, CONSOLIDATE_SCHEMA,
                                                 max_tokens=16000, model=model, thinking=True,
                                                 effort=llm.cfg.llm_dossier_effort))
    result = _clean({k: v for k, v in result.items() if not k.startswith("_")})
    done = {"merged": [], "dropped": [], "fixed": []}
    for m in result.get("merges", []):
        keep = find_character(script, m.get("keep", ""))
        for name in m.get("same_as", []):
            other = find_character(script, name)
            if keep and other and merge_characters(script, keep, other):
                done["merged"].append(f"{name} -> {script['characters'][keep]['name']}")
    for name in result.get("drop", []):
        cid = find_character(script, name)
        c = script["characters"].get(cid or "")
        if c and not c.get("user") and not c.get("voice") and not any(
                ln.get("speaker") == cid for ln in script.get("corrections", {}).values()):
            del script["characters"][cid]
            done["dropped"].append(name)
    for f in result.get("fixes", []):
        cid = find_character(script, f.get("name", ""))
        if cid is None:
            continue
        c = script["characters"][cid]
        d = c.setdefault("dossier", {})
        for key in ("gender", "apparent_age", "form", "true_nature"):
            if f.get(key) and f[key] != "unknown":
                d[key] = f[key]
        sync_character(c)
        done["fixed"].append(c["name"])
    logger.info("dossier consolidated: %s", done)
    return done


async def build(llm, session, book: dict, store, book_id: str, model: str, gpu_step, progress) -> None:
    """Pass 1 over the whole book (resumable). `gpu_step(make_coro)` runs one LLM step."""
    from .dramatize import LlmError, mixed_tense_runs

    runs = mixed_tense_runs(book) or []
    switches = sorted(((c, p), t) for c, p, t in runs)

    def tense_at(ci: int, pi: int) -> str | None:
        t = None
        for (c, p), tense in switches:
            if (c, p) <= (ci, pi):
                t = tense
        return t

    chunks = dossier_chunks(book, llm.cfg.llm_dossier_words)
    t0 = time.perf_counter()
    for i, (key, ci, paras) in enumerate(chunks):
        s = store.script(book_id)
        state = s.setdefault("dossier_state", {})
        if key in state.get("chunks_done", []):
            continue
        await progress(s, i / max(1, len(chunks)), f"reading the book for the cast: chapter {ci + 1}")
        chapter = book["chapters"][ci]
        text = "\n\n".join(chapter["paragraphs"][p] for p in paras)
        tense = tense_at(ci, paras[0])
        user = (f"Book: {book.get('title', '')} by {book.get('author', '')}\n"
                f"Chapter: {chapter.get('title', '')}\n\n"
                f"Dossier so far:\n{render_dossier(s)}\n\n"
                f"Passage:\n{text}")
        try:
            result = await gpu_step(lambda: llm.complete(session, DOSSIER_PROMPT, user, DOSSIER_SCHEMA,
                                                         max_tokens=14000, model=model, thinking=True,
                                                         effort=llm.cfg.llm_dossier_effort))
        except LlmError as exc:
            if "invalid JSON" not in str(exc):
                raise
            try:  # the reasoning ran out of room: think less
                result = await gpu_step(lambda: llm.complete(session, DOSSIER_PROMPT, user, DOSSIER_SCHEMA,
                                                             max_tokens=16000, model=model, thinking=True,
                                                             effort="low"))
            except LlmError as exc2:
                if "invalid JSON" not in str(exc2):
                    raise
                logger.warning("dossier: skipped passage %s (no usable answer)", key)
                result = {"characters": [], "present": [], "narrator": ""}
        async with store.lock(book_id):
            s = store.script(book_id)
            state = s.setdefault("dossier_state", {})
            present = merge_update(s, result, (ci, paras[0]), tense)
            state.setdefault("scenes", []).append([ci, paras[0], paras[-1], present])
            state.setdefault("chunks_done", []).append(key)
            state["model"] = model
            state["tokens"] = state.get("tokens", 0) + int(result.get("_usage", {}).get("total_tokens", 0))
            state["seconds"] = round(state.get("seconds", 0) + time.perf_counter() - t0, 1)
            t0 = time.perf_counter()
            store.save_script(book_id, s)
    s = store.script(book_id)
    if not s.get("dossier_state", {}).get("consolidated"):
        await progress(s, 0.99, "reading the book for the cast: putting the cast together")
        snapshot = store.script(book_id)
        try:
            changes = await consolidate(llm, session, book, snapshot, model, gpu_step)
        except LlmError as exc:
            if "invalid JSON" not in str(exc):
                raise
            changes = {"error": str(exc)[:200]}
        async with store.lock(book_id):
            # pass 1 has no line labels yet, so the snapshot can replace the stored script
            snapshot["dossier_state"]["consolidated"] = changes
            store.save_script(book_id, snapshot)
    async with store.lock(book_id):
        s = store.script(book_id)
        s.setdefault("dossier_state", {})["done"] = True
        store.save_script(book_id, s)
