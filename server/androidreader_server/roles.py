"""Character tiers: main, supporting or extra, per character (`role`).

The LLM sorts the cast from per-character evidence (lines, chapters with lines or mentions, first and
last appearance, named or not, narrating); a few rules back it up. The reader's choice (`role_user`)
always wins, and the series remembers tiers: a character never drops below the tier the series has
for them unless the reader changes it.
"""
from __future__ import annotations

import logging
import re
import time

from aiohttp import web

logger = logging.getLogger("androidreader.drama")

ROLES = ("main", "supporting", "extra")
RANK = {r: i for i, r in enumerate(reversed(ROLES))}  # extra 0, supporting 1, main 2
MAX_LLM_CHARACTERS = 80

ROLES_PROMPT = """You sort the characters of a novel into tiers for casting a multi-voice audiobook:
- main: the protagonists and the core recurring cast who drive the book: usually a small group (often two to six), the characters the story is about. A character who narrates the book (or part of it) is main.
- supporting: recurring, named characters with a real presence in the story, but not at its centre: they come back across several chapters.
- extra: walk-on characters: anyone unnamed (e.g. "a prisoner", "a soldier", "the driver"), and named characters who only appear in one chapter or scene.
Use the evidence for each character: how many lines they speak and what share of the book's dialogue that is, in how many chapters they speak or are mentioned, where they first and last appear, whether they have a name, and the description. If you know the book or its series, use that too. Return every character exactly once, with a short reason."""


def is_placeholder(name: str) -> bool:
    from .dramatize import is_placeholder as _p

    return _p(name)


def evidence(book: dict, script: dict) -> dict[str, dict]:
    """Per character: lines, share of all lines, chapters with lines, chapters mentioned, first/last
    chapter with a line, named, narrating."""
    from .dramatize import ScriptStore, find_character

    public = ScriptStore.public(script)
    total = max(1, sum(1 for line in public["lines"].values() if line.get("speaker")))
    spoke: dict[str, set[int]] = {}
    for qid, line in public["lines"].items():
        m = re.match(r"c(\d+)p", qid)
        if line.get("speaker") and m:
            spoke.setdefault(line["speaker"], set()).add(int(m.group(1)))
    narrating = {who for _, _, who in script.get("narration", []) if who}
    narrating |= {find_character(script, n) for n in (script.get("narrator_by_tense") or {}).values() if n}
    chapters = [" ".join(ch.get("paragraphs", [])) for ch in book.get("chapters", [])]
    out = {}
    for c in public["characters"]:
        names = [n for n in [c["name"], *c.get("aliases", [])] if n and not is_placeholder(n) and len(n) > 2]
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b") if names else None
        mentioned = {i for i, text in enumerate(chapters) if pattern and pattern.search(text)}
        with_lines = spoke.get(c["id"], set())
        out[c["id"]] = {
            "name": c["name"], "lines": c["lines"], "share": round(c["lines"] / total, 4),
            "chapters_with_lines": len(with_lines), "chapters_mentioned": len(mentioned | with_lines),
            "first": min(with_lines | mentioned) + 1 if with_lines | mentioned else None,
            "last": max(with_lines | mentioned) + 1 if with_lines | mentioned else None,
            "named": not is_placeholder(c["name"]), "narrates": c["id"] in narrating,
            "chapters": len(chapters),
        }
    return out


def rule_role(ev: dict) -> str:
    """Thresholds, used when there is no LLM answer and as a sanity check on it."""
    if ev["narrates"] or ev["share"] >= 0.08:
        return "main"
    if not ev["named"]:
        return "extra"
    if ev["share"] >= 0.01 or (ev["lines"] >= 5 and ev["chapters_with_lines"] >= 2):
        return "supporting"
    return "extra"


def settle(c: dict, ev: dict, proposed: str | None, reason: str = "", source: str = "llm") -> None:
    """Sets a character's automatic tier from the model's (or the rules') proposal, applying the guard
    rails, the series floor and the reader's override."""
    rule = rule_role(ev)
    role = proposed if proposed in ROLES else rule
    notes = [reason.strip()] if reason and reason.strip() else []
    if ev["narrates"] and role != "main":
        role, notes = "main", notes + ["narrates"]
    if not ev["named"] and role != "extra" and ev["share"] < 0.05 and not ev["narrates"]:
        role, notes = "extra", notes + ["unnamed"]
    if ev["share"] >= 0.05 and role == "extra":
        role, notes = "supporting", notes + [f"speaks {ev['share']:.0%} of the lines"]
    floor = c.get("role_series")
    if floor in ROLES and RANK[floor] > RANK[role]:
        role, notes = floor, notes + [f"{floor} in the series"]
    c["role_auto"] = role
    c["role_source"] = source if proposed in ROLES else "rule"
    c["role_note"] = "; ".join(notes)[:200]
    if proposed in ROLES and abs(RANK[proposed] - RANK[rule]) >= 2:
        c["role_flag"] = f"the line counts suggest {rule}"
    else:
        c.pop("role_flag", None)
    c["role"] = c.get("role_user") if c.get("role_user") in ROLES else role


def apply_rules(book: dict, script: dict) -> None:
    """Rule tiers for every character (no LLM); the reader's and the series' tiers still win."""
    for cid, ev in evidence(book, script).items():
        settle(script["characters"][cid], ev, None)


def _evidence_text(c: dict, ev: dict) -> str:
    where = (f"chapters {ev['first']}-{ev['last']} of {ev['chapters']}" if ev["first"] else "nowhere by name")
    bits = [f"{ev['lines']} lines ({ev['share']:.1%} of the dialogue)", f"speaks in {ev['chapters_with_lines']} chapters",
            f"speaks or is mentioned in {ev['chapters_mentioned']} chapters ({where})"]
    if not ev["named"]:
        bits.append("unnamed")
    if ev["narrates"]:
        bits.append("narrates")
    if c.get("role_series"):
        bits.append(f"{c['role_series']} in earlier volumes")
    about = (c.get("description") or c.get("appearance") or "")[:140]
    aliases = [a for a in c.get("aliases", []) if a != c["name"]][:4]
    return (f"- {c['name']}" + (f" (also {', '.join(aliases)})" if aliases else "") + ": " + "; ".join(bits)
            + (f". {about}" if about else ""))


async def classify(llm, session, book: dict, script: dict) -> int:
    """The LLM sorts the cast (most-heard first, up to MAX_LLM_CHARACTERS); everyone else gets the rules."""
    ev = evidence(book, script)
    ranked = sorted(ev, key=lambda cid: -ev[cid]["lines"])
    asked = ranked[:MAX_LLM_CHARACTERS]
    for cid in ranked[MAX_LLM_CHARACTERS:]:
        settle(script["characters"][cid], ev[cid], None)
    if not asked:
        return 0
    names = [script["characters"][cid]["name"] for cid in asked]
    schema = {"type": "object", "properties": {"characters": {
        "type": "array", "minItems": len(asked), "maxItems": len(asked),
        "items": {"type": "object", "properties": {
            "name": {"type": "string", "enum": names},
            "role": {"type": "string", "enum": list(ROLES)},
            "reason": {"type": "string", "maxLength": 120}},
            "required": ["name", "role", "reason"]}}}, "required": ["characters"]}
    user = (f"Book: {book.get('title', '')} by {book.get('author', '')}\n\nCharacters:\n"
            + "\n".join(_evidence_text(script["characters"][cid], ev[cid]) for cid in asked))
    result = await llm.complete(session, ROLES_PROMPT, user, schema, max_tokens=512 + 60 * len(asked))
    answers = {}
    for a in result.get("characters", []):
        answers.setdefault(a.get("name"), a)
    for cid in asked:
        c = script["characters"][cid]
        a = answers.get(c["name"]) or {}
        settle(c, ev[cid], a.get("role"), a.get("reason", ""))
    script["roles_state"] = {"done": True, "at": time.time(), "model": getattr(llm, "model_name", "")}
    return len(answers)


def set_user_role(c: dict, role: str | None) -> None:
    if role in ROLES:
        c["role_user"] = c["role"] = role
    else:
        c.pop("role_user", None)
        c["role"] = c.get("role_auto") or c.get("role")


# ----------------------------------------------------------------- route


async def classify_route(request: web.Request) -> web.Response:
    """POST /api/books/{book}/roles: sort an analysed book's cast into tiers (no re-labelling)."""
    import asyncio

    from .dramatize import ANALYZER_KEY, CAST_TASKS_KEY, DRAMA_STORE_KEY, ScriptStore

    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    book = store.book(book_id)
    if book is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    async with store.lock(book_id):
        s = store.script(book_id)
        if s.get("status") in ("queued", "running", "casting") or (s.get("roles_state") or {}).get("running"):
            return web.json_response({"error": "Busy; try again when it is done"}, status=409)
        s["roles_state"] = {"running": True}
        store.save_script(book_id, s)
    analyzer = request.app[ANALYZER_KEY]

    async def run():
        import aiohttp

        from .series import remember

        try:
            async with aiohttp.ClientSession() as session:
                work = store.script(book_id)
                await analyzer._gpu(lambda: classify(analyzer.llm, session, book, work))
            async with store.lock(book_id):
                fresh = store.script(book_id)
                for cid, c in work["characters"].items():
                    if cid in fresh["characters"]:
                        f = fresh["characters"][cid]
                        for k in ("role_auto", "role_source", "role_note"):
                            f[k] = c.get(k)
                        f.pop("role_flag", None)
                        if c.get("role_flag"):
                            f["role_flag"] = c["role_flag"]
                        f["role"] = f.get("role_user") if f.get("role_user") in ROLES else c.get("role_auto")
                fresh["roles_state"] = work.get("roles_state") or {"done": True}
                store.save_script(book_id, fresh)
            remember(request.app, book_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("sorting the cast of %s failed", book_id)
            async with store.lock(book_id):
                fresh = store.script(book_id)
                fresh["roles_state"] = {"error": str(exc)[:200]}
                store.save_script(book_id, fresh)
        finally:
            if analyzer.queue.empty() and analyzer.current is None:
                await analyzer.llm.stop()

    task = asyncio.ensure_future(run())
    request.app[CAST_TASKS_KEY].add(task)
    task.add_done_callback(request.app[CAST_TASKS_KEY].discard)
    return web.json_response(ScriptStore.public(store.script(book_id)), status=202)
