"""Reports of incorrect analysis, kept to clean the analysis up later.

Every manual edit is one (the reader fixed a speaker, a delivery, a name...), and the reader can flag a line or a
character as wrong without fixing it ("Incorrect analysis"). Each report keeps what the analysis said, the fix if
there is one, the text and the model, so reports double as test cases. One JSON object per line in
data/books/<id>/reports.jsonl.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time

from aiohttp import web

logger = logging.getLogger("elvenreader.reports")

TYPES = {
    "flag": "Marked as incorrect (not fixed yet)",
    "speaker": "Wrong speaker",
    "delivery": "Wrong delivery",
    "mention": "Not dialogue (a quoted word or term read as part of the narration), or the other way round",
    "new_character": "Speaker missing from the cast",
    "split_alias": "Two people filed as one (a name under the wrong character)",
    "merge": "One person split into several characters",
    "name": "Wrong name",
    "aliases": "Wrong aliases",
    "role": "Wrong tier",
    "profile": "Wrong voice profile",
    "voice": "Wrong voice",
    "narration": "Wrong narrator (who narrates a section)",
}
_lock = threading.Lock()


def _path(app: web.Application, book_id: str):
    from .dramatize import DRAMA_STORE_KEY

    return app[DRAMA_STORE_KEY].dir(book_id) / "reports.jsonl"


def record(app: web.Application, book_id: str, kind: str, **fields) -> dict:
    """Keeps one report. `fields`: line (quote id), character (id), text, before, after, note."""
    from .dramatize import DRAMA_STORE_KEY

    script = app[DRAMA_STORE_KEY].script(book_id)
    report = {"id": secrets.token_hex(5), "t": round(time.time(), 1), "book": book_id, "type": kind,
              "model": script.get("model", ""), **{k: v for k, v in fields.items() if v not in (None, "", {})}}
    try:
        with _lock, open(_path(app, book_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("%s: could not keep a report: %s", book_id, exc)
    return report


def read(app: web.Application, book_id: str) -> list[dict]:
    p = _path(app, book_id)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def quote_text(book: dict | None, qid: str) -> str:
    from .dramatize import _QID_RE

    m = _QID_RE.match(qid or "")
    if not book or not m:
        return ""
    ch = book["chapters"][int(m.group(1))] if int(m.group(1)) < len(book["chapters"]) else None
    for q in (ch or {}).get("quotes", []):
        if q[0] == qid:
            return ch["paragraphs"][q[1]][q[2]:q[3]]
    return ""


def character_view(c: dict | None) -> dict | None:
    if c is None:
        return None
    return {k: c.get(k) for k in ("name", "aliases", "role", "voice") if c.get(k) not in (None, "", [])}


async def book_reports(request: web.Request) -> web.Response:
    """GET /api/books/{book}/reports: the book's reports, oldest first."""
    try:
        return web.json_response({"reports": read(request.app, request.match_info["book"]), "types": TYPES})
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def all_reports(request: web.Request) -> web.Response:
    """GET /api/reports[?type=]: every book's reports, newest first (for cleaning up the analysis)."""
    from .dramatize import DRAMA_STORE_KEY

    kind = request.query.get("type")
    out = [r for b in request.app[DRAMA_STORE_KEY].ids() for r in read(request.app, b)
           if not kind or r.get("type") == kind]
    out.sort(key=lambda r: -r.get("t", 0))
    return web.json_response({"reports": out, "types": TYPES})


async def flag(request: web.Request) -> web.Response:
    """POST /api/books/{book}/reports {line? | character?, note?}: "Incorrect analysis" without a fix."""
    from .dramatize import DRAMA_STORE_KEY, ScriptStore

    book_id = request.match_info["book"]
    store = request.app[DRAMA_STORE_KEY]
    try:
        book = store.book(book_id)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if book is None:
        return web.json_response({"error": "no such book"}, status=404)
    body = await request.json() if request.can_read_body else {}
    body = body if isinstance(body, dict) else {}
    qid, cid = str(body.get("line") or ""), str(body.get("character") or "")
    if not qid and not cid:
        return web.json_response({"error": "Say which line or character is wrong"}, status=400)
    public = ScriptStore.public(store.script(book_id))
    fields: dict = {"note": str(body.get("note") or "").strip()[:1000]}
    if qid:
        line = public["lines"].get(qid)
        if line is None and not quote_text(book, qid):
            return web.json_response({"error": "no such line"}, status=404)
        fields.update(line=qid, text=quote_text(book, qid), before=line)
    if cid:
        c = next((c for c in public["characters"] if c["id"] == cid), None)
        if c is None:
            return web.json_response({"error": "no such character"}, status=404)
        fields.update(character=cid, before=fields.get("before") or character_view(c))
    return web.json_response(record(request.app, book_id, "flag", **fields), status=201)


def setup(app: web.Application) -> None:
    app.router.add_get("/api/reports", all_reports)
    app.router.add_get("/api/books/{book}/reports", book_reports)
    app.router.add_post("/api/books/{book}/reports", flag)
