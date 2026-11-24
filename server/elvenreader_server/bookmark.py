"""Where each reader is in each book: one bookmark per book and user (profiles.py: the user a request's token belongs
to), kept on the server so each person carries on from their own place on any of their devices. The website moves it as you listen (the passage
playing) or read (the paragraph at the top of the screen when you stop scrolling), and opens the book there.

A place is a chapter, a paragraph and a character offset in it: the book's text never changes, so it holds when the
book is analysed again (quotation ids don't). data/books/<id>/bookmarks.json ({profile id: place}); also in the
book's .elvenbook (x-server/bookmarks.json), so an exported book keeps them.
"""
from __future__ import annotations

import json
import threading
import time

from aiohttp import web

FILE_NAME = "bookmarks.json"
_lock = threading.Lock()


def path(app: web.Application, book_id: str):
    from .dramatize import DRAMA_STORE_KEY

    return app[DRAMA_STORE_KEY].dir(book_id) / FILE_NAME


def read_all(app: web.Application, book_id: str) -> dict:
    try:
        marks = json.loads(path(app, book_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {pid: bm for pid, bm in marks.items() if isinstance(bm, dict) and isinstance(bm.get("chapter"), int)} \
        if isinstance(marks, dict) else {}


def _write(app: web.Application, book_id: str, marks: dict) -> None:
    p = path(app, book_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(marks), encoding="utf-8")
    tmp.replace(p)  # the book file takes it in at its next sync (bookfile._inputs watches this file)


def merge(app: web.Application, book_id: str, marks: dict) -> None:
    """Bookmarks from an imported book file: each profile keeps the newer of its two."""
    with _lock:
        mine = read_all(app, book_id)
        for pid, bm in marks.items():
            if isinstance(bm, dict) and bm.get("updated", 0) > mine.get(pid, {}).get("updated", 0):
                mine[pid] = bm
        _write(app, book_id, mine)


def forget_profile(app: web.Application, pid: str) -> None:
    from .dramatize import DRAMA_STORE_KEY

    for book_id in app[DRAMA_STORE_KEY].ids():
        with _lock:
            marks = read_all(app, book_id)
            if pid in marks:
                del marks[pid]
                _write(app, book_id, marks)


def percent(book: dict, bm: dict) -> float:
    """How far into the book the bookmark is, by characters (0-100)."""
    total = before = 0
    for ci, ch in enumerate(book.get("chapters", [])):
        for pi, text in enumerate(ch.get("paragraphs", [])):
            if (ci, pi) < (bm["chapter"], bm["paragraph"]):
                before += len(text)
            elif (ci, pi) == (bm["chapter"], bm["paragraph"]):
                before += min(len(text), bm.get("offset", 0))
            total += len(text)
    return round(100 * before / total, 1) if total else 0.0


def summary(app: web.Application, book_id: str, book: dict | None, pid: str | None) -> dict | None:
    """For the library: {chapter, title, percent, updated} of this profile's bookmark."""
    bm = read_all(app, book_id).get(pid) if pid else None
    if bm is None or book is None or not 0 <= bm["chapter"] < len(book.get("chapters", [])):
        return None
    return {"chapter": bm["chapter"], "title": book["chapters"][bm["chapter"]].get("title", ""),
            "percent": percent(book, bm), "updated": bm.get("updated")}


def _book(request: web.Request):
    from .dramatize import DRAMA_STORE_KEY

    try:
        book = request.app[DRAMA_STORE_KEY].book(request.match_info["book"])
    except ValueError as exc:
        return None, web.json_response({"error": str(exc)}, status=400)
    if book is None:
        return None, web.json_response({"error": "no such book"}, status=404)
    return book, None


async def get_bookmark(request: web.Request) -> web.Response:
    """This user's place in the book ({} if none yet)."""
    from .profiles import current

    book, err = _book(request)
    if err:
        return err
    pid = current(request)
    if pid is None:
        return web.json_response({"error": "no user for this token"}, status=400)
    bm = read_all(request.app, request.match_info["book"]).get(pid)
    return web.json_response({**bm, "percent": percent(book, bm)} if bm else {})


async def put_bookmark(request: web.Request) -> web.Response:
    """{chapter, paragraph, offset?}: this user is here now (the latest one wins, from any of their devices)."""
    from .profiles import current

    book, err = _book(request)
    if err:
        return err
    pid = current(request)
    if pid is None:
        return web.json_response({"error": "no user for this token"}, status=400)
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"error": "expected JSON"}, status=400)
    chapters = book.get("chapters", [])
    ci, pi, offset = body.get("chapter"), body.get("paragraph", 0), body.get("offset", 0)
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (ci, pi, offset)) \
            or not 0 <= ci < len(chapters) or not 0 <= pi < max(1, len(chapters[ci].get("paragraphs", []))) \
            or offset < 0:
        return web.json_response({"error": "chapter, paragraph and offset must be a place in the book"}, status=400)
    bm = {"chapter": ci, "paragraph": pi, "offset": offset, "updated": time.time()}
    with _lock:
        marks = read_all(request.app, request.match_info["book"])
        marks[pid] = bm
        _write(request.app, request.match_info["book"], marks)
    return web.json_response({**bm, "percent": percent(book, bm)})


def setup(app: web.Application) -> None:
    app.router.add_get("/api/books/{book}/bookmark", get_bookmark)
    app.router.add_put("/api/books/{book}/bookmark", put_bookmark)
