"""Who may change what.

Every user reads the whole library. Each user changes only the books they added (a book's `users` in its meta.json):
their analysis, cast, voices, pronunciations, series. The same EPUB added by two people is one book that both may
change (its analysis and voices are shared). Books from before users are the owner's.

The owner (profiles.py: the server's own token) is the admin and may change anything. Other users change only their
books, the voices only their books' characters use, and pronunciations of their books (and of a series whose books
are all theirs). Site-wide things (settings, the Voices page, series names, the analysis queue's order, users) are the
admin's. Anyone may read, hear speech, keep their own bookmarks and report wrong analysis. `check` runs for every
request (app.auth_middleware); handlers that need the request body (pronunciations, a voice copy, design options)
check themselves.
"""
from __future__ import annotations

import re

from aiohttp import web

_BOOK = re.compile(r"^/api/books/([A-Za-z0-9_-]+)(/.*)?$")
_VOICE = re.compile(r"^/api/voices/([A-Za-z0-9_-]+)(/.*)?$")
_QUEUE_BOOK = re.compile(r"^/api/queue/([A-Za-z0-9_-]+)(/.*)?$")
# Changes anyone may make: hearing speech, their own bookmarks and reports, what a device saw.
_ANYONE = {"/api/tts", "/api/client-log", "/api/activity", "/api/transcribe", "/api/library/import"}
_READ = ("GET", "HEAD", "OPTIONS")


def _user(request: web.Request) -> dict | None:
    from .profiles import USER_KEY

    return request.get(USER_KEY)


def is_admin(request: web.Request) -> bool:
    user = _user(request)
    return bool(user and user.get("owner"))


def book_users(app: web.Application, book_id: str) -> list[str]:
    """The users whose library has the book (the owner's, for a book from before users)."""
    from .dramatize import DRAMA_STORE_KEY
    from .profiles import all_users

    try:
        meta = app[DRAMA_STORE_KEY].summary(book_id) or {}
    except ValueError:
        return []
    return meta.get("users") or [u["id"] for u in all_users(app) if u.get("owner")]


def can_book(request: web.Request, book_id: str) -> bool:
    """Whether the request's user may change the book: they added it (or they're the admin)."""
    if is_admin(request):
        return True
    user = _user(request)
    return bool(user) and user["id"] in book_users(request.app, book_id)


def can_voice(request: web.Request, voice_id: str) -> bool:
    """Whether the user may change a voice: only their books' characters use it (or they're the admin)."""
    if is_admin(request):
        return True
    from .voice_edit import voice_usage

    users = voice_usage(request.app).get(voice_id, [])
    return bool(users) and all(not u.get("default") and can_book(request, u["book"]) for u in users)


def can_series(request: web.Request, series: str) -> bool:
    """Whether the user may change something of a whole series (its pronunciations): all its books are theirs."""
    if is_admin(request):
        return True
    from .dramatize import DRAMA_STORE_KEY
    from .series import book_series, series_slug

    books = [b for b in request.app[DRAMA_STORE_KEY].ids() if series_slug(book_series(request.app, b) or "") == series_slug(series)]
    return bool(books) and all(can_book(request, b) for b in books)


def forbidden(what: str = "This belongs to someone else") -> web.Response:
    return web.json_response({"error": what}, status=403)


def check(request: web.Request) -> web.Response | None:
    """None if the request's user may do this; otherwise the 403 answer."""
    path, method = request.path, request.method
    if is_admin(request) or method in _READ or path in _ANYONE:
        return None
    if m := _BOOK.match(path):  # a book: anyone adds one (they may change it then); otherwise it must be theirs
        book, rest = m.group(1), m.group(2) or ""
        if (method == "POST" and rest == "") or rest in ("/bookmark", "/reports"):
            return None
        return None if can_book(request, book) else forbidden("You can only change books you added")
    if m := _VOICE.match(path):
        rest = m.group(2) or ""
        if method == "DELETE" and rest == "":
            return forbidden("Only the admin deletes voices")
        if rest in ("/duplicate", "/design"):  # checked by the handler (the character / book in the body)
            return None
        return None if can_voice(request, m.group(1)) else forbidden("That voice is also used in books that aren't yours")
    if path.startswith("/api/pronunciations"):
        return None  # checked by the handler (the entry's book or series)
    if (m := _QUEUE_BOOK.match(path)) and method in ("DELETE", "POST"):
        return None if can_book(request, m.group(1)) else forbidden("You can only change books you added")
    return forbidden("Only the admin can change this")
