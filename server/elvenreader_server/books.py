"""Per-book data on the server: uploaded text and dramatization scripts (see dramatize.py), each book's
.elvenbook file (bookfile.py), reports of incorrect analysis (reports.py), users (profiles.py) and where
each of them is in each book (bookmark.py)."""
from __future__ import annotations

from aiohttp import web


def setup(app: web.Application) -> None:
    from . import bookfile, bookmark, dramatize, library, profiles, reports

    dramatize.setup(app)
    library.setup(app)
    bookfile.setup(app)
    reports.setup(app)
    profiles.setup(app)
    bookmark.setup(app)
