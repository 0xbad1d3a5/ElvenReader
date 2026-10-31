"""Per-book data on the server: uploaded text and dramatization scripts (see dramatize.py)."""
from __future__ import annotations

from aiohttp import web


def setup(app: web.Application) -> None:
    from . import dramatize, library

    dramatize.setup(app)
    library.setup(app)
