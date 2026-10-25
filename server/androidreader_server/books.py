"""Per-book data on the server. Dramatization (dramatize.py) registers its routes here."""
from __future__ import annotations

from aiohttp import web


def setup(app: web.Application) -> None:
    """No per-book routes yet."""
