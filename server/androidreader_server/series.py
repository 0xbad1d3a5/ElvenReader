"""Series memory: the cast of a series (dossiers and voices) carried from one volume to the next.

The series is detected from the book title ("Sasaki and Peeps, Vol. 10" -> "Sasaki and Peeps")
and can be overridden per book. When a new volume is analysed, its characters start from the
series cast with the same names, aliases, dossier and voices; the volume's own read refines
the dossier. Finished casts and user edits are written back.
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

from aiohttp import web

_VOLUME = re.compile(
    r"[\s,:;\-\u2013\u2014(\[]*\b(?:vol(?:ume)?|v|book|part|no|tome|#)\.?\s*\d+(?:\.\d+)?\b.*$", re.IGNORECASE)
_TRAILING_NUMBER = re.compile(r"[\s,:\-\u2013\u2014]+\d+\s*$")
_SLUG = re.compile(r"[^a-z0-9]+")
MIN_LINES = 3


def series_name(title: str) -> str:
    name = _VOLUME.sub("", title or "").strip()
    name = _TRAILING_NUMBER.sub("", name).strip(" ,:;-")
    return name or (title or "").strip()


def series_slug(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-")[:60] or "series"


class SeriesStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{series_slug(name)}.json"

    def get(self, name: str) -> dict:
        p = self.path(name)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            from .dramatize import normalize_names
            from .profile import migrate as migrate_profiles

            normalize_names(data)  # older entries may hold "a prisoner"
            migrate_profiles(data)  # dossiers -> voice profiles
            data.pop("names_v", None)
            data.pop("profile_v", None)
            return data
        return {"name": name, "books": {}, "characters": {}}

    def save(self, data: dict) -> None:
        data["updated"] = time.time()
        self.path(data["name"]).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def all(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out


SERIES_KEY = web.AppKey("series", SeriesStore)


def book_series(app: web.Application, book_id: str) -> str:
    from .dramatize import DRAMA_STORE_KEY

    summary = app[DRAMA_STORE_KEY].summary(book_id) or {}
    return summary.get("series") or series_name(summary.get("title", ""))


def seed(app: web.Application, book_id: str, script: dict) -> int:
    """Start a book's cast from its series (names, aliases, dossier, voices); returns how many."""
    from .dramatize import ensure_character

    name = book_series(app, book_id)
    if not name:
        return 0
    data = app[SERIES_KEY].get(name)
    n = 0
    for entry in data["characters"].values():
        if book_id in entry.get("books", []) and len(entry.get("books", [])) == 1:
            continue  # learned from this very book: let it be read afresh
        cid = ensure_character(script, entry["name"], entry.get("gender", "unknown"), entry.get("age", "unknown"),
                               entry.get("aliases", []))
        if cid is None:
            continue
        c = script["characters"][cid]
        if entry.get("dossier") and not c.get("dossier"):
            c["dossier"] = copy.deepcopy(entry["dossier"])
        if entry.get("dossier_user"):
            c["dossier_user"] = {**entry["dossier_user"], **(c.get("dossier_user") or {})}
        if entry.get("profile") and not c.get("profile"):
            c["profile"] = copy.deepcopy(entry["profile"])
        if entry.get("profile_user"):
            c["profile_user"] = {**entry["profile_user"], **(c.get("profile_user") or {})}
        if entry.get("voice") and not c.get("voice"):
            c["voice"] = entry["voice"]
        if entry.get("description") and not c.get("description"):
            c["description"] = entry["description"]
        if entry.get("role"):  # a floor: the volume's own read can raise the tier, not lower it
            c["role_series"] = entry["role"]
            c.setdefault("role", entry["role"])
        if entry.get("role_user") and not c.get("role_user"):
            c["role_user"] = c["role"] = entry["role_user"]
        c["from_series"] = name
        if entry.get("unnamed"):
            c["unnamed"] = True
        n += 1
    return n


def _higher(new: str | None, old: str | None, user: bool = False) -> str | None:
    """The series' tier for a character: the reader's choice, otherwise the highest tier seen so far."""
    from .roles import RANK

    if user or old not in RANK:
        return new if new in RANK else old
    return new if new in RANK and RANK[new] > RANK[old] else old


def remember(app: web.Application, book_id: str, only: str | None = None) -> str:
    """Write the book's main cast (or one character) to its series."""
    from .dramatize import DRAMA_STORE_KEY, ScriptStore, norm_name

    store = app[DRAMA_STORE_KEY]
    name = book_series(app, book_id)
    if not name:
        return ""
    s = store.script(book_id)
    lines = {c["id"]: c["lines"] for c in ScriptStore.public(s)["characters"]}
    series = app[SERIES_KEY].get(name)
    series["books"][book_id] = (store.summary(book_id) or {}).get("title", "")
    by_name = {norm_name(e["name"]): k for k, e in series["characters"].items()}
    if not only:  # entries learned only from this book that are gone now (merged, renamed) are dropped
        current = {norm_name(c["name"]) for c in s["characters"].values()}
        for key, e in list(series["characters"].items()):
            if set(e.get("books", [])) <= {book_id} and norm_name(e["name"]) not in current:
                del series["characters"][key]
        by_name = {norm_name(e["name"]): k for k, e in series["characters"].items()}
    for cid, c in s["characters"].items():
        if only and cid != only:
            continue
        if not only and lines.get(cid, 0) < MIN_LINES and not c.get("user") and c.get("role") != "main":
            continue
        key = by_name.get(norm_name(c["name"])) or next(
            (by_name[norm_name(a)] for a in c.get("aliases", []) if norm_name(a) in by_name), cid)
        old = series["characters"].get(key, {})
        others = set(old.get("books", [])) - {book_id}  # aliases learned from other volumes are kept
        dossier = {k: v for k, v in (c.get("dossier") or {}).items() if k not in ("facts", "revisions")}
        series["characters"][key] = {
            "name": c["name"],
            **({"unnamed": True} if c.get("unnamed") else {}),
            "aliases": sorted((set(old.get("aliases", [])) if others else set()) | set(c.get("aliases", []))),
            "gender": c.get("gender", "unknown"),
            "age": c.get("age", "unknown"),
            "description": c.get("description", ""),
            "dossier": dossier or old.get("dossier", {}),
            "dossier_user": {**old.get("dossier_user", {}), **(c.get("dossier_user") or {})},
            "profile": c.get("profile") or old.get("profile", {}),
            "profile_user": {**old.get("profile_user", {}), **(c.get("profile_user") or {})},
            "voice": c.get("voice") or old.get("voice"),
            # Tiers from other volumes are a floor; this volume's own earlier tier is simply replaced.
            "role": _higher(c.get("role_user") or c.get("role_auto") or c.get("role"),
                            old.get("role") if others else None, user=bool(c.get("role_user"))),
            "role_user": c.get("role_user"),
            "books": sorted(set(old.get("books", [])) | {book_id}),
        }
        if not series["characters"][key]["role"]:
            series["characters"][key].pop("role")
        if not series["characters"][key]["role_user"]:
            series["characters"][key].pop("role_user")
    app[SERIES_KEY].save(series)
    return name


async def list_series(request: web.Request) -> web.Response:
    return web.json_response({"series": request.app[SERIES_KEY].all()})


async def set_book_series(request: web.Request) -> web.Response:
    """{"series": "Name"} sets a book's series (empty: from the title) and stores its cast there."""
    from .dramatize import DRAMA_STORE_KEY

    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    if store.book(book_id) is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    body = await request.json() if request.can_read_body else {}
    meta_path = store.dir(book_id) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if isinstance(body.get("series"), str):
        meta["series"] = body["series"].strip()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    name = remember(request.app, book_id) if body.get("remember", True) else book_series(request.app, book_id)
    return web.json_response({"series": name, "data": request.app[SERIES_KEY].get(name)})


def setup(app: web.Application) -> None:
    from .app import CONFIG_KEY

    app[SERIES_KEY] = SeriesStore(app[CONFIG_KEY].data_path / "series")
    app.router.add_get("/api/series", list_series)
    app.router.add_post("/api/books/{book}/series", set_book_series)
