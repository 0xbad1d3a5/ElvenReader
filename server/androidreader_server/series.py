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


_VOLUME_NUMBER = re.compile(r"\b(?:vol(?:ume)?|v|book|part|no|tome)\.?\s*(\d+(?:\.\d+)?)\b|#\s*(\d+)", re.IGNORECASE)


def volume_key(title: str) -> list[float]:
    """Sort key for the volumes of a series: "Part 2 Volume 3" -> [2, 3]; "Vol. 10" -> [10]; "Book 7" -> [7]."""
    nums = [float(a or b) for a, b in _VOLUME_NUMBER.findall(title or "")]
    if not nums:
        m = re.search(r"(\d+(?:\.\d+)?)\s*$", (title or "").strip())
        nums = [float(m.group(1))] if m else []
    return nums


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


def seed(app: web.Application, book_id: str, script: dict, replace_generated: bool = False) -> int:
    """Start a book's cast from its series (names, aliases, dossier, voices); returns how many.

    With `replace_generated` (applying a series to a book that was already cast), a voice this book made
    for itself (designed or pooled) gives way to the series' voice; voices the reader chose stay."""
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
        generated = (c.get("voice") or "").startswith((f"ch-{book_id[:8]}-", "pool-")) and not c.get("voice_user")
        if entry.get("voice") and (not c.get("voice") or (replace_generated and generated)):
            c["voice"] = entry["voice"]
            if entry.get("description"):
                c["description"] = entry["description"]
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
    """Every series: its stored memory plus `volumes` (its library books in volume order), `cast` and `voiced`."""
    from .dramatize import DRAMA_STORE_KEY

    store = request.app[DRAMA_STORE_KEY]
    groups: dict[str, dict] = {}
    for data in request.app[SERIES_KEY].all():
        chars = data.get("characters", {})
        groups[series_slug(data["name"])] = {**data, "cast": len(chars),
                                             "voiced": sum(1 for e in chars.values() if e.get("voice")), "volumes": []}
    for book_id in store.ids():
        info = series_info(request.app, book_id)
        g = groups.setdefault(series_slug(info["series"]), {"name": info["series"], "characters": {}, "books": {},
                                                            "cast": 0, "voiced": 0, "volumes": []})
        g["volumes"].append({"id": book_id, "title": info["title"], "auto": info["auto"], "volume": info["volume"]})
    for g in groups.values():
        g["volumes"].sort(key=lambda b: (b["volume"] or [float("inf")], b["title"]))
    return web.json_response({"series": sorted(groups.values(), key=lambda g: g["name"].lower())})


def series_info(app: web.Application, book_id: str) -> dict:
    from .dramatize import DRAMA_STORE_KEY

    summary = app[DRAMA_STORE_KEY].summary(book_id) or {}
    detected = series_name(summary.get("title", ""))
    return {"series": summary.get("series") or detected, "auto": not summary.get("series"), "detected": detected,
            "title": summary.get("title", ""), "volume": volume_key(summary.get("title", ""))}


def _write_meta_series(app: web.Application, book_id: str, name: str) -> None:
    from .dramatize import DRAMA_STORE_KEY

    store = app[DRAMA_STORE_KEY]
    meta_path = store.dir(book_id) / "meta.json"
    # A book stored before meta.json existed starts from its summary, so the title etc. aren't lost.
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else dict(store.summary(book_id) or {})
    meta["series"] = name.strip()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def forget(app: web.Application, book_id: str, name: str) -> None:
    """Take a book out of a series' memory: what only it taught is dropped, shared entries lose the book."""
    store = app[SERIES_KEY]
    if not store.path(name).exists():
        return
    data = store.get(name)
    data["books"].pop(book_id, None)
    for key, e in list(data["characters"].items()):
        books = [b for b in e.get("books", []) if b != book_id]
        if not books and book_id in e.get("books", []):
            del data["characters"][key]
        else:
            e["books"] = books
    if data["books"] or data["characters"]:
        store.save(data)
    else:
        store.path(name).unlink(missing_ok=True)


def unseed(script: dict, name: str) -> int:
    """Remove characters a series brought in that the book never gave a line (the series was wrong)."""
    from .dramatize import ScriptStore

    lines = {c["id"]: c["lines"] for c in ScriptStore.public(script)["characters"]}
    gone = [cid for cid, c in script["characters"].items()
            if c.get("from_series") and series_slug(c["from_series"]) == series_slug(name)
            and not lines.get(cid) and not c.get("user")]
    for cid in gone:
        del script["characters"][cid]
    return len(gone)


def _others_in(app: web.Application, book_id: str, name: str) -> int:
    """Characters the series learned from other books (what applying it would bring)."""
    data = app[SERIES_KEY].get(name)
    return sum(1 for e in data["characters"].values() if set(e.get("books", [])) - {book_id})


async def apply_series(app: web.Application, book_id: str) -> int:
    """Give a book its series' cast now, without labelling it again: series characters join the cast (so the
    labelling still to come can use them), matching characters take the series' voice, profile and tier.
    A finished book then drops series characters who have no lines in it."""
    from .dramatize import DRAMA_STORE_KEY, follow_voice, prune_characters
    from .app import VOICES_KEY

    store = app[DRAMA_STORE_KEY]
    async with store.lock(book_id):
        s = store.script(book_id)
        n = seed(app, book_id, s, replace_generated=True)
        s["series_seeded"] = max(n, 1)
        if s.get("status") == "done":
            prune_characters(s)
        for c in s["characters"].values():
            follow_voice(c, app[VOICES_KEY])
        store.save_script(book_id, s)
        done = s.get("status") == "done"
    if done:  # otherwise the cast is written to the series when it is created
        remember(app, book_id)
    return n


async def set_book_series(request: web.Request) -> web.Response:
    """{"series": "Name"} sets a book's series ("" detects it from the title). The book leaves its old
    series' memory. {"apply": true} also gives the book the new series' cast now (apply_series); otherwise
    the book's cast is written to the series only when that can't overwrite what other volumes taught it
    (`can_apply` then says the series has a cast to apply). GET returns the book's series info."""
    from .dramatize import DRAMA_STORE_KEY

    store = request.app[DRAMA_STORE_KEY]
    book_id = request.match_info["book"]
    if store.book(book_id) is None:
        return web.json_response({"error": "book not uploaded"}, status=404)
    if request.method == "GET":
        info = series_info(request.app, book_id)
        return web.json_response({**info, "can_apply": _others_in(request.app, book_id, info["series"]) > 0})
    body = await request.json() if request.can_read_body else {}
    old = book_series(request.app, book_id)
    if isinstance(body.get("series"), str):
        _write_meta_series(request.app, book_id, body["series"])
    name = book_series(request.app, book_id)
    if series_slug(name) != series_slug(old):
        forget(request.app, book_id, old)
        async with store.lock(book_id):
            s = store.script(book_id)
            if unseed(s, old):
                store.save_script(book_id, s)
    others = _others_in(request.app, book_id, name)
    applied = None
    if body.get("apply"):
        applied = await apply_series(request.app, book_id)
    elif body.get("remember", True) and not others:
        remember(request.app, book_id)
    return web.json_response({**series_info(request.app, book_id), "can_apply": others > 0 and applied is None,
                              "applied": applied, "data": request.app[SERIES_KEY].get(name)})


def _merge_entry(into: dict, e: dict) -> None:
    into["aliases"] = sorted(set(into.get("aliases", [])) | set(e.get("aliases", [])) | ({e["name"]} - {into["name"]}))
    into["books"] = sorted(set(into.get("books", [])) | set(e.get("books", [])))
    for k in ("voice", "description", "profile", "dossier"):
        if not into.get(k) and e.get(k):
            into[k] = e[k]
    for k in ("profile_user", "dossier_user"):
        into[k] = {**(e.get(k) or {}), **(into.get(k) or {})}
    if e.get("role_user") and not into.get("role_user"):
        into["role_user"] = into["role"] = e["role_user"]
    elif not into.get("role_user"):
        role = _higher(e.get("role"), into.get("role"))
        if role:
            into["role"] = role


async def rename_series(request: web.Request) -> web.Response:
    """{"from": "Old", "to": "New"}: renames a series for all its books; if "New" exists, merges into it
    (characters with the same name or alias become one; the existing series' voices win)."""
    from .dramatize import DRAMA_STORE_KEY, norm_name

    body = await request.json() if request.can_read_body else {}
    src_name, dst_name = str(body.get("from") or "").strip(), str(body.get("to") or "").strip()
    if not src_name or not dst_name:
        return web.json_response({"error": "Give the series' current and new name"}, status=400)
    app, store = request.app, request.app[SERIES_KEY]
    src = store.get(src_name)
    same = series_slug(src_name) == series_slug(dst_name)
    dst = src if same else store.get(dst_name)
    dst["name"] = dst_name
    books = [b for b in app[DRAMA_STORE_KEY].ids() if series_slug(book_series(app, b)) == series_slug(src_name)]
    if not same:
        index = {}
        for key, e in dst["characters"].items():
            for n in (e["name"], *e.get("aliases", [])):
                index.setdefault(norm_name(n), key)
        for key, e in src["characters"].items():
            hit = next((index[norm_name(n)] for n in (e["name"], *e.get("aliases", [])) if norm_name(n) in index), None)
            if hit:
                _merge_entry(dst["characters"][hit], e)
            else:
                new_key = key if key not in dst["characters"] else f"{key}-{series_slug(src_name)}"[:80]
                dst["characters"][new_key] = e
                for n in (e["name"], *e.get("aliases", [])):
                    index.setdefault(norm_name(n), new_key)
        dst["books"].update(src.get("books", {}))
    for b in books:
        _write_meta_series(app, b, dst_name)
        async with app[DRAMA_STORE_KEY].lock(b):
            s = app[DRAMA_STORE_KEY].script(b)
            changed = False
            for c in s["characters"].values():
                if c.get("from_series") and series_slug(c["from_series"]) == series_slug(src_name):
                    c["from_series"] = dst_name
                    changed = True
            if changed:
                app[DRAMA_STORE_KEY].save_script(b, s)
    store.save(dst)
    if not same:
        store.path(src_name).unlink(missing_ok=True)
    return web.json_response({"series": dst_name, "books": books, "merged": not same and bool(src["characters"]),
                              "data": dst})


def setup(app: web.Application) -> None:
    from .app import CONFIG_KEY

    app[SERIES_KEY] = SeriesStore(app[CONFIG_KEY].data_path / "series")
    app.router.add_get("/api/series", list_series)
    app.router.add_post("/api/series/rename", rename_series)
    app.router.add_get("/api/books/{book}/series", set_book_series)
    app.router.add_post("/api/books/{book}/series", set_book_series)
