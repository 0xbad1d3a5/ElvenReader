"""Adding books to the library: an EPUB uploaded from the browser or, from this PC only (process_book.py), a
file path.

The server parses the EPUB (epub.py, the same IDs the old phone app used), keeps a copy of it, stores the
text, cover and pictures, then analyses the book (speakers, tiers) and creates its cast's voices.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import uuid
from pathlib import Path

from aiohttp import web

from . import epub

logger = logging.getLogger("elvenreader.drama")
_LOOPBACK = {"127.0.0.1", "::1"}


async def import_book(app: web.Application, path: Path, *, analyze: bool = True, cast: bool = True,
                      reanalyze: bool = False, engine: str | None = None, series: str | None = None,
                      added_by: str | None = None) -> dict:
    """Parses and stores an EPUB; queues its analysis (and casting) unless it's already done. Returns the
    book's summary with its id and series. A book already in the library keeps its analysis (re-numbered if a
    newer parser split it differently). `series` names the book's series before the analysis starts (by default
    it is detected from the title). `added_by`: the user whose library it joins."""
    from .dramatize import ANALYZER_KEY, DRAMA_STORE_KEY, migrate_ids, picture_refs, reset_for_analysis
    from .series import _write_meta_series, series_info

    store = app[DRAMA_STORE_KEY]
    book_id = await asyncio.to_thread(epub.book_id, path)
    payload = await asyncio.to_thread(epub.parse, path)
    cover = payload.pop("cover")
    d = store.dir(book_id)
    d.mkdir(parents=True, exist_ok=True)
    async with store.lock(book_id):
        old = store.book(book_id)
        store.save_book(book_id, {**payload, **({"cover": base64.b64encode(cover).decode()} if cover else {})},
                        added_by=added_by)
        if old is not None:
            s = store.script(book_id)
            if migrate_ids(old, store.book(book_id), s):
                store.save_script(book_id, s)
                logger.info("book %s re-split; analysis moved to the new ids", book_id)
    target = d / "book.epub"
    if path.resolve() != target.resolve():
        await asyncio.to_thread(shutil.copyfile, path, target)
    await asyncio.to_thread(_store_pictures, store, book_id, target, picture_refs(store.book(book_id)))
    if series and series.strip():
        _write_meta_series(app, book_id, series)

    async with store.lock(book_id):
        s = store.script(book_id)
        busy = s.get("status") in ("queued", "running", "casting")
        if reanalyze and not busy:
            reset_for_analysis(s, keep_dossier=False)
        start = analyze and not busy and (reanalyze or s.get("status") in ("none", "error"))
        if start and cast:
            s["auto_cast"] = True
            if engine:
                s["auto_cast_engine"] = engine
        store.save_script(book_id, s)
    if start:
        await app[ANALYZER_KEY].enqueue(book_id)
    info = series_info(app, book_id)
    return {**(store.summary(book_id) or {}), "id": book_id, "status": store.script(book_id).get("status", "none"),
            "series": info["series"], "series_auto": info["auto"], "series_detected": info["detected"]}


def _store_pictures(store, book_id: str, epub_path: Path, refs: list[str]) -> int:
    import zipfile

    n = 0
    with zipfile.ZipFile(epub_path) as z:
        names = set(z.namelist())
        for ref in refs:
            f = store.picture_file(book_id, ref)
            if f is None or ref not in names or f.exists():
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(z.read(ref))
            n += 1
    return n


def _flag(value, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


async def import_package(app: web.Application, path: Path, replace: bool = False) -> dict:
    """Adds a book file (.elvenbook, see bookfile.py) to the library; returns the book's summary."""
    from .bookfile import import_file
    from .dramatize import DRAMA_STORE_KEY
    from .series import series_info

    book_id = await asyncio.to_thread(import_file, app, path, replace)
    store = app[DRAMA_STORE_KEY]
    info = series_info(app, book_id)
    return {**(store.summary(book_id) or {}), "id": book_id, "status": store.script(book_id).get("status", "none"),
            "series": info["series"], "series_auto": info["auto"], "series_detected": info["detected"]}


async def import_route(request: web.Request) -> web.Response:
    """POST /api/library/import: multipart with an "epub" (or "book") file, from any device; or JSON
    {"path": ...} for a file on this PC (loopback requests only; for process_book.py). The file is an EPUB or a
    book file (.elvenbook). EPUB options: analyze, cast (default true), reanalyze, engine, series (default:
    detected from the title). Book file option: replace (a book already in the library; default false)."""
    from .app import CONFIG_KEY
    from .bookfile import SUFFIX

    cfg = request.app[CONFIG_KEY]
    options: dict = {}
    tmp: Path | None = None
    try:
        if request.content_type.startswith("multipart/"):
            uploads = cfg.data_path / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            async for part in await request.multipart():
                if part.name in ("epub", "book"):
                    suffix = SUFFIX if (part.filename or "").lower().endswith(SUFFIX) else ".epub"
                    tmp = uploads / f"{uuid.uuid4().hex}{suffix}"
                    with open(tmp, "wb") as f:
                        while chunk := await part.read_chunk(1 << 20):
                            f.write(chunk)
                elif part.name:
                    options[part.name] = (await part.text()).strip()
            if tmp is None:
                return web.json_response({"error": "No EPUB file in the upload"}, status=400)
            path = tmp
        else:
            try:
                options = await request.json()
            except ValueError:
                return web.json_response({"error": "Send an EPUB file or {\"path\": ...}"}, status=400)
            if request.remote not in _LOOPBACK:
                return web.json_response({"error": "Paths can only be added from this PC; upload the file instead"},
                                         status=403)
            path = Path(str(options.get("path") or "").strip().strip('"'))
            if not path.is_file() or path.suffix.lower() not in (".epub", SUFFIX):
                return web.json_response({"error": f"No EPUB or {SUFFIX} file at {path}"}, status=400)
        if path.suffix.lower() == SUFFIX:
            from .bookfile import book_id_of
            from .dramatize import DRAMA_STORE_KEY
            from .permissions import can_book
            from .profiles import current

            try:
                existing = await asyncio.to_thread(book_id_of, path)
            except (ValueError, OSError):
                existing = None
            if existing and request.app[DRAMA_STORE_KEY].book(existing) is not None and _flag(options.get("replace"), False) \
                    and not can_book(request, existing):
                return web.json_response({"error": "That book is someone else's: it can't be replaced from here"}, status=403)
            try:
                book = await import_package(request.app, path, _flag(options.get("replace"), False))
            except FileExistsError as exc:
                return web.json_response({"error": f"{exc}; add it with replace to overwrite it"}, status=409)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=400)
            request.app[DRAMA_STORE_KEY].add_user(book["id"], current(request))
            return web.json_response(book)
        try:
            from .profiles import current

            book = await import_book(request.app, path, analyze=_flag(options.get("analyze"), True),
                                     cast=_flag(options.get("cast"), True), reanalyze=_flag(options.get("reanalyze"), False),
                                     engine=(options.get("engine") or None), series=(options.get("series") or None),
                                     added_by=current(request))
        except epub.EpubError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 - zipfile and XML errors from a broken file
            logger.exception("importing %s failed", path)
            return web.json_response({"error": f"Could not read this EPUB: {exc}"}, status=400)
        return web.json_response(book)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def setup(app: web.Application) -> None:
    app.router.add_post("/api/library/import", import_route)
