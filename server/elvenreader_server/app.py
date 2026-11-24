"""aiohttp application: the ElvenReader server's API and web pages."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import aiohttp
from aiohttp import web

from . import __version__
from . import delivery as delivery_tags
from .audio import FORMATS as AUDIO_FORMATS
from .audio import EncodeError, SilenceTrimmer, noise_gap_db, normalize, tighten, time_stretch, wav_bytes
from .audio import encode as encode_audio
from .cache import AudioCache, cache_key
from .config import ServerConfig
from .engines.manager import AudioChunk, Done, EngineError, EngineManager, StreamStart
from .engines.registry import ENGINES, EngineSpec
from .netinfo import lan_addresses
from .pronunciation import PHONEME_FORMATS, Entry, PronunciationStore, apply, entries_version
from .text import (
    MAX_CHUNK_CHARS,
    MAX_TEXT_CHARS,
    clamp_pause_ms,
    clamp_speed,
    clean_text,
    engine_text,
    is_speakable,
    split_for_synthesis,
    trailing_pause_ms,
)
from .voices import DESIGN_TEXT, VoiceStore  # noqa: F401 - DESIGN_TEXT is imported from here too

logger = logging.getLogger("elvenreader")

CONFIG_KEY = web.AppKey("config", ServerConfig)
ENGINES_KEY = web.AppKey("engines", EngineManager)
VOICES_KEY = web.AppKey("voices", VoiceStore)
PRON_KEY = web.AppKey("pronunciations", PronunciationStore)
CACHE_KEY = web.AppKey("cache", AudioCache)
EXTRA_KEY = web.AppKey("extra", dict)
READERS_KEY = web.AppKey("readers", dict)  # reading page -> (the passage it is playing, when it said so)

_PUBLIC_PATHS = {"/api/ping"}
_LOOPBACK_ONLY_PATHS = {"/"}
_LOOPBACK = {"127.0.0.1", "::1"}


def _json_error(status: int, detail: str) -> web.Response:
    return web.json_response({"error": detail}, status=status)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    path = request.path
    if path in _PUBLIC_PATHS or path == "/books" or path.startswith("/books/"):
        # The viewer page itself is public; it only embeds the token for loopback requests
        # and otherwise asks for it, and every API call it makes needs the token.
        return await handler(request)
    if path in _LOOPBACK_ONLY_PATHS:
        if request.remote not in _LOOPBACK:
            raise web.HTTPNotFound()
        return await handler(request)
    from .profiles import USER_KEY, for_token

    header = request.headers.get("Authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    user = for_token(request.app, supplied)  # each user has a token of their own (profiles.py)
    if user is None:
        return _json_error(401, "Missing or invalid token")
    request[USER_KEY] = user
    from .permissions import check

    if (denied := check(request)) is not None:  # each user's own library; site-wide changes are the admin's
        return denied
    response = await handler(request)
    # Voices, scripts and casts change under the same URLs: never let a browser reuse a stale answer.
    if isinstance(response, web.StreamResponse) and "Cache-Control" not in response.headers and not response.prepared:
        response.headers["Cache-Control"] = "no-cache"
    return response


async def _json_body(request: web.Request) -> dict:
    try:
        body = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text='{"error": "Body must be JSON"}', content_type="application/json") from exc
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text='{"error": "Body must be a JSON object"}', content_type="application/json")
    return body


def _str(body: dict, key: str) -> str:
    v = body.get(key)
    return v.strip() if isinstance(v, str) else ""


# ---------------------------------------------------------------- status


async def ping(request: web.Request) -> web.Response:
    return web.json_response({"service": "elvenreader-server", "version": __version__})


async def health(request: web.Request) -> web.Response:
    app = request.app
    cfg = app[CONFIG_KEY]
    manager = app[ENGINES_KEY]
    default_spec = manager.specs.get(cfg.default_engine)
    return web.json_response(
        {
            "server": "ok",
            "version": __version__,
            "default_engine": cfg.default_engine,
            "default_engine_label": default_spec.label if default_spec else None,
            "engine": manager.status(),
            "voices": len(app[VOICES_KEY].list()),
            "cache": app[CACHE_KEY].stats(),
        }
    )


async def engines(request: web.Request) -> web.Response:
    manager = request.app[ENGINES_KEY]
    return web.json_response(
        {"engines": manager.engines(), "default": request.app[CONFIG_KEY].default_engine, "status": manager.status()}
    )


async def engine_load(request: web.Request) -> web.Response:
    """Make an engine the site's speech model (same as PUT /api/settings {"speech_engine": ...})."""
    try:
        await set_speech_engine(request.app, request.match_info["engine"])
    except EngineError as exc:
        return _json_error(exc.status, exc.detail)
    return web.json_response({"status": request.app[ENGINES_KEY].status()})


async def engine_unload(request: web.Request) -> web.Response:
    await request.app[ENGINES_KEY].unload()
    return web.json_response({"status": request.app[ENGINES_KEY].status()})


# ---------------------------------------------------------------- site settings

LLM_MODES = ("adaptive", "shared", "auto", "gpu", "cpu")


def settings_json(app: web.Application) -> dict:
    from .activity import ACTIVITY_KEY

    cfg, manager = app[CONFIG_KEY], app[ENGINES_KEY]
    installed = [e for e in manager.engines() if e["installed"]]
    store = app[VOICES_KEY]
    default = store.default(cfg.default_voice)
    return {
        "speech_engine": cfg.default_engine,
        "narration_engine": cfg.narration_engine,
        "voice_design_engine": cfg.voice_design_engine,
        "default_voice": default.id if default else "",
        "voices": [{"id": v.id, "name": v.name} for v in store.list() if v.source != "character"],
        "llm_mode": cfg.llm_mode,
        # Which computer analyses books: "local" (this PC) or "remote" (the inference server, when one is set up).
        "llm_backend": cfg.llm_backend,
        "llm_local": {"label": "This PC", "model": Path(cfg.llm_model).stem if cfg.llm_model else ""},
        "llm_remote": ({"label": cfg.llm_remote_label or urlparse(cfg.llm_remote_url).hostname or "Inference server",
                        "model": cfg.llm_remote_model, "host": urlparse(cfg.llm_remote_url).netloc}
                       if cfg.llm_remote_url and cfg.llm_remote_model else None),
        "idle_after_s": cfg.idle_after_s,
        "design_pause_s": cfg.design_pause_s,
        "activity": app[ACTIVITY_KEY].json(),
        "engines": [{"id": e["id"], "label": e["label"], "voice_design": e["voice_design"], "stats": e["stats"],
                     "vram_gb": e.get("vram_gb")} for e in installed],
        "status": manager.status(),
        # The speech model's defaults for the per-line settings (the line dialog's sliders): {name: {value, model,
        # min, max, step}}; `model` is the engine's own.
        "tuning": {name: {"value": v, "model": float(spec.defaults.get(TUNING[name]["option"], TUNING[name]["fallback"])),
                          "min": TUNING[name]["range"][0], "max": TUNING[name]["range"][1], "step": TUNING[name]["step"]}
                   for spec in [manager.specs.get(cfg.default_engine)]
                   for name, v in default_tuning(cfg, cfg.default_engine, spec).items()},
    }


async def set_speech_engine(app: web.Application, engine_id: str) -> None:
    """The one speech model the site reads with: saved, then loaded in place of the old one."""
    from .config import save_settings

    manager = app[ENGINES_KEY]
    spec = manager.spec(engine_id)
    if not spec.installed(manager.models_dir):
        raise EngineError(f"{spec.label} is not installed on the PC", 400)
    save_settings(app[CONFIG_KEY], default_engine=engine_id)
    await manager.preload(engine_id)


async def settings_get(request: web.Request) -> web.Response:
    return web.json_response(settings_json(request.app))


async def settings_put(request: web.Request) -> web.Response:
    """{speech_engine?, narration_engine?, voice_design_engine?, llm_mode?, default_voice?, idle_after_s?,
    design_pause_s?, tuning?}. Changing the speech model (or the narration model: "" for none) loads it (about a
    minute) in the background; GET shows when it's ready. `tuning`
    ({"guidance": 2, "temperature": 1}) sets the speech model's defaults (engine_options); lines are made again
    with them as they are played."""
    from .config import save_settings

    app = request.app
    body = await _json_body(request)
    manager = app[ENGINES_KEY]
    changes = {}
    tuning = body.get("tuning")
    if tuning is not None:
        cfg = app[CONFIG_KEY]
        spec = manager.specs.get(cfg.default_engine)
        if not isinstance(tuning, dict) or spec is None:
            return _json_error(400, "tuning must be {setting: number}")
        options = cfg.engine_options.setdefault(cfg.default_engine, {})  # the same dict the engines were given
        for name, v in tuning.items():
            if name not in spec.tuning or isinstance(v, bool) or not isinstance(v, (int, float)):
                return _json_error(400, f"{spec.label} has no setting {name!r}")
            lo, hi = TUNING[name]["range"]
            if not lo <= v <= hi:
                return _json_error(400, f"{name} must be between {lo} and {hi}")
            option = TUNING[name]["option"]
            if round(float(v), 2) == float(spec.defaults.get(option, TUNING[name]["fallback"])):
                options.pop(option, None)  # the model's own value
            else:
                options[option] = round(float(v), 2)
        if not options:
            cfg.engine_options.pop(cfg.default_engine, None)
        changes["engine_options"] = cfg.engine_options
    design = _str(body, "voice_design_engine")
    if design:
        spec = manager.spec(design) if design in manager.specs else None
        if spec is None or not spec.voice_design or not spec.installed(manager.models_dir):
            return _json_error(400, f"{design} can't design voices on this PC")
        changes["voice_design_engine"] = design
    mode = _str(body, "llm_mode")
    if mode:
        if mode not in LLM_MODES:
            return _json_error(400, f"llm_mode must be one of {', '.join(LLM_MODES)}")
        changes["llm_mode"] = mode
    backend = _str(body, "llm_backend")
    if backend and backend != app[CONFIG_KEY].llm_backend:
        if backend not in ("local", "remote"):
            return _json_error(400, "llm_backend must be local or remote")
        if backend == "remote" and not (app[CONFIG_KEY].llm_remote_url and app[CONFIG_KEY].llm_remote_model):
            return _json_error(400, "No analysis server is set up (llm_remote_url and llm_remote_model in config.json)")
        changes["llm_backend"] = backend
    voice = _str(body, "default_voice")
    if voice:
        v = app[VOICES_KEY].get(voice)
        if v is None or v.source == "character":
            return _json_error(400, "Choose one of the voices in the list")
        changes["default_voice"] = voice
    from .activity import ACTIVITY_KEY

    for key in ("idle_after_s", "design_pause_s"):
        if body.get(key) is None:
            continue
        try:
            seconds = float(body[key])
        except (TypeError, ValueError):
            return _json_error(400, f"{key} must be a number of seconds")
        if not 30 <= seconds <= 24 * 3600:
            return _json_error(400, f"{key} must be between 30 seconds and a day")
        changes[key] = seconds
        setattr(app[ACTIVITY_KEY], key, seconds)
    speech = _str(body, "speech_engine")
    if speech and speech != app[CONFIG_KEY].default_engine:
        spec = manager.specs.get(speech)
        if spec is None or not spec.installed(manager.models_dir):
            return _json_error(400, f"{speech} is not installed on the PC")
    narration = body.get("narration_engine")
    if narration is not None:
        narration = str(narration).strip()
        spec = manager.specs.get(narration) if narration else None
        if narration and (spec is None or not spec.installed(manager.models_dir)):
            return _json_error(400, f"{narration} is not installed on the PC")
        if narration != app[CONFIG_KEY].narration_engine:
            changes["narration_engine"] = narration
    if changes:
        save_settings(app[CONFIG_KEY], **changes)
        if "voice_design_engine" in changes:
            await manager.unload_designer()
        if changes.get("llm_backend") == "remote":
            # The local LLM gives the GPU back now; a step it was in the middle of is done again on the server.
            from .dramatize import ANALYZER_KEY

            if ANALYZER_KEY in app and app[ANALYZER_KEY].llm.proc is not None:
                await app[ANALYZER_KEY].llm.preempt()
    if speech and speech != app[CONFIG_KEY].default_engine:
        save_settings(app[CONFIG_KEY], default_engine=speech)
        manager.resident = speech

        async def load():
            try:
                await manager.preload(speech)
            except EngineError as exc:
                logger.warning("could not load %s: %s", speech, exc.detail)

        app[EXTRA_KEY]["loading"] = asyncio.ensure_future(load())
        await asyncio.sleep(0)
    if "narration_engine" in changes or speech:
        async def load_narration():
            try:
                await manager.preload_narration(app[CONFIG_KEY].narration_engine)
            except EngineError as exc:
                logger.warning("could not load the narration model: %s", exc.detail)

        app[EXTRA_KEY]["loading_narration"] = asyncio.ensure_future(load_narration())
        await asyncio.sleep(0)
    return web.json_response(settings_json(app))


# ---------------------------------------------------------------- voices


async def voices(request: web.Request) -> web.Response:
    store = request.app[VOICES_KEY]
    default = store.default(request.app[CONFIG_KEY].default_voice)
    if request.query.get("details") == "1":  # the voice editor: every voice, with who uses it
        from .voice_edit import listing_details

        return web.json_response({"voices": listing_details(request.app, store),
                                  "default": default.id if default else None})
    include_characters = request.query.get("all") == "1"
    items = [v.public() for v in store.list() if include_characters or v.source != "character"]
    return web.json_response({"voices": items, "default": default.id if default else None})


async def voice_create(request: web.Request) -> web.Response:
    store = request.app[VOICES_KEY]
    fields: dict[str, str] = {}
    audio: bytes | None = None
    if request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        async for part in reader:
            if part.name == "audio":
                audio = await part.read(decode=False)
            elif part.name:
                fields[part.name] = (await part.text()).strip()
    else:
        body = await _json_body(request)
        fields = {k: str(v) for k, v in body.items() if isinstance(v, (str, int, float))}
        if isinstance(body.get("audio_base64"), str):
            audio = base64.b64decode(body["audio_base64"])
    if not audio:
        return _json_error(400, "Missing reference audio (WAV)")
    try:
        voice = store.add(
            fields.get("name", ""), audio, fields.get("ref_text", ""), language=fields.get("language", ""),
            instruct=fields.get("instruct", ""),
        )
    except ValueError as exc:
        return _json_error(400, str(exc))
    return web.json_response(voice.public(), status=201)


async def voice_delete(request: web.Request) -> web.Response:
    if not request.app[VOICES_KEY].delete(request.match_info["voice"]):
        return _json_error(404, "No such voice")
    return web.json_response({"deleted": True})


async def voice_audio(request: web.Request) -> web.StreamResponse:
    voice = request.app[VOICES_KEY].get(request.match_info["voice"])
    if voice is None or voice.wav_path is None:
        return _json_error(404, "No such voice")
    # The file changes when the voice is edited, so browsers must revalidate (ETag) every time.
    return web.FileResponse(voice.wav_path, headers={"Content-Type": "audio/wav", "Cache-Control": "no-cache"})


async def collect_audio(stream) -> tuple[int, bytes]:
    rate, parts = 24000, []
    async with contextlib.aclosing(stream):
        async for ev in stream:
            if isinstance(ev, StreamStart):
                rate = ev.sample_rate
            elif isinstance(ev, AudioChunk):
                parts.append(ev.pcm)
    return rate, b"".join(parts)


DESIGN_MIN_GAP_DB = 25.0  # Qwen3-TTS's designs measured 23-37 dB; VoxCPM2's 6-67 dB, by seed
DESIGN_TRIES = 4


async def design_clip(app: web.Application, engine: str, description: str, text: str,
                      seed: int | None, tries: int = DESIGN_TRIES) -> tuple[int, bytes, int | None]:
    """A designed clip without background hum: a design whose pauses sit less than DESIGN_MIN_GAP_DB under the
    speech (noise_gap_db) is made again with the next seed, up to `tries` times, and the cleanest is kept.
    Returns (rate, trimmed PCM, the seed it was made with)."""
    best = None
    for k in range(tries):
        s = seed if k == 0 else (int(seed or 0) + 7919 * k) % 100000
        rate, pcm = await collect_audio(app[ENGINES_KEY].design(engine, description, text, s))
        if not pcm:
            continue
        pcm = SilenceTrimmer(rate).feed(pcm)
        gap = noise_gap_db(pcm, rate)
        if best is None or (gap if gap is not None else 99.0) > best[3]:
            best = (rate, pcm, s, gap if gap is not None else 99.0)
        if gap is None or gap >= DESIGN_MIN_GAP_DB:
            break
        logger.info("design of %r hums (pauses %.1f dB under the speech, seed %s); trying another seed",
                    description[:60], gap, s)
    if best is None:
        raise EngineError("Voice design produced no audio")
    return best[0], best[1], best[2]


async def design_voice(app: web.Application, engine: str, description: str, name: str, *, text: str = "",
                       seed: int | None = None, source: str = "design", voice_id: str | None = None,
                       language: str = "English", tries: int = DESIGN_TRIES):
    """Create a voice from a description and lock it to the generated clip."""
    text = text or DESIGN_TEXT
    rate, pcm, seed = await design_clip(app, engine, description, text, seed, tries)
    return app[VOICES_KEY].add(
        name, wav_bytes(pcm, rate), text, language=language, instruct=description, seed=seed,
        source=source, description=description, voice_id=voice_id,
    )


async def voice_design(request: web.Request) -> web.Response:
    body = await _json_body(request)
    from .activity import touch

    touch(request.app, design=True)
    description = _str(body, "description")
    if not description:
        return _json_error(400, "Describe the voice, e.g. 'female, young adult, warm, british accent'")
    engine = _str(body, "engine") or request.app[CONFIG_KEY].voice_design_engine
    try:
        voice = await design_voice(request.app, engine, description, _str(body, "name") or description[:40],
                                   text=_str(body, "text"), seed=body.get("seed"))
    except EngineError as exc:
        return _json_error(exc.status, exc.detail)
    return web.json_response(voice.public(), status=201)


# ---------------------------------------------------------------- pronunciations


def _series_of(app: web.Application, book: str | None) -> str:
    """The series a book belongs to ("" without a book or a library)."""
    if not book:
        return ""
    try:
        from .series import book_series

        return book_series(app, book)
    except (KeyError, ValueError, OSError):
        return ""


def book_users(app: web.Application, book: str | None) -> list[str]:
    """Whose user-wide pronunciations a book is read with: the users whose library has it."""
    if not book:
        return []
    from .permissions import book_users as users

    return users(app, book)


def pronunciations_for(app: web.Application, book: str | None) -> list[Entry]:
    """The entries a book is read with: its users', its series' and its own."""
    return app[PRON_KEY].for_book(book, _series_of(app, book) or None, book_users(app, book))


def _all_pronunciations(request: web.Request) -> dict:
    """Every entry this user deals with (the Pronunciations page): their own user-wide ones, and those of the series
    and books in their library; the admin's, all of them. Each with `can_edit`; with the users' names and the books'
    titles, series and users (whose user-wide entries they're read with), and `series`: the series of those books, each with whether this user may change its entries."""
    from .dramatize import DRAMA_STORE_KEY
    from .permissions import can_book, can_series, is_admin
    from .profiles import all_users, current
    from .pronunciation import series_key

    app, me, admin = request.app, current(request), is_admin(request)
    store = app[DRAMA_STORE_KEY]
    books = [b for b in store.ids() if admin or can_book(request, b)]
    info = {b: {"title": (store.summary(b) or {}).get("title", b), "series": _series_of(app, b),
                "users": book_users(app, b)} for b in books}
    names = sorted({i["series"] for i in info.values() if i["series"]}, key=str.casefold)
    keys = {series_key(n) for n in names}
    entries = [e for e in app[PRON_KEY].all()
               if admin or (e.book and e.book in info) or (not e.book and e.series and series_key(e.series) in keys)
               or (not e.book and not e.series and e.user == me)]
    return {
        "entries": [{**e.to_json(), "can_edit": _may_pronounce(request, e.book, e.series, e.user)} for e in entries],
        "users": {u["id"]: u["name"] for u in all_users(app)},
        "books": info,
        "series": [{"name": n, "can_edit": can_series(request, n)} for n in names],
    }


async def pronunciations(request: web.Request) -> web.Response:
    """?book= (that book's entries, its series' and its users') or ?series= (a series' and the requesting user's).
    `users`: whose user-wide entries are listed. ?all=1: everything for the Pronunciations page."""
    from .profiles import current

    if request.query.get("all"):
        return web.json_response(_all_pronunciations(request))
    book = request.query.get("book") or None
    series = request.query.get("series") or _series_of(request.app, book)
    users = book_users(request.app, book) if book else [u for u in [current(request)] if u]
    return web.json_response({**request.app[PRON_KEY].listing(book, series or None, users), "users": users})


def _may_pronounce(request: web.Request, book: str | None, series: str | None, user: str | None) -> bool:
    """Pronunciations of a book are for those who added it; of a series, for those who added all its books; a
    user's, for that user (and the admin for all of them, permissions.py)."""
    from .permissions import can_book, can_series, is_admin
    from .profiles import current

    if book:
        return can_book(request, book)
    if series:
        return can_series(request, series)
    return bool(user) and (user == current(request) or is_admin(request))


def _entry(request: web.Request):
    return next((e for e in request.app[PRON_KEY].all() if e.id == request.match_info["entry"]), None)


async def pronunciation_add(request: web.Request) -> web.Response:
    """{term, respelling?, phonemes?, case_sensitive?, scope: "series"|"book"|"user", book?, series?}.
    scope "series" with a book means that book's series (the default); "user" is every book of the requesting
    user's library."""
    from .profiles import current

    body = await _json_body(request)
    scope = _str(body, "scope") or "series"
    if scope not in ("series", "book", "user"):
        return _json_error(400, "A pronunciation is for a series, a book or all your books")
    user = current(request) if scope == "user" else None
    book = _str(body, "book") if scope == "book" else ""
    series = (_str(body, "series") or _series_of(request.app, _str(body, "book"))) if scope == "series" else ""
    if scope == "book" and not book:
        return _json_error(400, "Say which book")
    if scope == "series" and not series:
        return _json_error(400, "Say which series (or a book of it)")
    if not _may_pronounce(request, book or None, series or None, user):
        return _json_error(403, "Only for your own books (a series: when all its books are yours)")
    try:
        entry = request.app[PRON_KEY].add(
            _str(body, "term"), _str(body, "respelling"), phonemes=_str(body, "phonemes"),
            case_sensitive=bool(body.get("case_sensitive")), book=book or None, series=series or None, user=user,
        )
    except ValueError as exc:
        return _json_error(400, str(exc))
    return web.json_response(entry.to_json(), status=201)


async def pronunciation_update(request: web.Request) -> web.Response:
    body = await _json_body(request)
    old = _entry(request)
    if old is None:
        return _json_error(404, "No such entry")
    if not _may_pronounce(request, old.book, old.series, old.user):
        return _json_error(403, "That pronunciation isn't yours to change")
    entry = request.app[PRON_KEY].update(request.match_info["entry"], **body)
    if entry is None:
        return _json_error(404, "No such entry")
    return web.json_response(entry.to_json())


async def pronunciation_delete(request: web.Request) -> web.Response:
    old = _entry(request)
    if old is not None and not _may_pronounce(request, old.book, old.series, old.user):
        return _json_error(403, "That pronunciation isn't yours to change")
    if not request.app[PRON_KEY].delete(request.match_info["entry"]):
        return _json_error(404, "No such entry")
    return web.json_response({"deleted": True})


# ---------------------------------------------------------------- synthesis


def _audio_headers(sample_rate: int, **extra: str) -> dict:
    h = {
        "Content-Type": f"audio/L16; rate={sample_rate}; channels=1",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "1",
        "X-Encoding": "pcm16le",
        "Cache-Control": "no-store",
    }
    h.update(extra)
    return h


def _extra_entries(body: dict) -> list[Entry]:
    out = []
    for raw in body.get("extra_pronunciations") or []:
        if isinstance(raw, dict) and str(raw.get("term", "")).strip():
            out.append(Entry(term=" ".join(str(raw["term"]).split()), respelling=str(raw.get("respelling") or ""),
                             phonemes=str(raw.get("phonemes") or "").upper(),
                             case_sensitive=bool(raw.get("case_sensitive")), book="__preview__"))
    return out


def chunk_limit(spec: EngineSpec) -> int:
    return spec.max_chunk_chars or MAX_CHUNK_CHARS


# Per-line settings a speech engine can take (EngineSpec.tuning): the engine option each sets, its range, and the
# value when the engine's options don't say.
TUNING = {
    "guidance": {"option": "cfg_value", "range": (1.0, 4.0), "step": 0.25, "fallback": 2.0},
    # 0.6 made one of Kirito's lines silent (another seed: -35 dB); 0.75-1.25 said every word, quieter and a
    # little clearer below 1, louder and breathier above.
    "temperature": {"option": "temperature", "range": (0.7, 1.3), "step": 0.05, "fallback": 1.0},
}


def default_tuning(cfg: ServerConfig, engine_id: str, spec: EngineSpec | None) -> dict[str, float]:
    """{setting: the engine's own value} for the per-line settings the engine takes."""
    if spec is None:
        return {}
    options = {**spec.defaults, **cfg.engine_options.get(engine_id, {})}
    return {name: float(options.get(TUNING[name]["option"], TUNING[name]["fallback"]))
            for name in spec.tuning if name in TUNING}


def line_tuning(cfg: ServerConfig, engine_id: str, spec: EngineSpec | None, values: dict) -> dict[str, float]:
    """A line's own settings from `values` ({setting: number}), kept to their range; settings the engine doesn't
    take, and ones equal to the engine's default (the same clip as without), are left out."""
    out = {}
    for name, default in default_tuning(cfg, engine_id, spec).items():
        v = values.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        lo, hi = TUNING[name]["range"]
        v = round(min(hi, max(lo, float(v))), 2)
        if v != default:
            out[name] = v
    return out


def synthesis_key(cfg: ServerConfig, engine_id: str, spec: EngineSpec, voice, text: str, language: str | None,
                  delivery: str | None, seed: int | None, tuning: dict | None = None) -> str:
    """The audio cache key of an utterance (`text` is final: cleaned, pronunciations applied)."""
    options = {**spec.defaults, **cfg.engine_options.get(engine_id, {})}
    # Text longer than the default limit is made in pieces of the engine's own size; that is part of the sound
    # (each piece starts its delivery afresh), so only those lines get a new key when the size changes.
    split = {"split": chunk_limit(spec)} if len(text) > MAX_CHUNK_CHARS and chunk_limit(spec) != MAX_CHUNK_CHARS else {}
    if spec.brackets_as_asides and ("(" in text or ")" in text):
        split["brackets"] = "asides"
    split.update(tuning or {})  # only lines given their own settings get another key
    if delivery and spec.styles_version:
        split["styles"] = spec.styles_version
    reread = sorted(name for name, pattern in spec.reread.items() if re.search(pattern, text))
    if reread:
        split["reread"] = ",".join(reread)
    return cache_key(
        engine=engine_id, options=options, voice=voice.fingerprint, language=language,
        # v3: clips made after a voice was edited in place could still have the old voice (engines cached
        # its reference by path); everything is made again.
        delivery=delivery, seed=seed, text=text, trim=cfg.trim_silence, v=3, **split,
    )


def engine_for(cfg: ServerConfig, specs: dict, models_dir, narration: bool) -> str:
    """The speech model a passage is read with: the narration model (Settings) for narration, in whatever voice
    reads it (the book's narrator, or a character narrating in the first person); the speech model for every
    quotation, which follows its delivery (VoxCPM2: "(shouting, urgent)")."""
    engine = cfg.narration_engine
    if narration and engine and engine != cfg.default_engine and engine in specs and specs[engine].installed(models_dir):
        return engine
    return cfg.default_engine


class SynthesisPlan:
    """Everything decided before synthesis: engine, voice, final text and cache key."""

    def __init__(self, app: web.Application, body: dict, text: str):
        cfg = app[CONFIG_KEY]
        manager = app[ENGINES_KEY]
        store = app[VOICES_KEY]
        requested = _str(body, "voice")
        self.voice = store.get(requested) if requested else None
        if self.voice is None:
            self.voice = store.default(cfg.default_voice)
        if self.voice is None:
            raise EngineError("No voices on the PC yet. Import or upload one first.", 503)
        # The site's speech models (Settings): the speech model, and the narration model for narration
        # ({"kind": "narration"}: the reading page and the book file say which passages are); a request can't swap
        # them for another.
        self.engine_id = engine_for(cfg, manager.specs, manager.models_dir, _str(body, "kind") == "narration")
        asked = _str(body, "engine")
        if asked and asked not in (cfg.default_engine, cfg.narration_engine):
            raise EngineError(f"The site reads with {cfg.default_engine}; change the speech model in Settings", 409)
        self.spec: EngineSpec = manager.spec(self.engine_id)
        if not self.spec.installed(manager.models_dir):
            raise EngineError(f"{self.spec.label} is not installed on the PC", 503)
        book = _str(body, "book") or None
        entries = pronunciations_for(app, book)
        extra = _extra_entries(body)
        if extra:
            keys = {e.key() for e in extra}
            entries = [e for e in entries if e.key() not in keys] + extra
        self.pron_version = entries_version(entries)
        text = engine_text(self.spec, text)
        self.text = apply(text, entries, PHONEME_FORMATS.get(self.spec.phonemes))
        self.language = _str(body, "language") or None
        self.delivery = _str(body, "delivery") or None
        if not self.spec.delivery:
            self.delivery = None
        seed = body.get("seed")
        self.seed = int(seed) if isinstance(seed, (int, float)) else self.voice.seed
        self.tuning = line_tuning(cfg, self.engine_id, self.spec, body)
        self.effective_tuning = {**default_tuning(cfg, self.engine_id, self.spec), **self.tuning}
        self.key = synthesis_key(cfg, self.engine_id, self.spec, self.voice, self.text, self.language,
                                 self.delivery, self.seed, self.tuning)

    def worker_payload(self, text: str, whole: bool = False) -> dict:
        return {
            "text": text,
            "voice": self.voice.worker_spec(),
            "language": self.language,
            "seed": self.seed,
            "delivery": self.delivery,
            # The settings as they apply to this line (the site's defaults, or the line's own): the engine read its
            # options when it loaded, before a default may have changed in Settings.
            **self.effective_tuning,
            # The client takes the line whole (compressed): the engine may make it whole and check it first.
            "whole": whole,
        }


def _urgency(app: web.Application, body: dict):
    """How soon this passage is needed, for the order the PC makes speech in (engines.manager.RankedLock). A
    reading page sends {reader, seq, playing}: the rank is how many passages after the one it plays now this one is,
    and it says where it plays with every request, so its waiting passages get more urgent as it moves on. Two people
    reading then get their next passages in turn, before either one's read-ahead. Without them (a sample, a word) it
    is wanted now: rank 0."""
    reader = _str(body, "reader")
    try:
        seq, playing = int(body["seq"]), int(body["playing"])
    except (KeyError, TypeError, ValueError):
        return None
    if not reader:
        return None
    readers = app[READERS_KEY]
    now = time.monotonic()
    readers[reader] = (playing, now)
    if len(readers) > 200:  # pages long gone
        for r in [r for r, (_, t) in readers.items() if now - t > 3600]:
            readers.pop(r, None)
    return lambda: seq - readers.get(reader, (playing, now))[0]


async def tts(request: web.Request) -> web.StreamResponse:
    """Synthesize one utterance. By default raw PCM16LE mono is streamed back as it is made; with
    {"format": "opus"|"mp3"} the whole utterance comes back compressed (Ogg Opus 64 kbit/s or MP3), about a
    sixth of the size (PCM if ffmpeg can't encode it; the Content-Type says which).

    Errors before any audio is produced are JSON with a 4xx/5xx status. An
    error after audio has started aborts the connection, so the client sees a
    truncated chunked body rather than a clean end. Closing the connection
    cancels synthesis.
    """
    app = request.app
    cfg = app[CONFIG_KEY]
    body = await _json_body(request)
    from .activity import ACTIVITY_KEY

    app[ACTIVITY_KEY].touch()  # someone is reading or listening
    rank = _urgency(app, body)  # also notes where the reader is, cached passage or not
    text = clean_text(str(body.get("text") or ""))
    if len(text) > MAX_TEXT_CHARS:
        return _json_error(413, f"Text longer than {MAX_TEXT_CHARS} characters")
    if not is_speakable(text):
        return web.Response(body=b"", headers=_audio_headers(24000))
    try:
        plan = SynthesisPlan(app, body, text)
    except EngineError as exc:
        return _json_error(exc.status, exc.detail)

    speed = clamp_speed(body.get("speed"))
    pause_ms = trailing_pause_ms(text, clamp_pause_ms(body.get("sentence_pause_ms"), cfg.sentence_pause_ms))
    use_cache = body.get("cache", True) is not False
    meta = {"X-Engine": plan.engine_id, "X-Voice": plan.voice.id, "X-Pronunciation-Version": plan.pron_version}

    fmt = _str(body, "format").lower()
    if fmt in ("", "pcm", "l16"):
        fmt = ""
    elif fmt not in AUDIO_FORMATS:
        return _json_error(400, f"format must be pcm, {', '.join(AUDIO_FORMATS)}")

    async def encoded(pcm: bytes, rate: int, cache_state: str) -> web.Response:
        """The whole utterance, compressed (or PCM when ffmpeg can't encode it)."""
        headers = {**meta, "X-Cache": cache_state}
        if fmt and pcm:
            try:
                data, content_type = await encode_audio(pcm, rate, fmt)
                return web.Response(body=data, headers={"Content-Type": content_type, "Cache-Control": "no-store",
                                                        "X-Sample-Rate": str(rate), "X-Encoding": fmt, **headers})
            except EncodeError as exc:
                logger.warning("sending PCM: %s", exc)
        return web.Response(body=pcm, headers=_audio_headers(rate, **headers))

    book_id = _str(body, "book")

    def finish(pcm: bytes, rate: int) -> bytes:
        """A whole clip as it is kept and served: no dead air, and at the common loudness."""
        if cfg.trim_silence:
            pcm = tighten(pcm, rate)
        if cfg.level_speech:
            pcm = normalize(pcm, rate, plan.delivery)
        return pcm

    cached = app[CACHE_KEY].get(plan.key) if use_cache else None
    source = "hit"
    if cached is None and use_cache and book_id:
        from .bookfile import stored_clip

        # The cache keeps a few GB; every clip made for a book is also kept in its book file.
        cached = await asyncio.to_thread(stored_clip, app, book_id, plan.key)
        if cached is not None:
            app[CACHE_KEY].put(plan.key, *cached)
            source = "file"
    if cached is not None:
        rate, pcm = cached
        pcm = finish(pcm, rate)  # clips cached before trimming and levelling existed
        pcm = time_stretch(pcm, rate, speed) + bytes(int(rate * pause_ms / 1000) * 2)
        return await encoded(pcm, rate, source)

    manager = app[ENGINES_KEY]
    if _str(body, "reader"):  # someone is listening to speech that has to be made: they count as a reader
        app[ACTIVITY_KEY].making(_str(body, "reader"))
    t0 = time.perf_counter()
    first_audio: float | None = None
    resp: web.StreamResponse | None = None  # streaming PCM; compressed audio is sent whole at the end
    started = False
    rate = 24000
    trimmer: SilenceTrimmer | None = None
    full = bytearray()  # the utterance as cached (trimmed, normal speed, no pause)
    out = bytearray()  # what a compressed response will contain
    sent = 0
    complete = False

    async def emit(data: bytes) -> None:
        nonlocal first_audio, sent
        if first_audio is None:
            first_audio = time.perf_counter() - t0
        sent += len(data)
        if resp is not None:
            await resp.write(data)
        else:
            out.extend(data)

    try:
        for piece in split_for_synthesis(plan.text, chunk_limit(plan.spec)):
            stream = manager.synthesize(plan.engine_id, plan.worker_payload(piece, whole=bool(fmt)), rank=rank)
            piece_pcm = bytearray()
            async with contextlib.aclosing(stream):
                async for ev in stream:
                    if isinstance(ev, StreamStart):
                        rate = ev.sample_rate
                        if trimmer is None and cfg.trim_silence:
                            trimmer = SilenceTrimmer(rate)
                        if not started:
                            started = True
                            if not fmt:
                                resp = web.StreamResponse(headers=_audio_headers(rate, **meta, **{"X-Cache": "miss"}))
                                resp.enable_chunked_encoding()
                                await resp.prepare(request)
                    elif isinstance(ev, AudioChunk) and started:
                        pcm = trimmer.feed(ev.pcm) if trimmer else ev.pcm
                        if not pcm:
                            continue
                        full += pcm
                        if speed == 1.0:
                            await emit(pcm)
                        else:
                            piece_pcm += pcm
            if piece_pcm and started:
                await emit(time_stretch(bytes(piece_pcm), rate, speed))
        if trimmer is not None and started:
            tail = trimmer.flush()
            if tail:
                full += tail
                await emit(time_stretch(tail, rate, speed) if speed != 1.0 else tail)
        complete = True
    except EngineError as exc:
        if resp is None:
            logger.warning("tts failed: %s", exc.detail)
            return _json_error(exc.status, exc.detail)
        logger.warning("tts failed mid-stream, aborting: %s", exc.detail)
        if request.transport is not None:
            request.transport.close()
        return resp
    except (ConnectionResetError, aiohttp.ClientConnectionResetError):
        logger.info("client cancelled after %d bytes", sent)
        return resp or web.Response(status=499)
    except asyncio.CancelledError:
        logger.info("client cancelled after %d bytes", sent)
        raise

    if complete and full and use_cache:
        full = bytearray(finish(bytes(full), rate))
        app[CACHE_KEY].put(plan.key, rate, bytes(full))
        if book_id:
            from .bookfile import mark_dirty

            mark_dirty(app, book_id)
    silence = bytes(int(rate * pause_ms / 1000) * 2) if sent and pause_ms else b""
    audio_s = len(full) / 2 / rate
    total = time.perf_counter() - t0
    logger.info(
        "tts %s/%s %d chars -> %.1fs audio, first audio %s, total %.2fs (RTF %.2f)%s",
        plan.engine_id, plan.voice.id, len(text), audio_s,
        f"{first_audio * 1000:.0f}ms" if first_audio is not None else "n/a", total,
        total / audio_s if audio_s else 0, f", sent as {fmt}" if fmt else "",
    )
    if resp is None:
        body = finish(bytes(out), rate)  # whole-utterance answers: no dead air, at the common loudness
        return await encoded(body + silence, rate, "miss")
    if silence:
        await resp.write(silence)
    await resp.write_eof()
    return resp


# ---------------------------------------------------------------- addresses


def server_urls(port: int, extra: list[str] | None = None) -> list[str]:
    """URLs other devices can open: this PC's addresses (private ones first), then any configured extra
    URLs, e.g. the PC's address or name on your own VPN."""
    addrs = lan_addresses()
    urls = [f"http://{a['ip']}:{port}" for a in addrs if a["kind"] == "lan"]
    urls += [f"http://{a['ip']}:{port}" for a in addrs if a["kind"] != "lan"]
    urls += [u.rstrip("/") for u in extra or [] if u.strip()]
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def _qr_svg(data: str) -> str:
    import qrcode
    import qrcode.image.svg

    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=6, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8")


def viewer_url(base: str, token: str) -> str:
    return f"{base}/books#token={quote(token, safe='')}"


async def index(request: web.Request) -> web.Response:
    """The PC's start page: the Users page, where each user's link and QR code open the library on their devices."""
    raise web.HTTPFound("/books/users")


def viewer_version() -> str:
    """When viewer.html last changed: open pages compare it with their own to notice the site was updated."""
    path = Path(__file__).parent / "web" / "viewer.html"
    return time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))


async def viewer_page(request: web.Request) -> web.Response:
    """Book viewer. Only loopback requests get the token embedded; other devices supply it."""
    path = Path(__file__).parent / "web" / "viewer.html"
    page = path.read_text(encoding="utf-8")
    token = request.app[CONFIG_KEY].token if request.remote in _LOOPBACK else None
    page = page.replace("__TOKEN__", json.dumps(token))
    # The page's version, so a device's reports show whether it has reloaded since, and it can offer a reload.
    page = page.replace("__VERSION__", json.dumps(viewer_version()))
    return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})


async def client_log(request: web.Request) -> web.Response:
    """What a reading device saw (after a chapter change, or a script error), for tracking down device-only bugs."""
    body = await _json_body(request)
    logger.info("client %s: %s", request.remote, json.dumps(body, ensure_ascii=False)[:1500])
    return web.json_response({"ok": True})


# ---------------------------------------------------------------- app


def make_app(
    cfg: ServerConfig,
    *,
    specs: dict[str, EngineSpec] | None = None,
    manager: EngineManager | None = None,
) -> web.Application:
    data = cfg.data_path
    app = web.Application(middlewares=[auth_middleware], client_max_size=20 * 1024 * 1024)
    app[CONFIG_KEY] = cfg
    app[READERS_KEY] = {}
    app[ENGINES_KEY] = manager or EngineManager(
        specs or ENGINES,
        cfg.models_path,
        options=cfg.engine_options,
        load_timeout_s=cfg.load_timeout_s,
        synth_timeout_s=cfg.synth_timeout_s,
        designer_idle_s=cfg.designer_idle_s,
        park_speech=cfg.park_speech_model,
        load_times_path=cfg.data_path / "load-times.json",
    )
    app[ENGINES_KEY].resident = app[ENGINES_KEY].resident or cfg.default_engine
    app[ENGINES_KEY].narration = cfg.narration_engine or None
    from .activity import ACTIVITY_KEY, Activity

    app[ACTIVITY_KEY] = Activity(cfg.idle_after_s, cfg.design_pause_s)
    app[VOICES_KEY] = VoiceStore(data / "voices")
    app[PRON_KEY] = PronunciationStore(data / "pronunciations.json")
    app[CACHE_KEY] = AudioCache(data / "cache", cfg.cache_max_mb * 1024 * 1024)
    app[EXTRA_KEY] = {}

    async def _lifecycle(app: web.Application):
        task = None
        if cfg.preload and cfg.default_engine in app[ENGINES_KEY].specs:
            async def _preload():
                try:
                    await app[ENGINES_KEY].preload(cfg.default_engine)
                except EngineError as exc:
                    logger.warning("could not preload %s: %s", cfg.default_engine, exc.detail)
                if cfg.narration_engine:
                    try:
                        await app[ENGINES_KEY].preload_narration(cfg.narration_engine)
                    except EngineError as exc:
                        logger.warning("could not preload %s: %s", cfg.narration_engine, exc.detail)
            task = asyncio.ensure_future(_preload())
        yield
        if task is not None:
            task.cancel()
        await app[ENGINES_KEY].close()

    app.cleanup_ctx.append(_lifecycle)
    r = app.router
    r.add_get("/", index)
    r.add_get("/books", viewer_page)
    r.add_get("/books/{book}", viewer_page)
    r.add_get("/books/{book}/settings", viewer_page)
    r.add_get("/books/{book}/characters/{character}", viewer_page)
    r.add_get("/api/ping", ping)
    r.add_get("/api/health", health)
    r.add_get("/api/engines", engines)
    r.add_get("/api/settings", settings_get)
    r.add_put("/api/settings", settings_put)
    from .activity import activity_route

    r.add_post("/api/activity", activity_route)
    r.add_post("/api/client-log", client_log)
    r.add_post("/api/engines/unload", engine_unload)
    r.add_post("/api/engines/{engine}/load", engine_load)
    r.add_get("/api/voices", voices)
    r.add_post("/api/voices", voice_create)
    r.add_post("/api/voices/design", voice_design)
    r.add_delete("/api/voices/{voice}", voice_delete)
    r.add_get("/api/voices/{voice}/audio", voice_audio)
    r.add_get("/api/pronunciations", pronunciations)
    r.add_post("/api/pronunciations", pronunciation_add)
    r.add_patch("/api/pronunciations/{entry}", pronunciation_update)
    r.add_delete("/api/pronunciations/{entry}", pronunciation_delete)
    r.add_post("/api/tts", tts)

    from . import books, voice_edit

    voice_edit.setup(app)
    books.setup(app)
    return app
