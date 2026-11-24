"""Voice editor API: edit, redesign, replace the reference of, duplicate and undo voices.

Every change keeps the previous version (VoiceStore history), and every change that
affects the sound changes the voice's fingerprint, which is part of the audio cache
key, so cached sentences in the old voice are never reused.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import secrets
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from aiohttp import web

from .app import (CONFIG_KEY, DESIGN_TEXT, ENGINES_KEY, VOICES_KEY, _json_body, _json_error, _str,
                  collect_audio, design_clip)
from .audio import SilenceTrimmer, wav_bytes
from .engines.manager import EngineError
from .voices import Voice, VoiceStore

logger = logging.getLogger("elvenreader.voices")

MAX_REF_SECONDS = 60
CANDIDATE_TTL_S = 24 * 3600
_CANDIDATE_RE = r"[0-9a-f]{12}"


class AudioError(ValueError):
    pass


# ---------------------------------------------------------------- audio


def to_wav(data: bytes, start: float | None = None, end: float | None = None) -> bytes:
    """Any audio clip (WAV, MP3, a browser recording...) -> mono 16-bit WAV, trimmed to [start, end] s."""
    pcm, rate = None, 0
    if data.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(data)) as w:
                if w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getcomptype() == "NONE":
                    rate, pcm = w.getframerate(), w.readframes(w.getnframes())
        except (wave.Error, EOFError):
            pass
    if pcm is None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise AudioError("Only mono 16-bit WAV can be read here (ffmpeg is not installed)")
        # From a file, not a pipe: phones' recordings (.m4a, .mp4) keep their index at the end of the file, which
        # ffmpeg can only reach by seeking ("partial file" from a pipe once the clip is longer than a few seconds).
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "upload"
            src.write_bytes(data)
            proc = subprocess.run([ffmpeg, "-v", "error", "-i", str(src), "-ac", "1", "-ar", "24000", "-f", "s16le",
                                   "pipe:1"], capture_output=True, timeout=120,
                                  creationflags=0x08000000 if os.name == "nt" else 0)
        if proc.returncode != 0 or not proc.stdout:
            logger.warning("could not read an uploaded clip (%d bytes): %s", len(data),
                           proc.stderr.decode("utf-8", "replace")[-300:])
            raise AudioError("Could not read that audio file")
        pcm, rate = proc.stdout, 24000
    n = len(pcm) // 2
    a = 0 if start is None else min(n, max(0, int(start * rate)))
    b = n if end is None else min(n, max(0, int(end * rate)))
    if b - a < rate // 2:
        raise AudioError("The clip is shorter than half a second")
    if (b - a) / rate > MAX_REF_SECONDS:
        raise AudioError(f"Keep the reference under {MAX_REF_SECONDS} seconds (trim it); 5-20 s works best")
    return wav_bytes(pcm[2 * a:2 * b], rate)


# ---------------------------------------------------------------- transcription


class TranscribeError(Exception):
    pass


class Transcriber:
    """Runs Whisper in an engine venv (one clip at a time)."""

    def __init__(self, python: Path, model: Path):
        self.python, self.model = Path(python), Path(model)
        self._lock = asyncio.Lock()
        # Replaceable in tests: async (wav bytes, language) -> text.
        self.impl = self._run

    @property
    def available(self) -> bool:
        return self.python.exists() and self.model.exists()

    async def __call__(self, wav: bytes, language: str = "") -> str:
        return await self.impl(wav, language)

    async def _run(self, wav: bytes, language: str) -> str:
        if not self.available:
            raise TranscribeError(f"Whisper is not installed ({self.model}); type the transcript")
        async with self._lock:
            fd, path = tempfile.mkstemp(suffix=".wav")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(wav)
                env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                proc = await asyncio.create_subprocess_exec(
                    str(self.python), str(Path(__file__).with_name("transcribe_worker.py")), str(self.model), path,
                    language or "", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), 600)
                except asyncio.TimeoutError:
                    proc.kill()
                    raise TranscribeError("Transcription timed out") from None
            finally:
                Path(path).unlink(missing_ok=True)
        lines = [ln for ln in out.decode("utf-8", "replace").splitlines() if ln.startswith("{")]
        if proc.returncode != 0 or not lines:
            logger.warning("whisper failed: %s", err.decode("utf-8", "replace")[-2000:])
            raise TranscribeError("Transcription failed; type the transcript")
        return json.loads(lines[-1])["text"]


TRANSCRIBE_KEY = web.AppKey("transcriber", Transcriber)


# ---------------------------------------------------------------- usage


def voice_usage(app: web.Application) -> dict[str, list[dict]]:
    """voice id -> who uses it: characters in analysed books, and the default reader voice."""
    from .dramatize import DRAMA_STORE_KEY
    from .series import book_series

    out: dict[str, list[dict]] = {}
    store = app.get(DRAMA_STORE_KEY)
    if store is not None:
        for book_id in store.ids():
            title = (store.summary(book_id) or {}).get("title", "")
            series = book_series(app, book_id)
            for cid, c in store.script(book_id).get("characters", {}).items():
                if c.get("voice"):
                    out.setdefault(c["voice"], []).append(
                        {"book": book_id, "title": title, "series": series, "character": cid,
                         "name": c.get("name", cid), "aliases": c.get("aliases", []), "role": c.get("role")})
    default = app[CONFIG_KEY].default_voice
    if default:
        out.setdefault(default, []).append({"default": True, "name": "Default reader voice"})
    return out


def details(app: web.Application, voice: Voice, usage: dict | None = None) -> dict:
    usage = voice_usage(app) if usage is None else usage
    return {**voice.details(), "used_by": usage.get(voice.id, []),
            "history": app[VOICES_KEY].history(voice.id)}


def _voice_or_404(request: web.Request) -> Voice:
    voice = request.app[VOICES_KEY].get(request.match_info["voice"])
    if voice is None:
        raise web.HTTPNotFound(text='{"error": "No such voice"}', content_type="application/json")
    return voice


# ---------------------------------------------------------------- routes


async def voice_get(request: web.Request) -> web.Response:
    return web.json_response(details(request.app, _voice_or_404(request)))


async def voice_patch(request: web.Request) -> web.Response:
    voice = _voice_or_404(request)
    body = await _json_body(request)
    before = voice
    try:
        voice = request.app[VOICES_KEY].update(voice.id, **body)
    except ValueError as exc:
        return _json_error(400, str(exc))
    await _characters_follow(request.app, voice.id)
    _report_voice_edit(request.app, "details", before, voice)
    return web.json_response(details(request.app, voice))


async def _characters_follow(app: web.Application, voice_id: str) -> None:
    from .dramatize import sync_voice_descriptions

    await sync_voice_descriptions(app, voice_id)


def _voice_view(voice: Voice | None) -> dict | None:
    if voice is None:
        return None
    return {k: v for k, v in (("id", voice.id), ("name", voice.name), ("description", voice.description or voice.instruct),
                              ("seed", voice.seed), ("audio", (voice.audio_sha or "")[:12])) if v not in (None, "")}


def _report_voice_edit(app: web.Application, edit: str, before: Voice | None, after: Voice | None,
                       only: tuple[str, str] | None = None) -> None:
    """A voice-editor change is a fix to the casting: kept as a "voice" report in every book whose characters use
    the voice (or for one character, `only` = (book, character))."""
    from . import reports

    if after is not None and before is not None and _voice_view(before) == _voice_view(after):
        return  # nothing changed
    users = [(only[0], only[1])] if only else [
        (u["book"], u["character"]) for u in voice_usage(app).get((after or before).id, []) if u.get("book")]
    for book, character in users:
        try:
            reports.record(app, book, "voice", character=character, edit=edit,
                           before=_voice_view(before), after=_voice_view(after))
        except (OSError, ValueError):
            continue


async def _read_audio_form(request: web.Request) -> tuple[bytes | None, dict[str, str]]:
    fields: dict[str, str] = {}
    audio = None
    if request.content_type.startswith("multipart/"):
        async for part in await request.multipart():
            if part.name == "audio":
                audio = await part.read(decode=False)
            elif part.name:
                fields[part.name] = (await part.text()).strip()
    else:
        audio = await request.read()
        fields = {k: v for k, v in request.query.items()}
    return audio or None, fields


def _seconds(fields: dict, key: str) -> float | None:
    raw = fields.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise AudioError(f"{key} must be a number of seconds") from None


async def transcribe(request: web.Request) -> web.Response:
    audio, fields = await _read_audio_form(request)
    if not audio:
        return _json_error(400, "Missing audio")
    try:
        wav = await asyncio.to_thread(to_wav, audio, _seconds(fields, "start"), _seconds(fields, "end"))
        text = await request.app[TRANSCRIBE_KEY](wav, fields.get("language", ""))
    except AudioError as exc:
        return _json_error(400, str(exc))
    except TranscribeError as exc:
        return _json_error(503, str(exc))
    return web.json_response({"text": text})


async def voice_put_audio(request: web.Request) -> web.Response:
    """New reference audio: a file in any format, optional start/end trim, transcript (or Whisper)."""
    voice = _voice_or_404(request)
    audio, fields = await _read_audio_form(request)
    if not audio:
        return _json_error(400, "Missing audio")
    try:
        wav = await asyncio.to_thread(to_wav, audio, _seconds(fields, "start"), _seconds(fields, "end"))
        ref_text = fields.get("ref_text", "")
        if not ref_text:
            ref_text = await request.app[TRANSCRIBE_KEY](wav, fields.get("language") or voice.language)
    except AudioError as exc:
        return _json_error(400, str(exc))
    except TranscribeError as exc:
        return _json_error(503, str(exc))
    meta = {"language": fields["language"]} if fields.get("language") else {}
    before = voice
    voice = request.app[VOICES_KEY].replace_audio(voice.id, wav, ref_text, recorded=True, **meta)
    _report_voice_edit(request.app, "new reference audio", before, voice)
    return web.json_response(details(request.app, voice))


async def voice_duplicate(request: web.Request) -> web.Response:
    """Copy a voice; with book + character, the copy is assigned to that character only."""
    from .dramatize import DRAMA_STORE_KEY

    voice = _voice_or_404(request)
    body = await _json_body(request) if request.can_read_body else {}
    book, character = _str(body, "book"), _str(body, "character")
    from .permissions import can_book, can_voice

    if not (can_book(request, book) if book and character else can_voice(request, voice.id)):
        return _json_error(403, "That book isn't in your library")
    store = request.app.get(DRAMA_STORE_KEY)
    if book and character:
        if store is None or character not in store.script(book).get("characters", {}):
            return _json_error(404, "No such character")
    name = _str(body, "name")
    if not name and book and character:
        name = store.script(book)["characters"][character].get("name", character)
    copy = request.app[VOICES_KEY].duplicate(voice.id, name)
    if book and character:
        from .series import propagate_voice, remember

        async with store.lock(book):
            s = store.script(book)
            s["characters"][character]["voice"] = copy.id
            s["characters"][character]["voice_user"] = True
            store.save_script(book, s)
        # The same character in the series' other volumes gets the copy too: they are one character.
        await propagate_voice(request.app, book, character, voice.id, copy.id, may=lambda b: can_book(request, b))
        remember(request.app, book, only=character)
        await _characters_follow(request.app, copy.id)
        _report_voice_edit(request.app, "own copy of a shared voice", voice, copy, only=(book, character))
    return web.json_response(details(request.app, copy), status=201)


async def voice_history_audio(request: web.Request) -> web.StreamResponse:
    """GET /api/voices/{voice}/history/{stamp}/audio: an earlier version's reference clip."""
    wav = request.app[VOICES_KEY].history_wav(request.match_info["voice"], request.match_info["stamp"])
    if wav is None:
        return _json_error(404, "No such version")
    return web.FileResponse(wav, headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"})


async def voice_restore(request: web.Request) -> web.Response:
    """POST /api/voices/{voice}/history/{stamp}/restore: bring back an earlier version (undoable)."""
    voice = _voice_or_404(request)
    restored = request.app[VOICES_KEY].restore(voice.id, request.match_info["stamp"])
    if restored is None:
        return _json_error(404, "No such version")
    await _characters_follow(request.app, voice.id)
    _report_voice_edit(request.app, "an earlier version brought back", voice, restored)
    return web.json_response(details(request.app, restored))


PAST_VOICES = 3  # shown in the voice editor


async def character_past_voices(request: web.Request) -> web.Response:
    """GET /api/books/{book}/characters/{character}/past-voices: the voices a character had before, newest first (at
    most PAST_VOICES): earlier versions of their current voice (kept for undo), and other voices they were given
    (from the book's reports of voice changes) that still exist. Copies of the current audio are left out."""
    from . import reports
    from .dramatize import DRAMA_STORE_KEY, is_pool

    store = request.app[DRAMA_STORE_KEY]
    book, cid = request.match_info["book"], request.match_info["character"]
    try:
        c = store.script(book).get("characters", {}).get(cid)
    except ValueError as exc:
        return _json_error(400, str(exc))
    if c is None:
        return _json_error(404, "No such character")
    voices = request.app[VOICES_KEY]
    current = voices.get(c.get("voice"))
    seen = {(current.id, current.audio_sha[:12])} if current else set()
    items = []
    if current and not is_pool(current.id):  # a generic voice's versions aren't this character's
        for v in voices.history(current.id):
            items.append({"kind": "version", "voice": current.id, "stamp": v["stamp"], "t": v["saved"],
                          "name": v["name"], "description": v["description"], "key": (current.id, v["audio"])})
    for r in reports.read(request.app, book):
        if r.get("type") != "voice" or r.get("character") != cid:
            continue
        before = r.get("before") or {}
        vid = before.get("voice") or before.get("id")
        if not vid or vid == (current.id if current else None) or not (v := voices.get(vid)):
            continue
        items.append({"kind": "voice", "voice": vid, "t": r.get("t", 0), "name": v.name,
                      "description": v.description or v.instruct, "key": (vid, v.audio_sha[:12])})
    out = []
    for it in sorted(items, key=lambda x: -x["t"]):
        if it["key"] in seen:  # the same audio again (or the voice as it is now)
            continue
        seen.add(it.pop("key"))
        out.append(it)
        if len(out) == PAST_VOICES:
            break
    return web.json_response({"past": out})


def _candidates_dir(app: web.Application) -> Path:
    d = app[CONFIG_KEY].data_path / "voice-candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def voice_design_candidates(request: web.Request) -> web.Response:
    """Design `count` voices from a description (different seeds) to listen to before keeping one.

    {"age": -1 | 1} makes them one age group younger or older: the description's age is moved
    (voice_age.shift_age) and the first option keeps the voice's seed, the closest to the current voice."""
    from .voice_age import shift_age

    voice = _voice_or_404(request)
    body = await _json_body(request)
    from .permissions import can_book, can_voice

    # Options change nothing yet: designing them is for those who may change the voice, or give their character
    # (in `book`) a voice of its own from it.
    if not (can_voice(request, voice.id) or (_str(body, "book") and can_book(request, _str(body, "book")))):
        return _json_error(403, "That voice is also used in books that aren't yours")
    from .activity import touch

    touch(request.app, design=True)  # designing voices: analysis pauses while you do
    description = _str(body, "description") or voice.instruct or voice.description
    if not description:
        return _json_error(400, "Describe the voice, e.g. 'male, elderly, low pitch, british accent'")
    aged = None
    try:
        step = max(-2, min(2, int(body.get("age") or 0)))
    except (TypeError, ValueError):
        return _json_error(400, "age must be -1 (younger) or 1 (older)")
    if step:
        try:
            aged = shift_age(description, step)
        except ValueError as exc:
            return _json_error(400, str(exc))
        description = aged["description"]
    engine = _str(body, "engine") or request.app[CONFIG_KEY].voice_design_engine
    text = _str(body, "text") or DESIGN_TEXT  # every design says the same passage, so they compare by voice alone
    count = max(1, min(4, int(body.get("count") or 3)))
    seeds = [int(s) for s in body.get("seeds") or [] if isinstance(s, (int, float))][:count]
    if step and not seeds and voice.seed is not None:
        seeds = [int(voice.seed)]
    seeds += [random.randrange(100000) for _ in range(count - len(seeds))]
    folder = _candidates_dir(request.app)
    for old in folder.glob("*.json"):
        if time.time() - old.stat().st_mtime > CANDIDATE_TTL_S:
            old.unlink(missing_ok=True)
            old.with_suffix(".wav").unlink(missing_ok=True)
    out = []
    for seed in seeds:
        try:
            rate, pcm, seed = await design_clip(request.app, engine, description, text, seed)
        except EngineError as exc:
            return _json_error(exc.status, exc.detail)
        cid = secrets.token_hex(6)
        (folder / f"{cid}.wav").write_bytes(wav_bytes(pcm, rate))
        meta = {"id": cid, "voice": voice.id, "seed": seed, "description": description, "text": text,
                "engine": engine}
        (folder / f"{cid}.json").write_text(json.dumps(meta), encoding="utf-8")
        out.append(meta)
    return web.json_response({"candidates": out, "description": description,
                              **({"age_from": aged["from"], "age_to": aged["to"]} if aged else {})})


def _candidate(request: web.Request) -> tuple[Path, dict] | None:
    path = _candidates_dir(request.app) / f"{request.match_info['candidate']}.json"
    if not path.exists():
        return None
    return path, json.loads(path.read_text(encoding="utf-8"))


async def candidate_audio(request: web.Request) -> web.StreamResponse:
    found = _candidate(request)
    if found is None:
        return _json_error(404, "No such candidate (they are kept for a day)")
    return web.FileResponse(found[0].with_suffix(".wav"), headers={"Content-Type": "audio/wav"})


async def voice_select_candidate(request: web.Request) -> web.Response:
    """Keep a designed candidate: it becomes the voice's locked reference. The candidate may have been
    made from another voice (a shared voice the user then chose to copy for one character)."""
    voice = _voice_or_404(request)
    found = _candidate(request)
    if found is None:
        return _json_error(404, "No such option (they are kept for a day)")
    path, meta = found
    made_for = {voice.id, meta.get("voice")}
    before = voice
    voice = request.app[VOICES_KEY].replace_audio(
        voice.id, path.with_suffix(".wav").read_bytes(), meta["text"], instruct=meta["description"],
        description=meta["description"], seed=meta["seed"])
    await _characters_follow(request.app, voice.id)
    _report_voice_edit(request.app, "redesigned", before, voice)
    for p in _candidates_dir(request.app).glob("*.json"):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("voice") in made_for:
                p.unlink(missing_ok=True)
                p.with_suffix(".wav").unlink(missing_ok=True)
        except (OSError, ValueError):
            continue
    return web.json_response(details(request.app, voice))


def listing_details(app: web.Application, store: VoiceStore) -> list[dict]:
    usage = voice_usage(app)
    return [details(app, v, usage) for v in store.list()]


def setup(app: web.Application) -> None:
    cfg = app[CONFIG_KEY]
    models = cfg.models_path
    app[TRANSCRIBE_KEY] = Transcriber(models / "venvs" / cfg.whisper_venv / "Scripts" / "python.exe",
                                      models / cfg.whisper_model)
    r = app.router
    r.add_post("/api/transcribe", transcribe)
    r.add_get("/api/voices/{voice}", voice_get)
    r.add_patch("/api/voices/{voice}", voice_patch)
    r.add_put("/api/voices/{voice}/audio", voice_put_audio)
    r.add_post("/api/voices/{voice}/duplicate", voice_duplicate)
    r.add_get("/api/voices/{voice}/history/{stamp:\\d+}/audio", voice_history_audio)
    r.add_post("/api/voices/{voice}/history/{stamp:\\d+}/restore", voice_restore)
    r.add_get("/api/books/{book}/characters/{character}/past-voices", character_past_voices)
    r.add_post("/api/voices/{voice}/design", voice_design_candidates)
    r.add_post("/api/voices/{voice}/design/{candidate:" + _CANDIDATE_RE + "}", voice_select_candidate)
    r.add_get("/api/voice-candidates/{candidate:" + _CANDIDATE_RE + "}/audio", candidate_audio)
