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
                  collect_audio)
from .audio import SilenceTrimmer, wav_bytes
from .engines.manager import EngineError
from .voices import Voice, VoiceStore

logger = logging.getLogger("androidreader.voices")

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
        proc = subprocess.run([ffmpeg, "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "24000", "-f", "s16le",
                               "pipe:1"], input=data, capture_output=True, timeout=120)
        if proc.returncode != 0 or not proc.stdout:
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

    out: dict[str, list[dict]] = {}
    store = app.get(DRAMA_STORE_KEY)
    if store is not None:
        for book_id in store.ids():
            title = (store.summary(book_id) or {}).get("title", "")
            for cid, c in store.script(book_id).get("characters", {}).items():
                if c.get("voice"):
                    out.setdefault(c["voice"], []).append(
                        {"book": book_id, "title": title, "character": cid, "name": c.get("name", cid)})
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
    try:
        voice = request.app[VOICES_KEY].update(voice.id, **body)
    except ValueError as exc:
        return _json_error(400, str(exc))
    return web.json_response(details(request.app, voice))


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
    voice = request.app[VOICES_KEY].replace_audio(voice.id, wav, ref_text, **meta)
    return web.json_response(details(request.app, voice))


async def voice_duplicate(request: web.Request) -> web.Response:
    """Copy a voice; with book + character, the copy is assigned to that character only."""
    from .dramatize import DRAMA_STORE_KEY

    voice = _voice_or_404(request)
    body = await _json_body(request) if request.can_read_body else {}
    book, character = _str(body, "book"), _str(body, "character")
    store = request.app.get(DRAMA_STORE_KEY)
    if book and character:
        if store is None or character not in store.script(book).get("characters", {}):
            return _json_error(404, "No such character")
    name = _str(body, "name")
    if not name and book and character:
        name = f"{store.script(book)['characters'][character].get('name', character)} ({voice.name})"
    copy = request.app[VOICES_KEY].duplicate(voice.id, name)
    if book and character:
        async with store.lock(book):
            s = store.script(book)
            s["characters"][character]["voice"] = copy.id
            store.save_script(book, s)
    return web.json_response(details(request.app, copy), status=201)


async def voice_revert(request: web.Request) -> web.Response:
    voice = _voice_or_404(request)
    reverted = request.app[VOICES_KEY].revert(voice.id)
    if reverted is None:
        return _json_error(409, "No earlier version to go back to")
    return web.json_response(details(request.app, reverted))


def _candidates_dir(app: web.Application) -> Path:
    d = app[CONFIG_KEY].data_path / "voice-candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def voice_design_candidates(request: web.Request) -> web.Response:
    """Design `count` voices from a description (different seeds) to listen to before keeping one."""
    voice = _voice_or_404(request)
    body = await _json_body(request)
    description = _str(body, "description") or voice.instruct or voice.description
    if not description:
        return _json_error(400, "Describe the voice, e.g. 'male, elderly, low pitch, british accent'")
    engine = _str(body, "engine") or request.app[CONFIG_KEY].voice_design_engine
    text = _str(body, "text") or voice.ref_text or DESIGN_TEXT
    count = max(1, min(4, int(body.get("count") or 3)))
    seeds = [int(s) for s in body.get("seeds") or [] if isinstance(s, (int, float))][:count]
    seeds += [random.randrange(100000) for _ in range(count - len(seeds))]
    folder = _candidates_dir(request.app)
    for old in folder.glob("*.json"):
        if time.time() - old.stat().st_mtime > CANDIDATE_TTL_S:
            old.unlink(missing_ok=True)
            old.with_suffix(".wav").unlink(missing_ok=True)
    out = []
    for seed in seeds:
        try:
            rate, pcm = await collect_audio(request.app[ENGINES_KEY].design(engine, description, text, seed))
        except EngineError as exc:
            return _json_error(exc.status, exc.detail)
        if not pcm:
            return _json_error(502, "Voice design produced no audio")
        cid = secrets.token_hex(6)
        (folder / f"{cid}.wav").write_bytes(wav_bytes(SilenceTrimmer(rate).feed(pcm), rate))
        meta = {"id": cid, "voice": voice.id, "seed": seed, "description": description, "text": text,
                "engine": engine}
        (folder / f"{cid}.json").write_text(json.dumps(meta), encoding="utf-8")
        out.append(meta)
    return web.json_response({"candidates": out})


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
    """Keep a designed candidate: it becomes the voice's locked reference."""
    voice = _voice_or_404(request)
    found = _candidate(request)
    if found is None or found[1].get("voice") != voice.id:
        return _json_error(404, "No such candidate for this voice")
    path, meta = found
    voice = request.app[VOICES_KEY].replace_audio(
        voice.id, path.with_suffix(".wav").read_bytes(), meta["text"], instruct=meta["description"],
        description=meta["description"], seed=meta["seed"])
    for p in _candidates_dir(request.app).glob("*.json"):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("voice") == voice.id:
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
    r.add_post("/api/voices/{voice}/revert", voice_revert)
    r.add_post("/api/voices/{voice}/design", voice_design_candidates)
    r.add_post("/api/voices/{voice}/design/{candidate:" + _CANDIDATE_RE + "}", voice_select_candidate)
    r.add_get("/api/voice-candidates/{candidate:" + _CANDIDATE_RE + "}/audio", candidate_audio)
