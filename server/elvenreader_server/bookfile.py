"""Book files (.elvenbook): one file per book holding everything a reader needs, including the speech made so far.

It is a storage format, not something the website downloads: the server keeps `data/books/<id>/book.elvenbook`
up to date in the background, and the file can be copied to another server (POST /api/library/import) or read by
another client. Layout (a zip, written append-only; see zipappend.py):

    mimetype                      "application/x-elvenbook" (stored, first entry)
    manifest.json                 format, format_version, id, title, author, language, series, volume, engine,
                                  segments, voiced, cover, images ({EPUB path: entry}), created, updated
    text/book.json                chapters: title, paragraphs, quotes ([id, paragraph, start, end]),
                                  images ({paragraph: [EPUB paths]}, shown before it), end_images
    images/...                    the book's pictures (manifest.images maps EPUB paths to them), images/cover.jpg
    analysis/script.json          characters (cast: name, aliases, role, profile, voice...), lines
                                  ({quote id: speaker, delivery}), narration ([chapter, paragraph, narrator])
    analysis/segments.jsonl       the spoken units in reading order (see segments())
    analysis/pronunciations.json  the pronunciation entries that apply to the book
    voices/<id>.json, <id>.wav    every voice the book uses: description, reference clip and its transcript
    audio/index.json              {segment id: {entry, duration, rate, voice, engine, key, content}}
    audio/c<chapter>/<id>.opus    speech made so far, Ogg Opus mono, one clip per segment
    source/book.epub              the original EPUB
    x-server/script.json, meta.json   the server's own records, so an import loses nothing; readers ignore x-*

New clips are appended in batches, never one file write per sentence; replaced data stays behind as dead bytes
until the file is compacted (when more than a quarter of it is dead). Readers must ignore unknown keys and files.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from .audio import normalize
from .text import MAX_TEXT_CHARS, clean_text, engine_text, is_speakable, trailing_pause_ms
from .zipappend import AppendZip, ZipFormatError

logger = logging.getLogger("elvenreader.bookfile")

FILE_NAME = "book.elvenbook"
SUFFIX = ".elvenbook"
MIMETYPE = b"application/x-elvenbook"
FORMAT = "elvenbook"
FORMAT_VERSION = 1
OPUS_BITRATE = "32k"  # mono speech; the website gets 64k, encoded on the fly
CLIPS_PER_COMMIT = 300
COMPACT_MIN_DEAD = 16 << 20
COMPACT_DEAD_SHARE = 0.25
# The viewer's pauses after a passage: before the next passage of the same paragraph, a new paragraph, or a new scene
# (the EPUB's scene breaks, book.json "breaks").
SAME_PARAGRAPH_PAUSE_MS, PARAGRAPH_PAUSE_MS, SCENE_BREAK_PAUSE_MS = 120, 450, 1200
# Language names the engines take, by the book's language code (viewer.html: LANGS).
LANGS = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
         "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ru": "Russian", "nl": "Dutch", "pl": "Polish"}
_BUSY = ("queued", "running", "casting")
_WRITING = ("running", "casting")  # a queued book doesn't change until its analysis runs: its file may follow edits


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------- segments


def narrator_at(narration: list, ci: int, pi: int) -> str | None:
    who = None
    for c, p, cid in narration:
        if c < ci or (c == ci and p <= pi):
            who = cid
        else:
            break
    return who


def _pieces(text: str, quotes: list, lines: dict) -> list[tuple]:
    """("n", start, end) narration and ("q", start, end, id) quotation pieces of a paragraph; quoted terms
    inside a sentence (mentions) stay in the narration."""
    out, pos = [], 0
    for qid, _, s, e in quotes:
        line = lines.get(qid)
        if line and line.get("mention") and not line.get("speaker"):
            continue
        if s > pos:
            out.append(("n", pos, s))
        out.append(("q", s, e, qid))
        pos = e
    if pos < len(text):
        out.append(("n", pos, len(text)))
    return out


_TUNING_NAMES = ("guidance", "temperature")  # app.TUNING: a line's own speech settings


def segments(book: dict, public: dict) -> list[dict]:
    """The book as the reader speaks it: one segment per narration piece or quotation, in reading order, each
    with its text, who voices it (a character id; None = the book's default narrator voice), the voice id set
    for that character (None = the default voice), delivery, and the pause to leave after it.

    Must match how viewer.html splits paragraphs (pieces(), renderParagraph()) and asks for speech
    (unitRequest()), or the clips stored here are not the ones the website plays."""
    chars = {c["id"]: c for c in public.get("characters", [])}
    lines = public.get("lines", {})
    narration = public.get("narration") or []
    out: list[dict] = []
    for ci, ch in enumerate(book.get("chapters", [])):
        by_para: dict[int, list] = {}
        for q in ch.get("quotes", []):
            by_para.setdefault(q[1], []).append(q)
        for pi, text in enumerate(ch.get("paragraphs", [])):
            narrator = chars.get(narrator_at(narration, ci, pi))
            n = 0
            for piece in _pieces(text, sorted(by_para.get(pi, []), key=lambda q: q[2]), lines):
                if piece[0] == "n":
                    _, s, e = piece
                    if not any(c.isalnum() for c in text[s:e]):
                        continue
                    out.append({"id": f"c{ci}p{pi}n{n}", "chapter": ci, "paragraph": pi, "start": s, "end": e,
                                "kind": "narration", "text": text[s:e].strip(),
                                "speaker": narrator["id"] if narrator else None,
                                "voice": narrator.get("voice") if narrator else None, "delivery": None})
                    n += 1
                    continue
                _, s, e, qid = piece
                line = lines.get(qid)
                c = chars.get(line["speaker"]) if line and line.get("speaker") else None
                # Quotations given to the narrator (signs, titles) are read in the narration's voice.
                reader = c or (narrator if line else None)
                out.append({"id": qid, "chapter": ci, "paragraph": pi, "start": s, "end": e, "kind": "quote",
                            "text": (line or {}).get("say") or text[s:e], "speaker": reader["id"] if reader else None,
                            "voice": reader.get("voice") if reader else None,
                            "delivery": line.get("delivery") if line else None,
                            **{k: line[k] for k in _TUNING_NAMES if line and line.get(k) is not None}})
    chapters = book.get("chapters", [])
    for i, seg in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        same = nxt is not None and (nxt["chapter"], nxt["paragraph"]) == (seg["chapter"], seg["paragraph"])
        scene = (nxt is not None and not same
                 and nxt["paragraph"] in (chapters[nxt["chapter"]].get("breaks") or []))
        seg["pause_ms"] = trailing_pause_ms(clean_text(seg["text"]), SAME_PARAGRAPH_PAUSE_MS if same
                                            else SCENE_BREAK_PAUSE_MS if scene else PARAGRAPH_PAUSE_MS)
    return out


@dataclass
class Render:
    key: str  # the audio cache key /api/tts uses
    content: str  # words, voice, delivery and language only: the same across speech models
    voice: object
    engine: str


class Renderer:
    """What /api/tts (app.SynthesisPlan) decides for a segment, for a whole book at once."""

    def __init__(self, app: web.Application, book_id: str, book: dict):
        from .app import CONFIG_KEY, ENGINES_KEY, VOICES_KEY, pronunciations_for
        from .pronunciation import PHONEME_FORMATS

        self.cfg = app[CONFIG_KEY]
        self.engine_id = self.cfg.default_engine
        manager = app[ENGINES_KEY]
        self.specs, self.models_dir = manager.specs, manager.models_dir
        self.spec = self.specs.get(self.engine_id)
        self.store = app[VOICES_KEY]
        self._voices: dict = {}
        self.default = self.store.default(self.cfg.default_voice)
        self.entries = pronunciations_for(app, book_id)
        self.formats = PHONEME_FORMATS
        self.language = LANGS.get((book.get("language") or "en")[:2].lower())

    def get(self, voice_id: str | None):
        if voice_id not in self._voices:
            self._voices[voice_id] = self.store.get(voice_id) if voice_id else None
        return self._voices[voice_id]

    def voice(self, voice_id: str | None):
        return self.get(voice_id) or self.default

    def render(self, seg: dict) -> Render | None:
        from .app import engine_for, line_tuning, synthesis_key
        from .pronunciation import apply

        text = clean_text(seg["text"])
        voice = self.voice(seg.get("voice"))
        if self.spec is None or voice is None or not is_speakable(text) or len(text) > MAX_TEXT_CHARS:
            return None
        # Narration may be read by the narration model (app.engine_for), as /api/tts does.
        engine_id = engine_for(self.cfg, self.specs, self.models_dir, seg.get("kind") == "narration")
        spec = self.specs[engine_id]
        final = apply(engine_text(spec, text), self.entries, self.formats.get(spec.phonemes))
        delivery = ((seg.get("delivery") or "").strip() or None) if spec.delivery else None
        tuning = line_tuning(self.cfg, engine_id, spec, seg)
        key = synthesis_key(self.cfg, engine_id, spec, voice, final, self.language, delivery, voice.seed, tuning)
        content = hashlib.sha256(_json_bytes([final, voice.fingerprint, delivery, self.language]
                                             + ([tuning] if tuning else []))).hexdigest()[:16]
        return Render(key, content, voice, engine_id)


# ---------------------------------------------------------------- audio


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")
    r = subprocess.run([ffmpeg, "-v", "error", *args], input=data, capture_output=True, timeout=120,
                       creationflags=_NO_WINDOW)
    if r.returncode or not r.stdout:
        raise RuntimeError("ffmpeg failed: " + r.stderr.decode("utf-8", "replace")[-300:])
    return r.stdout


def encode_opus(pcm: bytes, rate: int) -> bytes:
    return _ffmpeg(["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0", "-c:a", "libopus",
                    "-b:a", OPUS_BITRATE, "-f", "ogg", "pipe:1"], pcm)


def decode_opus(data: bytes, rate: int) -> bytes:
    return _ffmpeg(["-i", "pipe:0", "-f", "s16le", "-ar", str(rate), "-ac", "1", "pipe:1"], data)


# ---------------------------------------------------------------- state


@dataclass
class BookFiles:
    dirty: set = field(default_factory=set)  # books with new speech since their last sync
    seen: dict = field(default_factory=dict)  # book -> its inputs when last synced
    voice_files: dict = field(default_factory=dict)  # book -> the voice files it used
    readers: dict = field(default_factory=dict)  # book -> (file stamp, AppendZip, {cache key: index entry})
    locks: dict = field(default_factory=dict)
    guard: threading.Lock = field(default_factory=threading.Lock)

    def lock(self, book_id: str) -> threading.Lock:
        with self.guard:
            return self.locks.setdefault(book_id, threading.Lock())


BOOKFILES_KEY = web.AppKey("bookfiles", BookFiles)


def _state(app: web.Application) -> BookFiles:
    if BOOKFILES_KEY not in app:
        app[BOOKFILES_KEY] = BookFiles()
    return app[BOOKFILES_KEY]


def mark_dirty(app: web.Application, book_id: str) -> None:
    if BOOKFILES_KEY in app:
        app[BOOKFILES_KEY].dirty.add(book_id)


def file_path(app: web.Application, book_id: str) -> Path:
    from .dramatize import DRAMA_STORE_KEY

    return app[DRAMA_STORE_KEY].dir(book_id) / FILE_NAME


def stored_clip(app: web.Application, book_id: str, key: str) -> tuple[int, bytes] | None:
    """The clip with this cache key from the book's file, as PCM (sample rate, bytes); None if it has none.
    No lock: a commit only appends, so the file as last opened stays readable; after a compaction the
    file's stamp changes and it is opened again (a read caught mid-swap fails its CRC check and gives None)."""
    try:
        path = file_path(app, book_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    st = _state(app)
    try:
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        cached = st.readers.get(book_id)
        if cached is None or cached[0] != stamp:
            z = AppendZip.open(path)
            index = json.loads(z.read("audio/index.json")) if "audio/index.json" in z else {}
            cached = (stamp, z, {v["key"]: v for v in index.values() if v.get("entry") in z})
            st.readers[book_id] = cached
        hit = cached[2].get(key)
        data = cached[1].read(hit["entry"]) if hit else None
    except (OSError, ValueError, KeyError, ZipFormatError) as exc:
        logger.warning("%s: could not read %s: %s", book_id, FILE_NAME, exc)
        return None
    if data is None:
        return None
    try:
        return hit["rate"], decode_opus(data, hit["rate"])
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        logger.warning("%s: could not decode a stored clip: %s", book_id, exc)
        return None


# ---------------------------------------------------------------- sync


def sync(app: web.Application, book_id: str, max_clips: int = CLIPS_PER_COMMIT, keep_audio: bool | None = None) -> dict:
    """Bring the book's file up to date: text, pictures, analysis, voices and every clip of it in the audio
    cache that the file doesn't have yet (at most `max_clips` new ones; "more" says whether some are left).
    Clips whose words, voice or delivery changed are dropped; clips made with other settings (another speech
    model) are kept until new ones replace them. While the book waits for (or is in) an analysis, `keep_audio`
    (the default then) keeps every clip it has: lines not labelled again yet look changed, but mostly come back
    the same (dropping them once lost 843 clips of a book half way through being labelled again)."""
    from .app import CACHE_KEY, CONFIG_KEY
    from .dramatize import DRAMA_STORE_KEY, ScriptStore, picture_refs
    from .series import series_info

    cfg = app[CONFIG_KEY]
    store = app[DRAMA_STORE_KEY]
    st = _state(app)
    with st.lock(book_id):
        book = store.book(book_id)
        if book is None:
            return {"error": "no such book"}
        script = store.script(book_id)
        if keep_audio is None:
            keep_audio = script.get("status") in _BUSY
        public = ScriptStore.public(script)
        d = store.dir(book_id)
        path = d / FILE_NAME
        try:
            z = AppendZip.open(path)
        except ZipFormatError as exc:
            damaged = path.with_name(f"{path.name}.damaged-{int(time.time())}")
            logger.warning("%s: %s is unreadable (%s); kept as %s, starting a new one", book_id, FILE_NAME, exc,
                           damaged.name)
            path.replace(damaged)
            z = AppendZip(path)
        writes: dict[str, tuple[bytes, bool]] = {}

        def put(name: str, data: bytes, compress: bool = True) -> None:
            if not z.same(name, data):
                writes[name] = (data, compress)

        if "mimetype" not in z:
            writes["mimetype"] = (MIMETYPE, False)
        put("text/book.json", _json_bytes(book))
        images = {}
        for ref in picture_refs(book):
            f = store.picture_file(book_id, ref)
            if f is not None and f.exists():
                images[ref] = f"images/{f.name}"
                if images[ref] not in z:
                    writes[images[ref]] = (f.read_bytes(), False)
        cover = store.cover(book_id)
        if cover is not None:
            put("images/cover.jpg", cover.read_bytes(), False)
        epub = d / "book.epub"
        if epub.exists() and (z.member("source/book.epub") is None or z.member("source/book.epub").size != epub.stat().st_size):
            writes["source/book.epub"] = (epub.read_bytes(), False)
        put("analysis/script.json", _json_bytes(public))
        put("x-server/script.json", _json_bytes(script))
        if (d / "meta.json").exists():
            put("x-server/meta.json", (d / "meta.json").read_bytes())
        if (d / "bookmarks.json").exists():  # where each reader is (bookmark.py)
            put("x-server/bookmarks.json", (d / "bookmarks.json").read_bytes())

        renderer = Renderer(app, book_id, book)
        put("analysis/pronunciations.json", _json_bytes([e.to_json() for e in renderer.entries]))
        segs = segments(book, public)
        renders = []
        for seg in segs:
            r = renderer.render(seg)
            voice = r.voice if r else renderer.voice(seg.get("voice"))
            seg["voice"] = voice.id if voice else None  # the voice it is read in (the default when none is set)
            seg["speakable"] = is_speakable(clean_text(seg["text"]))
            renders.append(r)
        put("analysis/segments.jsonl", b"".join(_json_bytes(s) + b"\n" for s in segs))

        used = {s["voice"] for s in segs if s.get("voice")} | {c["voice"] for c in public["characters"] if c.get("voice")}
        voice_entries, voice_files = set(), []
        for vid in sorted(used):
            v = renderer.get(vid)
            if v is None or v.wav_path is None:
                continue
            jname, wname = f"voices/{vid}.json", f"voices/{vid}.wav"
            voice_entries |= {jname, wname}
            voice_files.append(v.wav_path.with_suffix(".json"))
            vjson = _json_bytes(v.to_json())
            if not z.same(jname, vjson) or wname not in z:
                writes[jname] = (vjson, True)
                writes[wname] = (v.wav_path.read_bytes(), False)
        deletes = {n for n in z.names() if n.startswith("voices/") and n not in voice_entries}

        index = json.loads(z.read("audio/index.json")) if "audio/index.json" in z else {}
        cache = app[CACHE_KEY]
        new_index: dict[str, dict] = {}
        clips, more, encode_error = 0, False, None
        for seg, r in zip(segs, renders):
            old = index.get(seg["id"])
            if old is not None and old.get("entry") not in z:
                old = None
            if r is None:
                if (renderer.spec is None or keep_audio) and old is not None:
                    new_index[seg["id"]] = old  # the speech model is unknown here, or the analysis isn't done
                continue
            if old is not None and old.get("key") == r.key:
                new_index[seg["id"]] = old
                continue
            if encode_error is None and cache.has(r.key):
                if clips >= max_clips:
                    more = True
                    if old is not None:
                        new_index[seg["id"]] = old
                    continue
                got = cache.peek(r.key)
                if got is not None:
                    rate, pcm = got
                    if cfg.level_speech:  # clips cached before levelling (or evening out) existed
                        pcm = normalize(pcm, rate, seg.get("delivery"))
                    try:
                        data = encode_opus(pcm, rate)
                    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                        encode_error = str(exc)
                        logger.warning("%s: speech not stored: %s", book_id, exc)
                    else:
                        name = f"audio/c{seg['chapter']}/{seg['id']}.opus"
                        writes[name] = (data, False)
                        new_index[seg["id"]] = {"entry": name, "duration": round(len(pcm) / 2 / rate, 3), "rate": rate,
                                                "voice": r.voice.id, "engine": r.engine, "key": r.key,
                                                "content": r.content}
                        clips += 1
                        continue
            if old is not None and (old.get("content") == r.content or keep_audio):
                # Made with other settings: still the right words in the right voice. Or the book is being
                # analysed again: the line will likely come back as it was.
                new_index[seg["id"]] = old
        live = {v["entry"] for v in new_index.values()}
        if keep_audio:  # clips of segments the new analysis has (for now) merged or split stay too
            new_index.update({sid: v for sid, v in index.items()
                              if sid not in new_index and v.get("entry") in z and v["entry"] not in live})
            live = {v["entry"] for v in new_index.values()}
        deletes |= {v["entry"] for v in index.values() if v.get("entry") in z and v["entry"] not in live}
        put("audio/index.json", _json_bytes(new_index))

        if writes or deletes:
            info = series_info(app, book_id)
            try:
                created = json.loads(z.read("manifest.json")).get("created") if "manifest.json" in z else None
            except (ValueError, ZipFormatError):
                created = None
            now = round(time.time())
            manifest = {
                "format": FORMAT, "format_version": FORMAT_VERSION, "id": book_id,
                "title": book.get("title", ""), "author": book.get("author", ""), "language": book.get("language", ""),
                "series": info["series"], "volume": info["volume"], "engine": renderer.engine_id,
                "segments": len(segs), "voiced": len(new_index),
                "cover": "images/cover.jpg" if cover is not None else None, "images": images,
                "created": created or now, "updated": now,
            }
            writes["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"), True)
            z.commit(writes, deletes, meta={"format": FORMAT})
        compacted = bool(z.size) and z.dead > max(COMPACT_MIN_DEAD, z.size * COMPACT_DEAD_SHARE) and z.compact()
        st.readers.pop(book_id, None)
        st.voice_files[book_id] = voice_files
        return {"written": len(writes), "deleted": len(deletes), "clips": clips, "voiced": len(new_index),
                "segments": len(segs), "more": more, "size": z.size, "dead": z.dead, "generation": z.generation,
                "compacted": compacted, **({"error": encode_error} if encode_error else {})}


def forget(app: web.Application, book_id: str, segment_ids) -> list[str]:
    """Drop these passages' speech from the book file and the audio cache, so it is made again when they are next
    played: a take the speech checks let through before they were made stricter (one of Kirito's lines stopped
    before its last sentence; '"paper"' came out "tap heaper"). Speech only in the cache so far (not yet in the
    book file) is dropped too. Returns the passages that had speech."""
    from .app import CACHE_KEY
    from .dramatize import DRAMA_STORE_KEY, ScriptStore

    st = _state(app)
    path = file_path(app, book_id)
    wanted = list(dict.fromkeys(segment_ids))
    cached = []
    store = app[DRAMA_STORE_KEY]
    book = store.book(book_id)
    if book is not None:
        renderer = Renderer(app, book_id, book)
        for seg in segments(book, ScriptStore.public(store.script(book_id))):
            if seg["id"] in wanted and (r := renderer.render(seg)) is not None and app[CACHE_KEY].has(r.key):
                app[CACHE_KEY].discard(r.key)
                cached.append(seg["id"])
    with st.lock(book_id):
        if not path.exists():
            return cached
        z = AppendZip.open(path)
        index = json.loads(z.read("audio/index.json")) if "audio/index.json" in z else {}
        gone = [sid for sid in wanted if sid in index]
        if not gone:
            return cached
        for sid in gone:
            app[CACHE_KEY].discard(index[sid]["key"])
        kept = {sid: v for sid, v in index.items() if sid not in gone}
        live = {v.get("entry") for v in kept.values()}
        deletes = {index[sid]["entry"] for sid in gone if index[sid].get("entry") in z} - live
        writes = {"audio/index.json": (_json_bytes(kept), True)}
        if "manifest.json" in z:
            manifest = json.loads(z.read("manifest.json"))
            manifest.update(voiced=len(kept), updated=round(time.time()))
            writes["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"), True)
        z.commit(writes, deletes, meta={"format": FORMAT})
        st.readers.pop(book_id, None)
    logger.info("%s: speech of %s forgotten; made again when played", book_id, ", ".join(gone))
    return list(dict.fromkeys(cached + gone))


def sync_all(app: web.Application, book_id: str) -> dict:
    """sync() until every cached clip of the book is in its file."""
    total = 0
    while True:
        r = sync(app, book_id)
        total += r.get("clips", 0)
        if not r.get("more"):
            return {**r, "clips": total}


def _inputs(app: web.Application, book_id: str) -> tuple:
    """Everything a sync depends on except the audio cache (new speech marks the book dirty instead)."""
    from .app import CONFIG_KEY, PRON_KEY
    from .dramatize import DRAMA_STORE_KEY

    d = app[DRAMA_STORE_KEY].dir(book_id)
    files = [d / n for n in ("book.json", "script.json", "meta.json", "bookmarks.json", "cover.jpg", FILE_NAME)]
    files += [app[PRON_KEY].path, *_state(app).voice_files.get(book_id, [])]
    stamps = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
    cfg = app[CONFIG_KEY]
    return stamps + (cfg.default_engine, cfg.default_voice, cfg.trim_silence,
                     json.dumps(cfg.engine_options, sort_keys=True))


async def sweep(app: web.Application) -> None:
    """Sync every book that changed since its last sync, except while it is being analysed or cast."""
    from .dramatize import DRAMA_STORE_KEY

    store = app[DRAMA_STORE_KEY]
    st = _state(app)
    for book_id in store.ids():
        if book_id not in st.dirty and st.seen.get(book_id) == _inputs(app, book_id):
            continue
        if store.script(book_id).get("status") in _WRITING:
            continue  # its labels change every few seconds: synced when it is done
        st.dirty.discard(book_id)
        try:
            r = await asyncio.to_thread(sync_all, app, book_id)
            if r.get("written") or r.get("deleted"):
                logger.info("%s: %s updated (%d new clips, %d of %d segments voiced, %.1f MB%s)", book_id, FILE_NAME,
                            r["clips"], r["voiced"], r["segments"], r["size"] / 1e6,
                            ", compacted" if r["compacted"] else "")
        except Exception:  # noqa: BLE001 - one book's trouble must not stop the others
            logger.exception("%s: updating %s failed", book_id, FILE_NAME)
        st.seen[book_id] = _inputs(app, book_id)


# ---------------------------------------------------------------- import


def _script_from_public(public: dict) -> dict:
    """The server's script record from a book file's analysis/script.json (a file without x-server/)."""
    lines, corrections = {}, {}
    tuned = {name: {qid: line[name] for qid, line in (public.get("lines") or {}).items() if line.get(name) is not None}
             for name in _TUNING_NAMES}
    for qid, line in (public.get("lines") or {}).items():
        if line.get("corrected"):
            corrections[qid] = {"speaker": line.get("speaker"), "delivery": line.get("delivery"),
                                **({"mention": True} if line.get("mention") else {})}
            if line.get("say"):  # the spoken form belongs to the analysis under the correction
                base = line.get("analysis") or {}
                lines[qid] = {"speaker": base.get("speaker"), "delivery": base.get("delivery"), "say": line["say"]}
        else:
            lines[qid] = {"speaker": line.get("speaker"), "delivery": line.get("delivery"),
                          **({"mention": True} if line.get("mention") else {}),
                          **({"say": line["say"]} if line.get("say") else {})}
    characters = {c["id"]: {k: v for k, v in c.items() if k not in ("id", "lines")}
                  for c in public.get("characters") or [] if c.get("id")}
    return {"status": public.get("status", "none"), "progress": public.get("progress", 0.0),
            "detail": public.get("detail", ""), "model": public.get("model", ""), "characters": characters,
            "lines": lines, "corrections": corrections, "chunks_done": [], "narration": public.get("narration") or [],
            **{name: v for name, v in tuned.items() if v}}


def book_id_of(path: Path) -> str:
    """The id of the book in a book file (its manifest), without adding it."""
    try:
        z = AppendZip.open(path)
        return str(json.loads(z.read("manifest.json")).get("id") or "")
    except (ZipFormatError, KeyError, ValueError, OSError) as exc:
        raise ValueError(f"Not an ElvenReader book file: {exc}") from exc


def import_file(app: web.Application, path: Path, replace: bool = False) -> str:
    """Add a book file to the library: its book, pictures, analysis, voices (those this PC doesn't have; a voice
    with the same id but other audio stays as it is here) and pronunciations; the file itself is copied in and
    keeps its speech. Raises ValueError for a file that isn't a book file, FileExistsError for a book already
    in the library (unless `replace`). Returns the book's id."""
    from .app import PRON_KEY, VOICES_KEY, pronunciations_for
    from .dramatize import DRAMA_STORE_KEY
    from .pronunciation import series_key

    store = app[DRAMA_STORE_KEY]
    try:
        z = AppendZip.open(path)
        manifest = json.loads(z.read("manifest.json"))
    except (ZipFormatError, KeyError, ValueError, OSError) as exc:
        raise ValueError(f"Not an ElvenReader book file: {exc}") from exc
    if manifest.get("format") != FORMAT:
        raise ValueError("Not an ElvenReader book file")
    if int(manifest.get("format_version") or 0) > FORMAT_VERSION:
        raise ValueError("This book file was made by a newer ElvenReader")
    book_id = str(manifest.get("id") or "")
    d = store.dir(book_id)  # ValueError for a bad id
    if store.book(book_id) is not None and not replace:
        raise FileExistsError(f"'{manifest.get('title') or book_id}' is already in the library")
    book = json.loads(z.read("text/book.json"))
    cover = z.read(manifest["cover"]) if manifest.get("cover") in z else None
    st = _state(app)
    with st.lock(book_id):
        d.mkdir(parents=True, exist_ok=True)
        store.save_book(book_id, {**book, **({"cover": base64.b64encode(cover).decode()} if cover else {})})
        if "x-server/meta.json" in z:
            kept = json.loads(z.read("x-server/meta.json"))
            meta_path = d / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({k: kept[k] for k in ("series", "title", "title_user") if k in kept})
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if "x-server/bookmarks.json" in z:  # where each reader was: each profile keeps its newer place
            from .bookmark import merge

            merge(app, book_id, json.loads(z.read("x-server/bookmarks.json")))
        for ref, name in (manifest.get("images") or {}).items():
            f = store.picture_file(book_id, ref)
            if f is not None and name in z:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(z.read(name))
        if "source/book.epub" in z:
            (d / "book.epub").write_bytes(z.read("source/book.epub"))
        if "x-server/script.json" in z:
            script = json.loads(z.read("x-server/script.json"))
        else:
            script = _script_from_public(json.loads(z.read("analysis/script.json")))
        if script.get("status") in _BUSY:
            script.update(status="none", detail="imported before its analysis had finished")
        store.save_script(book_id, script)

        voices = app[VOICES_KEY]
        for name in z.names():
            if not (name.startswith("voices/") and name.endswith(".json")):
                continue
            raw = json.loads(z.read(name))
            vid, wav = raw.get("id"), name[:-5] + ".wav"
            if not vid or wav not in z:
                continue
            have = voices.get(vid)
            if have is None:
                voices.add(raw.get("name") or vid, z.read(wav), raw.get("ref_text") or "",
                           language=raw.get("language") or "", instruct=raw.get("instruct") or "",
                           seed=raw.get("seed"), source=raw.get("source") or "upload",
                           description=raw.get("description") or "", voice_id=vid)
            elif have.audio_sha != raw.get("audio_sha"):
                logger.warning("%s: voice %s differs on this PC; this PC's is kept", book_id, vid)

        if "analysis/pronunciations.json" in z:
            from .series import book_series

            pron = app[PRON_KEY]
            series = book_series(app, book_id)
            here = {(e.key(), e.respelling, e.phonemes) for e in pronunciations_for(app, book_id)}
            for e in json.loads(z.read("analysis/pronunciations.json")):
                term = e.get("term") or ""
                key = term if e.get("case_sensitive") else term.casefold()
                if (key, e.get("respelling") or "", e.get("phonemes") or "") in here:
                    continue
                # Its series' entries stay series entries; the others (another PC's user-wide entries, or
                # site-wide ones from before users) apply to this book only.
                in_series = e.get("series") and not e.get("book") and series_key(e["series"]) == series_key(series)
                try:
                    pron.add(term, e.get("respelling") or "", phonemes=e.get("phonemes") or "",
                             case_sensitive=bool(e.get("case_sensitive")),
                             book=None if in_series else book_id, series=series if in_series else None)
                except ValueError:
                    continue

        target = d / FILE_NAME
        if Path(path).resolve() != target.resolve():
            tmp = target.with_name(target.name + ".import")
            shutil.copyfile(path, tmp)
            tmp.replace(target)
        st.readers.pop(book_id, None)
        st.seen.pop(book_id, None)
    if script.get("status") == "done":
        from .series import remember

        try:
            remember(app, book_id)
        except Exception:  # noqa: BLE001 - the book is in; the series memory can catch up later
            logger.exception("%s: could not add the imported cast to its series", book_id)
    return book_id


# ---------------------------------------------------------------- routes


async def file_status(request: web.Request) -> web.Response:
    """GET /api/books/{book}/file: the book file's size, dead bytes, generation and how much of it is voiced."""
    book_id = request.match_info["book"]
    try:
        path = file_path(request.app, book_id)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if not path.exists():
        return web.json_response({"exists": False})

    def read() -> dict:
        z = AppendZip.open(path)
        manifest = json.loads(z.read("manifest.json")) if "manifest.json" in z else {}
        return {"exists": True, "path": str(path), "size": z.size, "dead": z.dead, "generation": z.generation,
                "segments": manifest.get("segments", 0), "voiced": manifest.get("voiced", 0),
                "updated": manifest.get("updated")}

    try:
        return web.json_response(await asyncio.to_thread(read))
    except (ZipFormatError, ValueError, OSError) as exc:
        return web.json_response({"error": f"unreadable: {exc}"}, status=500)


async def file_sync(request: web.Request) -> web.Response:
    """POST /api/books/{book}/file/sync: bring the book file up to date now."""
    from .dramatize import DRAMA_STORE_KEY

    book_id = request.match_info["book"]
    try:
        if request.app[DRAMA_STORE_KEY].book(book_id) is None:
            return web.json_response({"error": "no such book"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    st = _state(request.app)
    st.dirty.discard(book_id)
    r = await asyncio.to_thread(sync_all, request.app, book_id)
    st.seen[book_id] = _inputs(request.app, book_id)
    return web.json_response(r)


async def file_forget(request: web.Request) -> web.Response:
    """POST /api/books/{book}/file/forget {"segments": [passage ids]}: make these passages' speech again when next
    played (forget()). Returns {"forgotten": [the ones that had speech]}."""
    from .dramatize import DRAMA_STORE_KEY

    book_id = request.match_info["book"]
    try:
        if request.app[DRAMA_STORE_KEY].book(book_id) is None:
            return web.json_response({"error": "no such book"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    try:
        body = await request.json()
    except ValueError:
        body = None
    ids = body.get("segments") if isinstance(body, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or not 0 < len(ids) <= 10000:
        return web.json_response({"error": "segments must be a list of passage ids"}, status=400)
    gone = await asyncio.to_thread(forget, request.app, book_id, ids)
    return web.json_response({"forgotten": gone})


def setup(app: web.Application) -> None:
    from .app import CONFIG_KEY

    _state(app)
    interval = app[CONFIG_KEY].book_file_sync_s

    async def _ctx(app: web.Application):
        task = None
        if interval > 0:
            async def _loop():
                await asyncio.sleep(min(60.0, interval))
                while True:
                    try:
                        await sweep(app)
                    except Exception:  # noqa: BLE001
                        logger.exception("book file sweep failed")
                    await asyncio.sleep(interval)

            task = asyncio.ensure_future(_loop())
        yield
        if task is not None:
            task.cancel()

    app.cleanup_ctx.append(_ctx)
    app.router.add_get("/api/books/{book}/file", file_status)
    app.router.add_post("/api/books/{book}/file/sync", file_sync)
    app.router.add_post("/api/books/{book}/file/forget", file_forget)
