"""Voice store: one reference WAV plus a JSON description per voice.

Every voice has reference audio and its transcript, so any cloning engine can
use it and the speaker stays the same from sentence to sentence.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# What every designed voice says: a voice whose clip says anything else is a real recording (uploaded or imported).
DESIGN_TEXT = (
    "The old house stood at the end of the lane, and every evening the lamps came on one by one, "
    "as if someone inside were still waiting for us to come home."
)


@dataclass
class Voice:
    id: str
    name: str
    ref_text: str = ""
    language: str = ""
    # Description of the voice ("male, elderly, low pitch, british accent").
    # Engines with voice design can use it; cloning engines ignore it.
    instruct: str = ""
    seed: int | None = None
    source: str = "upload"
    description: str = ""
    created: float = field(default_factory=time.time)
    audio_sha: str = ""
    recorded: bool = False  # set when a real recording is uploaded as the voice's reference
    # Set by the store, not persisted.
    wav_path: Path | None = field(default=None, compare=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("wav_path", None)
        return d

    @property
    def is_recording(self) -> bool:
        """A real recording, not a design: uploaded as one (`recorded`), or a clip that doesn't say DESIGN_TEXT, which
        every design says (voices from before the flag, imported ones)."""
        return self.recorded or " ".join(self.ref_text.split()) != DESIGN_TEXT

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language or None,
            "description": self.description or self.instruct or None,
            "source": self.source,
            "stable": True,
            "recorded": self.is_recording,
        }

    def details(self) -> dict:
        """Everything the voice editor shows and edits."""
        return {
            **self.public(),
            "description": self.description,
            "instruct": self.instruct,
            "ref_text": self.ref_text,
            "seed": self.seed,
            "created": self.created,
            "version": self.fingerprint,
            "recorded": self.is_recording,
        }

    def worker_spec(self) -> dict:
        """What an engine worker needs to clone this voice. The reference is a copy named by the audio's hash:
        engines cache what they derive from a reference file by its path, so an edited voice (same file name,
        new audio) must come with a new path or they keep speaking with the old voice."""
        return {
            "id": self.id,
            "name": self.name,
            "ref_audio": self.reference_path(),
            "ref_text": self.ref_text,
            "language": self.language,
            "instruct": self.instruct,
            "seed": self.seed,
        }

    def reference_path(self) -> str | None:
        if not self.wav_path:
            return None
        sha = (self.audio_sha or "")[:16]
        if not sha:
            return str(self.wav_path)
        copy = self.wav_path.parent / ".by-sha" / f"{sha}.wav"
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            tmp = copy.with_suffix(f".{os.getpid()}.tmp")
            shutil.copyfile(self.wav_path, tmp)
            os.replace(tmp, copy)
        return str(copy)

    @property
    def fingerprint(self) -> str:
        """Changes whenever anything that affects the voice's sound changes."""
        h = hashlib.sha256()
        for part in (self.id, self.audio_sha, self.ref_text, self.language, str(self.seed), self.instruct):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class VoiceStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _load(self, json_path: Path) -> Voice | None:
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
        known = set(Voice.__dataclass_fields__) - {"wav_path"}
        voice = Voice(**{k: v for k, v in raw.items() if k in known})
        wav = json_path.with_suffix(".wav")
        if not wav.exists():
            return None
        voice.wav_path = wav
        if not voice.audio_sha:
            voice.audio_sha = _sha_file(wav)
            self._save(voice)
        return voice

    def _save(self, voice: Voice) -> None:
        (self.root / f"{voice.id}.json").write_text(json.dumps(voice.to_json(), indent=2), encoding="utf-8")

    def list(self) -> list[Voice]:
        voices = [v for p in sorted(self.root.glob("*.json")) if (v := self._load(p))]
        return sorted(voices, key=lambda v: (v.source == "character", v.name.lower()))

    def get(self, voice_id: str | None) -> Voice | None:
        if not voice_id or not _ID_RE.match(voice_id):
            return None
        path = self.root / f"{voice_id}.json"
        return self._load(path) if path.exists() else None

    def default(self, preferred: str = "") -> Voice | None:
        if preferred and (v := self.get(preferred)):
            return v
        # Your own voices (imported or uploaded) before designed ones, which are mostly made for characters.
        rank = {"voicestudio": 0, "upload": 1}
        voices = sorted((v for v in self.list() if v.source != "character"),
                        key=lambda v: (rank.get(v.source, 2), v.name.lower()))
        return voices[0] if voices else None

    def add(
        self,
        name: str,
        wav: bytes | Path,
        ref_text: str,
        *,
        language: str = "",
        instruct: str = "",
        seed: int | None = None,
        source: str = "upload",
        description: str = "",
        voice_id: str | None = None,
        recorded: bool | None = None,
    ) -> Voice:
        voice_id = voice_id or secrets.token_hex(4)
        if not _ID_RE.match(voice_id):
            raise ValueError(f"invalid voice id {voice_id!r}")
        dest = self.root / f"{voice_id}.wav"
        if isinstance(wav, Path):
            shutil.copyfile(wav, dest)
        else:
            if not wav.startswith(b"RIFF"):
                raise ValueError("reference audio must be a WAV file")
            dest.write_bytes(wav)
        voice = Voice(
            id=voice_id,
            name=name.strip() or voice_id,
            ref_text=ref_text.strip(),
            language=language,
            instruct=instruct,
            seed=seed,
            source=source,
            description=description,
            audio_sha=_sha_file(dest),
            recorded=source == "upload" if recorded is None else recorded,  # a voice made from your own clip
        )
        voice.wav_path = dest
        self._save(voice)
        return voice

    # -- editing ---------------------------------------------------------------

    HISTORY_KEEP = 5

    def _history_dir(self, voice_id: str) -> Path:
        return self.root / ".history" / voice_id

    def _snapshot(self, voice: Voice) -> None:
        """Keep the current version so an edit can be undone."""
        d = self._history_dir(voice.id)
        d.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns()}"
        shutil.copyfile(voice.wav_path, d / f"{stamp}.wav")
        (d / f"{stamp}.json").write_text(json.dumps(voice.to_json(), indent=2), encoding="utf-8")
        for old in sorted(d.glob("*.json"))[:-self.HISTORY_KEEP]:
            old.unlink(missing_ok=True)
            old.with_suffix(".wav").unlink(missing_ok=True)

    def history(self, voice_id: str) -> list[dict]:
        """Earlier versions, newest first (`stamp` names one: history_wav, restore)."""
        out = []
        for p in sorted(self._history_dir(voice_id).glob("*.json"), reverse=True):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"stamp": p.stem, "saved": int(p.stem) / 1e9, "name": raw.get("name", ""),
                        "description": raw.get("description") or raw.get("instruct") or "",
                        "audio": (raw.get("audio_sha") or "")[:12]})
        return out

    def history_wav(self, voice_id: str, stamp: str) -> Path | None:
        """An earlier version's reference audio."""
        if not stamp.isdigit() or not _ID_RE.match(voice_id):
            return None
        p = self._history_dir(voice_id) / f"{stamp}.wav"
        return p if p.exists() else None

    def restore(self, voice_id: str, stamp: str) -> Voice | None:
        """Bring back an earlier version; the current one is kept as a version too (so this can be undone)."""
        voice, wav = self.get(voice_id), self.history_wav(voice_id, stamp)
        if voice is None or wav is None:
            return None
        raw = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        audio = wav.read_bytes()  # read first: keeping the current version may prune the oldest ones
        self._snapshot(voice)
        known = set(Voice.__dataclass_fields__) - {"wav_path"}
        old = Voice(**{k: val for k, val in raw.items() if k in known})
        old.id = voice_id
        voice.wav_path.write_bytes(audio)
        old.wav_path = voice.wav_path
        old.audio_sha = _sha_file(voice.wav_path)
        self._save(old)
        return old

    def update(self, voice_id: str, **changes) -> Voice | None:
        voice = self.get(voice_id)
        if voice is None:
            return None
        new = Voice(**voice.to_json())
        for key in ("name", "ref_text", "language", "instruct", "description"):
            if key in changes and isinstance(changes[key], str):
                setattr(new, key, changes[key].strip())
        if "seed" in changes:
            seed = changes["seed"]
            if seed in (None, ""):
                new.seed = None
            elif isinstance(seed, (int, float, str)) and str(seed).strip().lstrip("-").isdigit():
                new.seed = int(seed)
            else:
                raise ValueError("seed must be a whole number or empty")
        if not new.name:
            new.name = voice.name
        if new.to_json() == voice.to_json():
            return voice
        self._snapshot(voice)
        new.wav_path = voice.wav_path
        self._save(new)
        return new

    def replace_audio(self, voice_id: str, wav: bytes, ref_text: str, recorded: bool = False, **meta) -> Voice | None:
        """New reference audio (and its transcript); `meta` may set instruct, description, seed, language.
        `recorded`: a real recording you uploaded (a design's audio isn't)."""
        voice = self.get(voice_id)
        if voice is None:
            return None
        if not wav.startswith(b"RIFF"):
            raise ValueError("reference audio must be a WAV file")
        self._snapshot(voice)
        voice.wav_path.write_bytes(wav)
        voice.audio_sha = _sha_file(voice.wav_path)
        voice.ref_text = ref_text.strip()
        voice.recorded = recorded
        for key in ("instruct", "description", "language"):
            if isinstance(meta.get(key), str):
                setattr(voice, key, meta[key].strip())
        if "seed" in meta:
            voice.seed = meta["seed"]
        self._save(voice)
        return voice

    def duplicate(self, voice_id: str, name: str = "") -> Voice | None:
        voice = self.get(voice_id)
        if voice is None:
            return None
        return self.add(
            name or f"{voice.name} (copy)", voice.wav_path, voice.ref_text, language=voice.language,
            instruct=voice.instruct, seed=voice.seed, source=voice.source, description=voice.description,
            recorded=voice.recorded,
        )

    def delete(self, voice_id: str) -> bool:
        voice = self.get(voice_id)
        if voice is None:
            return False
        for suffix in (".json", ".wav"):
            (self.root / f"{voice_id}{suffix}").unlink(missing_ok=True)
        shutil.rmtree(self._history_dir(voice_id), ignore_errors=True)
        return True

    def import_omnivoice(self, db_path: Path, voices_dir: Path) -> list[Voice]:
        """Copy VoiceStudio/OmniVoice profiles that have reference audio.

        The database is opened read-only; nothing in VoiceStudio is modified.
        Profiles already imported (same id) are skipped.
        """
        con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        imported: list[Voice] = []
        try:
            rows = con.execute("SELECT * FROM voice_profiles").fetchall()
        finally:
            con.close()
        for row in rows:
            r = dict(row)
            ref = r.get("ref_audio_path") or (r.get("is_locked") and r.get("locked_audio_path")) or ""
            if not ref or not r.get("ref_text"):
                continue
            wav = Path(ref) if Path(ref).is_absolute() else Path(voices_dir) / ref
            if not wav.exists() or self.get(r["id"]):
                continue
            imported.append(
                self.add(
                    r.get("name") or r["id"],
                    wav,
                    r.get("ref_text") or "",
                    language=r.get("language") or "",
                    instruct=r.get("instruct") or "",
                    seed=r.get("seed"),
                    source="voicestudio",
                    description=r.get("description") or "",
                    voice_id=r["id"],
                )
            )
        return imported
