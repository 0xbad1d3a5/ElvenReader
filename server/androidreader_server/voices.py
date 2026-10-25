"""Voice store: one reference WAV plus a JSON description per voice.

Every voice has reference audio and its transcript, so any cloning engine can
use it and the speaker stays the same from sentence to sentence.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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
    # Set by the store, not persisted.
    wav_path: Path | None = field(default=None, compare=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("wav_path", None)
        return d

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language or None,
            "description": self.description or self.instruct or None,
            "source": self.source,
            "stable": True,
        }

    def worker_spec(self) -> dict:
        """What an engine worker needs to clone this voice."""
        return {
            "id": self.id,
            "name": self.name,
            "ref_audio": str(self.wav_path) if self.wav_path else None,
            "ref_text": self.ref_text,
            "language": self.language,
            "instruct": self.instruct,
            "seed": self.seed,
        }

    @property
    def fingerprint(self) -> str:
        """Changes whenever anything that affects the voice's sound changes."""
        h = hashlib.sha256()
        for part in (self.id, self.audio_sha, self.ref_text, self.language, str(self.seed)):
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
        voices = [v for v in self.list() if v.source != "character"]
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
        )
        voice.wav_path = dest
        self._save(voice)
        return voice

    def update(self, voice_id: str, **changes) -> Voice | None:
        voice = self.get(voice_id)
        if voice is None:
            return None
        for key in ("name", "ref_text", "language", "instruct", "description"):
            if key in changes and isinstance(changes[key], str):
                setattr(voice, key, changes[key].strip())
        self._save(voice)
        return voice

    def delete(self, voice_id: str) -> bool:
        voice = self.get(voice_id)
        if voice is None:
            return False
        for suffix in (".json", ".wav"):
            (self.root / f"{voice_id}{suffix}").unlink(missing_ok=True)
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
