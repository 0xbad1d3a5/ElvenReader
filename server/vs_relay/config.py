"""Relay configuration: a small JSON file plus environment overrides."""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PORT = 3950
DEFAULT_VOICESTUDIO_URL = "http://127.0.0.1:3900"
# Unambiguous characters so the token is easy to type on a phone if needed.
_TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
_TOKEN_LENGTH = 16


def generate_token() -> str:
    return "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(_TOKEN_LENGTH))


def default_data_dir() -> Path:
    env = os.environ.get("VS_RELAY_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


@dataclass
class RelayConfig:
    token: str
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    voicestudio_url: str = DEFAULT_VOICESTUDIO_URL
    # Voice profile used when the phone has not picked one. Empty = first saved
    # profile with reference audio, so the speaker never changes between sentences.
    default_voice: str = ""
    # Waiting for the first audio covers a cold model load in VoiceStudio.
    first_audio_timeout_s: float = 90.0
    # Gap allowed between audio chunks (one sentence of synthesis).
    chunk_timeout_s: float = 60.0
    connect_timeout_s: float = 5.0

    def to_json(self) -> dict:
        return asdict(self)


def load_config(data_dir: Path | None = None) -> RelayConfig:
    """Load ``config.json`` from the data dir, creating it (with a fresh token) if missing."""
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "config.json"
    raw: dict = {}
    if path.exists():
        # utf-8-sig: tolerate the BOM that Notepad / Windows PowerShell add when editing by hand.
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not raw.get("token"):
        raw["token"] = generate_token()
    known = RelayConfig.__dataclass_fields__.keys()
    cfg = RelayConfig(**{k: v for k, v in raw.items() if k in known})
    path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")

    # Environment overrides are applied after saving so they never get persisted.
    if os.environ.get("VS_RELAY_TOKEN"):
        cfg.token = os.environ["VS_RELAY_TOKEN"]
    if os.environ.get("VS_RELAY_HOST"):
        cfg.host = os.environ["VS_RELAY_HOST"]
    if os.environ.get("VS_RELAY_PORT"):
        cfg.port = int(os.environ["VS_RELAY_PORT"])
    if os.environ.get("VOICESTUDIO_URL"):
        cfg.voicestudio_url = os.environ["VOICESTUDIO_URL"]
    cfg.voicestudio_url = cfg.voicestudio_url.rstrip("/")
    return cfg


def rotate_token(data_dir: Path | None = None) -> RelayConfig:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    raw["token"] = generate_token()
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return load_config(data_dir)
