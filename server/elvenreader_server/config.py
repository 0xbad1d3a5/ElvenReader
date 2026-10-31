"""Server configuration: a small JSON file plus environment overrides."""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PORT = 3950
DEFAULT_ENGINE = "omnivoice"
# Unambiguous characters so the token is easy to type on another device if needed.
_TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
_TOKEN_LENGTH = 16


def generate_token() -> str:
    return "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(_TOKEN_LENGTH))


def _env(name: str) -> str | None:
    """ELVENREADER_<name>, or the ANDROIDREADER_<name> of the project's old name."""
    return os.environ.get(f"ELVENREADER_{name}") or os.environ.get(f"ANDROIDREADER_{name}")


def default_data_dir() -> Path:
    env = _env("DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


@dataclass
class ServerConfig:
    token: str
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    # Where engine virtual environments (venvs/<engine>) and model weights
    # (hf/, the Hugging Face cache) live. Kept out of git; several GB per engine.
    models_dir: str = ""
    # Engine used when a request does not name one.
    default_engine: str = DEFAULT_ENGINE
    # Voice used when a request has not picked one. Empty = first voice in the store.
    default_voice: str = ""
    # Extra URLs to put in pairing codes, e.g. this PC's address on your own VPN
    # ("http://10.8.0.2:3950"). They get links and QR codes on the server's page too.
    extra_urls: list = field(default_factory=list)
    # Load the default engine at start-up so the first sentence is not slow.
    preload: bool = True
    # Cut the models' start/end padding: readers wait for each sentence to
    # finish before requesting the next, so padding lengthens every gap.
    trim_silence: bool = True
    # Silence added after an utterance ending a sentence (a third of it after a
    # clause). A request can override it.
    sentence_pause_ms: int = 300
    # On-disk cache of synthesized sentences.
    cache_max_mb: int = 4096
    # Loading a quality model can take a few minutes the first time.
    load_timeout_s: float = 900.0
    # One sentence of synthesis; quality models can be slower than real time.
    synth_timeout_s: float = 180.0
    # Per-engine overrides, e.g. {"omnivoice": {"num_step": 32}}.
    engine_options: dict = field(default_factory=dict)
    # Local OpenAI-compatible LLM used to dramatize books. With llm_model set (a
    # GGUF file name under <models_dir>/llm), the server starts llama.cpp itself.
    llm_url: str = "http://127.0.0.1:3951/v1"
    llm_model: str = "Qwen3.5-35B-A3B-Q4_K_M.gguf"
    llm_context: int = 12288
    llm_threads: int = 12
    # Where the LLM runs: "adaptive" (default) follows what the site is used for (activity.py): while
    # someone reads or listens, analysis runs next to the speech model (experts of llm_cpu_moe_shared layers
    # in RAM); while someone designs voices and design_pause_s after, it waits; when nobody has used the site
    # for idle_after_s, it gets the whole GPU (experts of llm_cpu_moe layers in RAM) and new books' voices
    # are created. Fixed modes: "shared" (always next to the speech model), "auto" (the whole GPU after two
    # minutes without speech requests, otherwise CPU), "gpu" (always the whole GPU) or "cpu".
    llm_mode: str = "adaptive"
    idle_after_s: float = 300.0
    design_pause_s: float = 600.0
    llm_cpu_moe: int = 20
    llm_cpu_moe_shared: int = 32
    # Pass 1 of the analysis: a reasoning model reads the whole book and builds a character
    # dossier (names, apparent age and form, speech style...), which pass 2 (llm_model) uses
    # to attribute lines. Empty (default) = skip pass 1: on our gold excerpts it lowered
    # attribution accuracy (see README), so it is opt-in, e.g. "gpt-oss-20b-MXFP4.gguf".
    llm_dossier_model: str = ""
    llm_dossier_context: int = 24576
    # Words per pass-1 passage, and how hard the reasoning model thinks ("low", "medium", "high";
    # gpt-oss honours it, Qwen3 thinking models always think fully).
    llm_dossier_words: int = 3500
    llm_dossier_effort: str = "medium"
    # Let the pass-2 model reason before labelling (slower; only useful with a reasoning llm_model).
    llm_label_thinking: bool = False
    # Context for each labelling call: words of the story just before the passage, and how many
    # earlier attributions are shown. 600 / 12 (default) scored about 5 points higher than the
    # old 0 / 6 on our hardest gold excerpts for ~7% more time (README); 0 / 6 = the old fast pass.
    llm_label_context_words: int = 600
    llm_label_recent_lines: int = 12
    # Words of the story just after the passage (speech tags and replies there show who said its last lines).
    llm_label_after_words: int = 300
    # Let each labelling call revise up to this many earlier lines of the chapter when the new passage
    # shows they were wrong (a name revealed later...); 0 = off. _unnamed: only lines given to
    # unnamed speakers ("a prisoner"). Reader corrections are never revised.
    llm_label_revise_window: int = 0
    llm_label_revise_unnamed: bool = False
    # Let the labelling pass note what each character looks and sounds like (the profile used to design
    # voices) as it reads; off = the profile comes only from the roster step at the end.
    llm_label_profiles: bool = True
    # Cap on reasoning tokens per request (llama.cpp --reasoning-budget); -1 = unlimited.
    llm_reasoning_budget: int = -1
    # Engine used to create character voices from a description. When it isn't the speech model it runs
    # as a voice designer next to it, loaded while voices are designed and unloaded after designer_idle_s.
    voice_design_engine: str = "qwen3tts"
    designer_idle_s: float = 180.0
    # Casting: main characters and narrators always get a designed voice of their own, supporting
    # characters (most lines first) up to this many, extras share pooled voices.
    cast_supporting_voices: int = 12
    # Whisper fills in the transcript of new reference audio: the engine venv that has
    # openai-whisper installed, and the model file under <models_dir>.
    whisper_venv: str = "indextts"
    whisper_model: str = "whisper/large-v3-turbo.pt"
    # Version of the settings below the defaults written to config.json (for one-time migrations).
    settings_v: int = 3

    data_dir: str = field(default="", repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("data_dir", None)
        return d

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir) if self.data_dir else default_data_dir()

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir) if self.models_dir else self.data_path / "models"


def load_config(data_dir: Path | None = None) -> ServerConfig:
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
    if raw and int(raw.get("settings_v") or 0) < 1:
        # v1: the LLM shares the GPU instead of unloading the speech model ("auto" was the old default).
        if raw.get("llm_mode", "auto") == "auto":
            raw["llm_mode"] = "shared"
        raw["settings_v"] = 1
    if raw and int(raw.get("settings_v") or 0) < 2:
        # v2: the LLM follows what the site is used for ("shared" was the v1 default).
        if raw.get("llm_mode", "shared") == "shared":
            raw["llm_mode"] = "adaptive"
        raw["settings_v"] = 2
    if raw and int(raw.get("settings_v") or 0) < 3:
        # v3: OmniVoice no longer designs voices (it only takes fixed tags, not descriptions).
        if raw.get("voice_design_engine") == "omnivoice":
            raw["voice_design_engine"] = "qwen3tts"
        raw["settings_v"] = 3
    known = set(ServerConfig.__dataclass_fields__) - {"data_dir"}
    cfg = ServerConfig(**{k: v for k, v in raw.items() if k in known})
    path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")
    cfg.data_dir = str(data_dir)

    # Environment overrides are applied after saving so they never get persisted.
    if _env("TOKEN"):
        cfg.token = _env("TOKEN")
    if _env("HOST"):
        cfg.host = _env("HOST")
    if _env("PORT"):
        cfg.port = int(_env("PORT"))
    if _env("MODELS"):
        cfg.models_dir = _env("MODELS")
    return cfg


def save_settings(cfg: ServerConfig, **changes) -> None:
    """Change settings in the running config and in config.json (keeping anything else in the file)."""
    for k, v in changes.items():
        setattr(cfg, k, v)
    path = cfg.data_path / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else cfg.to_json()
    raw.update(changes)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def rotate_token(data_dir: Path | None = None) -> ServerConfig:
    data_dir = data_dir or default_data_dir()
    path = data_dir / "config.json"
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    raw["token"] = generate_token()
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return load_config(data_dir)
