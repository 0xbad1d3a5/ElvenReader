"""Engine catalogue: which models the server can run and how.

Each engine runs in its own worker process, usually from its own virtual
environment (``<models_dir>/venvs/<id>``) because the models pin conflicting
versions of torch/transformers. Only one engine is loaded on the GPU at a time.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .voxcpm_styles import INLINE, RENDER_VERSION


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    module: str  # worker-side module exposing ``Engine``
    description: str = ""
    licence: str = ""
    repos: tuple[str, ...] = ()  # Hugging Face repos to download
    vram_gb: float = 0.0
    # How a per-sentence delivery ("shouting, urgent") is applied:
    # "tags" (inline text tags), "instruct" (instruction field), "emotion"
    # (emotion text/vector) or "" (ignored).
    delivery: str = ""
    phonemes: str = ""  # "cmu_brackets" / "cmu_annotated": accepts CMU phonemes inline
    voice_design: bool = False  # can create a voice from a text description
    languages: str = "en"
    defaults: dict = field(default_factory=dict)
    # Longer text is synthesized piece by piece (text.split_for_synthesis); each piece starts its delivery afresh,
    # so the tone can change at the seams. 0 = text.MAX_CHUNK_CHARS.
    max_chunk_chars: int = 0
    # The adapter reads brackets as asides (VoxCPM2 takes them as style instructions); lines with brackets made
    # before that are made again.
    brackets_as_asides: bool = False
    # Per-line settings the engine takes (app.TUNING: "guidance", "temperature"): the line dialog shows a slider
    # for each, the engine's own value being the default.
    tuning: tuple = ()
    # Text ending on a comma or semicolon is given a colon (a period for a quotation): text.close_clause.
    close_clauses: bool = False
    # Stutters written as consonants and a hyphen ("R-right") are spelled as they sound ("Ruh-right"):
    # text.spell_stutters.
    spell_stutters: bool = False
    # The adapter maps approved delivery tags (delivery.py) to its own words; bumped when that mapping changes,
    # which gives lines with a delivery a new cache key (lines without one keep theirs).
    styles_version: int = 0
    # VRAM a worker still holds with its model parked in system RAM (BaseEngine.park: its CUDA context and CUDA
    # graphs); the LLM leaves this much room. VoxCPM2: 1.75 GB measured (scripts/voxcpm_park_probe.py).
    parked_vram_gb: float = 0.0
    # Texts the adapter now reads differently from before (name -> regex): lines matching one get a new cache key,
    # so clips made the old way are made again; other lines keep theirs.
    reread: dict = field(default_factory=dict)
    # Sound marks in the text ("{laugh|Ahaha}", sounds.py): kind -> the engine's own tag for it; other kinds (and
    # every kind, for engines without any) are read as written.
    sound_tags: dict = field(default_factory=dict)
    # Run with the server's own interpreter (tests) instead of a venv.
    in_process_python: bool = False

    def python(self, models_dir: Path) -> Path:
        if self.in_process_python:
            return Path(sys.executable)
        return Path(models_dir) / "venvs" / self.id / "Scripts" / "python.exe"

    def installed(self, models_dir: Path) -> bool:
        return self.python(models_dir).exists()

    def public(self) -> dict:
        d = asdict(self)
        d.pop("module", None)
        d.pop("in_process_python", None)
        d["repos"] = list(self.repos)
        return d


ENGINES: dict[str, EngineSpec] = {
    spec.id: spec
    for spec in [
        EngineSpec(
            id="omnivoice",
            label="OmniVoice",
            module="elvenreader_server.engines.omnivoice_engine",
            description="k2-fsa diffusion-LM TTS, 600+ languages; the model VoiceStudio uses, "
            "so imported voices sound the same. 32 decoding steps (quality preset).",
            licence="Apache-2.0",
            repos=("k2-fsa/OmniVoice",),
            vram_gb=4.0,
            delivery="tags",
            phonemes="cmu_brackets",
            # It can design from fixed tags only ("male, low pitch"); voices are designed with Qwen3-TTS.
            voice_design=False,
            languages="600+",
            defaults={"num_step": 32, "guidance_scale": 2.0},
        ),
        EngineSpec(
            id="fish_s2",
            label="Fish Audio S2 Pro",
            module="elvenreader_server.engines.fish_s2_engine",
            description="4.4B dual-AR model; best measured clarity and expressiveness. Free-form "
            "inline delivery tags such as [shouting] or [whispers]. 44.1 kHz.",
            licence="Fish Audio Research License (non-commercial)",
            repos=("fishaudio/s2-pro",),
            vram_gb=12.0,
            delivery="tags",
            languages="80+",
            # torch.compile (via triton-windows) roughly doubles token throughput.
            defaults={"compile": True, "max_seq_len": 4096},
        ),
        EngineSpec(
            id="indextts",
            label="IndexTTS 2.5",
            module="elvenreader_server.engines.indextts_engine",
            description="bilibili's expressive cloning model; emotion from a delivery text, "
            "CMU phoneme overrides. en/zh/ja/es/ar.",
            licence="bilibili IndexTTS licence (non-commercial)",
            repos=("IndexTeam/IndexTTS-2.5",),
            vram_gb=7.0,
            delivery="emotion",
            phonemes="cmu_annotated",
            languages="en, zh, ja, es, ar",
            defaults={"compile": True},
        ),
        EngineSpec(
            id="qwen3tts",
            label="Qwen3-TTS 1.7B",
            module="elvenreader_server.engines.qwen3tts_engine",
            description="Alibaba Qwen3-TTS Base for cloning, plus the VoiceDesign model to create "
            "character voices from a description. 10 languages.",
            licence="Apache-2.0",
            repos=(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            ),
            vram_gb=6.0,
            voice_design=True,
            languages="10",
        ),
        EngineSpec(
            id="voxcpm",
            label="VoxCPM2",
            module="elvenreader_server.engines.voxcpm_engine",
            description="OpenBMB's 2B tokenizer-free diffusion-AR model: cloning that follows a free-text "
            "delivery (\"(shouting, urgent)\"), 30 languages, 48 kHz.",
            licence="Apache-2.0",
            repos=("openbmb/VoxCPM2",),
            vram_gb=7.0,  # 5.9 GB measured, and SenseVoice (hears short lines back) ~1 GB
            delivery="instruct",
            voice_design=True,
            languages="30",
            # cfg_value: guidance. normalize: VoxCPM2's own text normalisation ("70%" -> "seventy percent", "Vol." ->
            # "Volume"; see the adapter's _for_normalizer). temperature: the noise each patch of speech starts
            # from (hidden in its diffusion decoder; lower is steadier).
            defaults={"mode": "auto", "cfg_value": 2.0, "inference_timesteps": 10, "optimize": True,
                      "denoise_refs": True, "normalize": True, "temperature": 1.0},
            max_chunk_chars=2000,  # 1,342 characters in one go: steady pitch, every word said (no seams in tone)
            brackets_as_asides=True,
            tuning=("guidance", "temperature"),
            close_clauses=True,
            spell_stutters=True,
            styles_version=RENDER_VERSION,
            parked_vram_gb=2.0,
            # "No." the word was read "number" by VoxCPM2's normaliser (voxcpm_engine._for_normalizer).
            reread={"no-word": r"\bN[Oo]\.(?!\s*\d)"},
            sound_tags=INLINE,
        ),
        EngineSpec(
            id="breeze",
            label="Breeze TTS 2",
            module="elvenreader_server.engines.breeze_engine",
            description="BreezeBlue's model, #1 open-weight on the Artificial Analysis TTS leaderboard: cloning "
            "steered by a free-text instruction (the line's delivery), and voice design. English and Chinese, 24 kHz.",
            licence="BreezeBlue Research and Non-Commercial License",
            repos=("BreezeBlue/Breeze-TTS-2",),
            vram_gb=9.0,  # 8.7 GB measured (nvidia-smi) with the per-token stages compiled, the mode it runs in
            delivery="instruct",
            voice_design=True,
            languages="en, zh",
            defaults={"fast": "decode"},
            max_chunk_chars=700,  # its prompt (reference + text) has to fit 2,048 tokens
            styles_version=RENDER_VERSION,
        ),
    ]
}

FAKE_ENGINE = EngineSpec(
    id="fake",
    label="Test tone",
    module="elvenreader_server.engines.fake_engine",
    description="Generates a tone per character; used by the tests.",
    delivery="instruct",
    phonemes="cmu_brackets",
    voice_design=True,
    in_process_python=True,
)
