"""Engine catalogue: which models the server can run and how.

Each engine runs in its own worker process, usually from its own virtual
environment (``<models_dir>/venvs/<id>``) because the models pin conflicting
versions of torch/transformers. Only one engine is loaded on the GPU at a time.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
