# Shared helpers for the server's PowerShell scripts.

function Get-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) { return $found.FullName }
    Write-Host "Installing uv (Python package manager) with winget..."
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements --disable-interactivity | Out-Null
    $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $found) { throw "uv is not installed. Install it from https://docs.astral.sh/uv/ and retry." }
    return $found.FullName
}

function Get-ModelsDir {
    # Same rule as the server: ELVENREADER_MODELS, else "models_dir" in data\config.json, else <repo>\models.
    if ($env:ELVENREADER_MODELS) { return $env:ELVENREADER_MODELS }
    $data = if ($env:ELVENREADER_DATA) { $env:ELVENREADER_DATA } else { Join-Path (Split-Path $PSScriptRoot) "data" }
    $config = Join-Path $data "config.json"
    if (Test-Path $config) {
        $cfg = Get-Content $config -Raw | ConvertFrom-Json
        if ($cfg.models_dir) { return $cfg.models_dir }
    }
    return Join-Path (Split-Path (Split-Path $PSScriptRoot)) "models"
}

# Engines, their Python version and the upstream source they need (pinned commits).
$Engines = [ordered]@{
    omnivoice = @{ Python = "3.12"; Repos = @("k2-fsa/OmniVoice") }
    fish_s2   = @{ Python = "3.12"; Repos = @("fishaudio/s2-pro");
                   Git = "https://github.com/fishaudio/fish-speech.git"; Commit = "541566932abbaebcb0a40e562a1f846738383f38"; Dir = "fish-speech" }
    indextts  = @{ Python = "3.11"; Repos = @();  # downloaded into checkpoints\IndexTTS-2.5 (see download-models.ps1)
                   Git = "https://github.com/index-tts/index-tts.git"; Commit = "d9e41aac89fd00b3d71497fddb287b7f24613712"; Dir = "index-tts" }
    qwen3tts  = @{ Python = "3.12"; Repos = @("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen/Qwen3-TTS-Tokenizer-12Hz") }
    voxcpm    = @{ Python = "3.12"; Repos = @("openbmb/VoxCPM2") }
    breeze    = @{ Python = "3.12"; Repos = @("BreezeBlue/Breeze-TTS-2");
                   Git = "https://github.com/breezeblue-ai/breeze-tts.git"; Commit = "daa6cb2efe5471f09369229f322d20bb956cb3c3"; Dir = "breeze-tts" }
}

$TorchIndex = "https://download.pytorch.org/whl/cu128"  # CUDA 12.8 builds: RTX 40-series (sm_89) and 50-series (sm_120)

# What setup-engines.ps1 and download-models.ps1 install when no engine is named: VoxCPM2 reads (and transcribes
# new reference clips with Whisper, in the same environment); Qwen3-TTS designs voices from descriptions. The
# others are optional: name them, or pass -All.
$DefaultEngines = @("voxcpm", "qwen3tts")
$WhisperEngine = "voxcpm"
