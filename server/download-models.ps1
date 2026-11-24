# Downloads model weights into <models_dir>\hf (the Hugging Face cache used by the engines).
# Usage:  .\download-models.ps1                 # VoxCPM2 and Qwen3-TTS (and Whisper, for the voice editor)
#         .\download-models.ps1 -All            # every engine
#         .\download-models.ps1 fish_s2 indextts
# Run setup-engines.ps1 first: each engine's environment provides huggingface_hub.
param([switch]$All, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Names)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\scripts\common.ps1"

$models = Get-ModelsDir
if (-not $Names) { $Names = if ($All) { @($Engines.Keys) } else { $DefaultEngines } }
$env:HF_HOME = Join-Path $models "hf"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

foreach ($name in $Names) {
    $e = $Engines[$name]
    if (-not $e) { throw "Unknown engine '$name'. Known: $($Engines.Keys -join ', ')" }
    $py = Join-Path $models "venvs\$name\Scripts\python.exe"
    if (-not (Test-Path $py)) { throw "Engine '$name' is not set up. Run .\setup-engines.ps1 $name first." }
    Write-Host "`n=== $name" -ForegroundColor Cyan
    foreach ($repo in $e.Repos) {
        & $py -c "from huggingface_hub import snapshot_download as d; print(d('$repo', max_workers=8))"
    }
    if ($name -eq "indextts") {
        # IndexTTS loads from a plain folder and adds its helper models next to it.
        $dir = Join-Path $models "checkpoints\IndexTTS-2.5"
        & $py -c "from huggingface_hub import snapshot_download as d; print(d('IndexTeam/IndexTTS-2.5', local_dir=r'$dir', max_workers=8))"
    }
    if ($name -eq $WhisperEngine) {
        # Whisper large-v3-turbo transcribes new reference clips in the voice editor (whisper_model in config.json).
        $root = Join-Path $models "whisper"
        & $py -c "import whisper; print(whisper._download(whisper._MODELS['large-v3-turbo'], r'$root', False))"
    }
}
Write-Host "`nDone. Start the server with .\run-server.ps1" -ForegroundColor Green
