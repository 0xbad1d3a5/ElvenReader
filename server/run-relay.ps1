# Starts the VoiceStudio TTS relay, creating its virtual environment on first run.
# Extra arguments are passed through, e.g.:  .\run-relay.ps1 --port 3960
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    & $python -m pip install --quiet --disable-pip-version-check -e .
}

& $python -m vs_relay @args
