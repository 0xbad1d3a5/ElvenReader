# Starts the ElvenReader server, creating its virtual environment on first run.
# Extra arguments are passed through, e.g.:  .\run-server.ps1 --port 3960 --engine fish_s2
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. "$PSScriptRoot\scripts\common.ps1"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Creating the server's virtual environment..."
    $uv = Get-Uv
    & $uv venv --python 3.12 .venv
    & $uv pip install --python $python -e ".[test]"
}

& $python -m elvenreader_server @args
