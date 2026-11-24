# Runs the server for the "ElvenReader server" scheduled task (install-service.ps1), hidden, and starts it again
# if it ever stops. Output goes to data\service.log (the server's own log is data\server.log).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$log = Join-Path $PSScriptRoot "data\service.log"
if (-not (Test-Path $python)) {
    & "$PSScriptRoot\run-server.ps1" --help | Out-Null  # creates the virtual environment
}
while ($true) {
    Add-Content $log "$(Get-Date -Format s) starting the server"
    # Through cmd: PowerShell 5.1 turns a native program's stderr into errors when it is redirected.
    cmd /c "`"$python`" -m elvenreader_server >> `"$log`" 2>&1"
    Add-Content $log "$(Get-Date -Format s) the server stopped (exit code $LASTEXITCODE); starting it again in 10 s"
    Start-Sleep 10
}
