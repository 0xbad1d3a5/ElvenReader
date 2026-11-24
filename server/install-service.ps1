# Runs the ElvenReader server in the background whenever you are logged on to Windows: a scheduled task, "ElvenReader
# server", for your account (no administrator rights needed). It starts at logon, runs hidden, never times out and
# is started again if it stops (run-service.ps1).
#   .\install-service.ps1             install (or update) and start it now
#   .\install-service.ps1 -Uninstall  stop and remove it
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$name = "ElvenReader server"

if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
}
# Ending the task ends its PowerShell; the server it started is stopped here too.
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*-m elvenreader_server*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
if ($Uninstall) {
    Write-Host "Removed '$name'."
    return
}

$script = Join-Path $PSScriptRoot "run-service.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $PSScriptRoot `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$user = "$env:USERDOMAIN\$env:USERNAME"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description "ElvenReader: the PC server for the library (http://127.0.0.1:3950/books)." | Out-Null
Start-ScheduledTask -TaskName $name
Write-Host "Installed '$name': it starts when you log on and is running now. Log: $PSScriptRoot\data\service.log"
