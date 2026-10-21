#Requires -RunAsAdministrator
# Allows phones on your private network (and Tailscale) to reach the relay.
# Run once from an elevated PowerShell:  .\allow-firewall.ps1
param([int]$Port = 3950)

$name = "VoiceStudio TTS relay (TCP $Port)"
Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port `
    -Action Allow -Profile Domain, Private | Out-Null
Write-Host "Firewall rule '$name' added (Domain + Private networks)."

$public = Get-NetConnectionProfile | Where-Object NetworkCategory -eq "Public"
if ($public) {
    Write-Warning ("These networks are marked Public, so the rule does not apply to them: " +
        ($public.Name -join ", ") + ". Mark your home network Private in Windows Settings > Network.")
}
