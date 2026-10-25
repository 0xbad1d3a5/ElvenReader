#Requires -RunAsAdministrator
# Lets phones on your private network and on your Tailscale tailnet reach the server.
# Run once from an elevated PowerShell:  .\allow-firewall.ps1
param([int]$Port = 3950)

$name = "AndroidReader server (TCP $Port)"
Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port `
    -Action Allow -Profile Domain, Private | Out-Null
Write-Host "Firewall rule '$name' added (Domain + Private networks)."

# Windows often files the Tailscale adapter under the Public profile. Allow the port on that
# adapter only, so the server is reachable from your tailnet but not from public Wi-Fi.
$ts = Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -like "*Tailscale*" -or $_.Name -like "Tailscale*" }
if ($ts) {
    $tsName = "AndroidReader server via Tailscale (TCP $Port)"
    Get-NetFirewallRule -DisplayName $tsName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    New-NetFirewallRule -DisplayName $tsName -Direction Inbound -Protocol TCP -LocalPort $Port `
        -Action Allow -Profile Any -InterfaceAlias $ts.Name | Out-Null
    Write-Host "Firewall rule '$tsName' added for adapter '$($ts.Name)'."
} else {
    Write-Host "Tailscale adapter not found; run this script again after installing Tailscale."
}

$public = Get-NetConnectionProfile | Where-Object { $_.NetworkCategory -eq "Public" -and $_.InterfaceAlias -notlike "Tailscale*" }
if ($public) {
    Write-Warning ("These networks are marked Public, so the LAN rule does not apply to them: " +
        ($public.Name -join ", ") + ". Mark your home network Private in Windows Settings > Network.")
}
