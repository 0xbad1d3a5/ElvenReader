"""Network helpers: which addresses can the phone reach this PC on?"""
from __future__ import annotations

import ipaddress
import socket


def _primary_ipv4() -> str | None:
    # No packets are sent: connecting a UDP socket just selects the outbound interface.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def lan_addresses() -> list[dict]:
    """IPv4 addresses of this machine, primary LAN address first.

    Each entry is ``{"ip": ..., "kind": "lan" | "tailscale" | "other"}``.
    """
    found: list[str] = []
    primary = _primary_ipv4()
    if primary:
        found.append(primary)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in found:
                found.append(ip)
    except OSError:
        pass

    out = []
    for ip in found:
        addr = ipaddress.ip_address(ip)
        if addr.is_loopback or addr.is_link_local:
            continue
        if addr in ipaddress.ip_network("100.64.0.0/10"):
            kind = "tailscale"
        elif addr.is_private:
            kind = "lan"
        else:
            kind = "other"
        out.append({"ip": ip, "kind": kind})
    return out
