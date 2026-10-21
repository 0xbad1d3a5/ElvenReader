package com.androidvoicestudio.tts

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import java.net.InetAddress
import java.net.URI

/**
 * Picks the network that can actually reach the relay.
 *
 * Phones often keep Wi-Fi connected while routing traffic over mobile data
 * (Samsung "Switch to mobile data", Android's own fallback when Wi-Fi looks
 * weak). A LAN address such as 192.168.1.20 is unreachable over mobile data, so
 * when the relay host is on a local subnet we bind our sockets to the network
 * that owns that subnet instead of trusting the system default.
 */
object LocalNetworks {
    fun networkFor(context: Context, serverUrl: String): Network? {
        val target = hostOf(serverUrl)?.let(::parseIpLiteral) ?: return null
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return null
        @Suppress("DEPRECATION") // allNetworks is the simplest synchronous snapshot
        for (network in cm.allNetworks) {
            val caps = cm.getNetworkCapabilities(network) ?: continue
            // VPNs (e.g. Tailscale) are reached through normal routing.
            if (caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN)) continue
            val lp = cm.getLinkProperties(network) ?: continue
            if (lp.linkAddresses.any { prefixContains(it.address.address, it.prefixLength, target) }) {
                return network
            }
        }
        return null
    }

    fun hostOf(serverUrl: String): String? = try {
        URI(normalizeServerUrl(serverUrl)).host?.removePrefix("[")?.removeSuffix("]")
    } catch (_: Exception) {
        null
    }
}

/** Parses an IPv4/IPv6 literal without any DNS lookup; null for host names. */
fun parseIpLiteral(host: String): ByteArray? {
    val parts = host.split('.')
    if (parts.size == 4 && parts.all { p -> p.isNotEmpty() && p.length <= 3 && p.all(Char::isDigit) && p.toInt() <= 255 }) {
        return ByteArray(4) { parts[it].toInt().toByte() }
    }
    if (host.contains(':') && host.all { it.isLetterOrDigit() || it == ':' || it == '.' }) {
        return try {
            InetAddress.getByName(host).address // literal: no lookup happens
        } catch (_: Exception) {
            null
        }
    }
    return null
}

/** True when [address] lies within [network]/[prefixLength] (same address family only). */
fun prefixContains(network: ByteArray, prefixLength: Int, address: ByteArray): Boolean {
    if (network.size != address.size || prefixLength < 0 || prefixLength > network.size * 8) return false
    val fullBytes = prefixLength / 8
    for (i in 0 until fullBytes) if (network[i] != address[i]) return false
    val remBits = prefixLength % 8
    if (remBits == 0) return true
    val mask = (0xFF shl (8 - remBits)) and 0xFF
    return (network[fullBytes].toInt() and mask) == (address[fullBytes].toInt() and mask)
}
