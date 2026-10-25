package com.androidreader.net

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import java.net.InetAddress
import java.net.URI

/**
 * Picks the network that can actually reach a LAN server URL.
 *
 * Phones often keep Wi-Fi connected while routing traffic over mobile data
 * (Samsung "Switch to mobile data", Android's own fallback when Wi-Fi looks
 * weak). A LAN address such as 192.168.1.20 is unreachable over mobile data, so
 * when the server host is on a local subnet we bind our sockets to the network
 * that owns that subnet instead of trusting the system default. Addresses
 * reached through a VPN are left to normal routing.
 */
object LocalNetworks {
    /** Set once binding was refused (e.g. a VPN that forbids bypass); stop trying afterwards. */
    @Volatile var bindingRefused = false

    fun networkFor(context: Context, serverUrl: String): Network? {
        if (bindingRefused) return null
        val target = hostOf(serverUrl)?.let(::parseIpLiteral) ?: return null
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return null
        // Behind a VPN, routing is the VPN's decision and binding around it is usually refused.
        val active = cm.activeNetwork?.let { cm.getNetworkCapabilities(it) }
        if (active?.hasTransport(NetworkCapabilities.TRANSPORT_VPN) == true) return null
        @Suppress("DEPRECATION") // allNetworks is the simplest synchronous snapshot
        for (network in cm.allNetworks) {
            val caps = cm.getNetworkCapabilities(network) ?: continue
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

/** True when Android refused to bind a socket to a network (e.g. "bind ... failed: EPERM"). */
fun isBindRefusal(e: Throwable): Boolean {
    var t: Throwable? = e
    while (t != null) {
        if (t.message?.contains("EPERM") == true) return true
        t = t.cause
    }
    return false
}

/**
 * Accepts what a user types ("192.168.1.5", "pc:3950", "http://pc:3950/")
 * and returns a base URL like "http://192.168.1.5:3950".
 */
fun normalizeServerUrl(input: String, defaultPort: Int = 3950): String {
    var s = input.trim().trimEnd('/')
    if (s.isEmpty()) return ""
    if (!s.contains("://")) s = "http://$s"
    val schemeEnd = s.indexOf("://") + 3
    val hostPart = s.substring(schemeEnd).substringBefore('/')
    val hasPort = if (hostPart.startsWith("[")) hostPart.contains("]:") else hostPart.contains(':')
    val path = s.substring(schemeEnd + hostPart.length)
    val host = if (hasPort) hostPart else "$hostPart:$defaultPort"
    return s.substring(0, schemeEnd) + host + path
}

/** Splits user input (one URL per line, or comma-separated) into normalised, de-duplicated URLs. */
fun parseServerUrls(input: String): List<String> =
    input.split('\n', ',', ' ').map { normalizeServerUrl(it) }.filter { it.isNotEmpty() }.distinct()

/** "lan" for addresses on a private local network (failover tries these first), otherwise "remote". */
fun urlKind(url: String): String {
    val ip = LocalNetworks.hostOf(url)?.let(::parseIpLiteral) ?: return "remote"
    if (ip.size == 4) {
        val a = ip[0].toInt() and 0xFF
        val b = ip[1].toInt() and 0xFF
        if (a == 10 || (a == 172 && b in 16..31) || (a == 192 && b == 168)) return "lan"
    } else if ((ip[0].toInt() and 0xFE) == 0xFC) {
        return "lan"
    }
    return "remote"
}
