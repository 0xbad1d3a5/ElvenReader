package com.androidreader

import com.androidreader.library.Languages
import com.androidreader.net.LocalNetworks
import com.androidreader.net.isBindRefusal
import com.androidreader.net.normalizeServerUrl
import com.androidreader.net.parseIpLiteral
import com.androidreader.net.prefixContains
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NetworkAndLanguageTest {
    @Test
    fun mapsBookLanguageCodesToNames() {
        assertEquals("English", Languages.displayName("eng"))
        assertEquals("English", Languages.displayName("en-GB"))
        assertEquals("German", Languages.displayName("deu"))
        assertEquals("French", Languages.displayName("fr"))
        assertEquals(null, Languages.displayName("xx"))
        assertEquals(null, Languages.displayName(""))
        assertEquals(null, Languages.displayName(null))
    }

    @Test
    fun normalizesServerUrls() {
        assertEquals("http://192.168.1.5:3950", normalizeServerUrl("192.168.1.5"))
        assertEquals("http://192.168.1.5:8000", normalizeServerUrl(" 192.168.1.5:8000/ "))
        assertEquals("http://pc.home.arpa:3950", normalizeServerUrl("http://pc.home.arpa"))
        assertEquals("https://reader.example:443", normalizeServerUrl("https://reader.example:443/"))
        assertEquals("http://[fd00::1]:3950", normalizeServerUrl("[fd00::1]"))
        assertEquals("", normalizeServerUrl("  "))
    }

    @Test
    fun detectsBindRefusal() {
        val refused = java.net.SocketException("Binding socket to network 112 failed: EPERM (Operation not permitted)")
        assertTrue(isBindRefusal(refused))
        assertTrue(isBindRefusal(java.io.IOException("wrapped", refused)))
        assertFalse(isBindRefusal(java.net.ConnectException("failed to connect: ECONNREFUSED")))
    }

    @Test
    fun parsesIpLiteralsWithoutDns() {
        assertArrayEquals(byteArrayOf(192.toByte(), 168.toByte(), 86, 10), parseIpLiteral("192.168.86.10"))
        assertEquals(16, parseIpLiteral("fd00::1")?.size)
        assertEquals(null, parseIpLiteral("my-pc.lan"))
        assertEquals(null, parseIpLiteral("300.1.1.1"))
        assertEquals(null, parseIpLiteral("1.2.3"))
    }

    @Test
    fun prefixMatching() {
        val lan = parseIpLiteral("192.168.86.21")!!
        assertTrue(prefixContains(lan, 24, parseIpLiteral("192.168.86.10")!!))
        assertFalse(prefixContains(lan, 24, parseIpLiteral("192.168.87.10")!!))
        assertTrue(prefixContains(parseIpLiteral("10.77.77.2")!!, 20, parseIpLiteral("10.77.64.1")!!))
        assertFalse(prefixContains(parseIpLiteral("10.77.77.2")!!, 20, parseIpLiteral("10.77.96.1")!!))
        assertFalse(prefixContains(lan, 24, parseIpLiteral("fd00::1")!!))
        assertEquals("192.168.86.10", LocalNetworks.hostOf("192.168.86.10"))
        assertEquals("fd00::1", LocalNetworks.hostOf("[fd00::1]:3950"))
    }
}
