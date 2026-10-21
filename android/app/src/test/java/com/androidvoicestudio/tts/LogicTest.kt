package com.androidvoicestudio.tts

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PcmChunkerTest {
    private class Sink(private val acceptCalls: Int = Int.MAX_VALUE) {
        val blocks = mutableListOf<ByteArray>()
        fun emit(buf: ByteArray, off: Int, len: Int): Boolean {
            if (blocks.size >= acceptCalls) return false
            blocks += buf.copyOfRange(off, off + len)
            return true
        }
        val all: ByteArray get() = blocks.fold(ByteArray(0)) { acc, b -> acc + b }
    }

    @Test
    fun splitsIntoBlocksNoLargerThanMax() {
        val sink = Sink()
        val chunker = PcmChunker(10, sink::emit)
        val data = ByteArray(25) { it.toByte() }
        assertTrue(chunker.feed(data))
        assertTrue(sink.blocks.all { it.size <= 10 && it.size % 2 == 0 })
        assertArrayEquals(data.copyOfRange(0, 24), sink.all)
        assertEquals(1, chunker.pendingBytes)
    }

    @Test
    fun carriesOddByteIntoNextFeed() {
        val sink = Sink()
        val chunker = PcmChunker(4096, sink::emit)
        chunker.feed(byteArrayOf(1, 2, 3))
        chunker.feed(byteArrayOf(4, 5, 6, 7), 3)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 5, 6), sink.all)
        assertTrue(sink.blocks.all { it.size % 2 == 0 })
        assertEquals(0, chunker.pendingBytes)
    }

    @Test
    fun oddMaxBufferIsRoundedDownToWholeSamples() {
        val sink = Sink()
        PcmChunker(7, sink::emit).feed(ByteArray(20))
        assertTrue(sink.blocks.all { it.size <= 6 && it.size % 2 == 0 })
        assertEquals(20, sink.all.size)
    }

    @Test
    fun stopsWhenFrameworkRejectsAudio() {
        val sink = Sink(acceptCalls = 1)
        assertFalse(PcmChunker(4, sink::emit).feed(ByteArray(12)))
        assertEquals(1, sink.blocks.size)
    }
}

class TextTest {
    @Test
    fun mapsIso3AndIso2CodesToVoiceStudioNames() {
        assertEquals("English", LanguageMapper.displayName("eng"))
        assertEquals("German", LanguageMapper.displayName("deu"))
        assertEquals("French", LanguageMapper.displayName("fr"))
        assertEquals(null, LanguageMapper.displayName("xx"))
        assertEquals(null, LanguageMapper.displayName(""))
        assertEquals(null, LanguageMapper.displayName(null))
    }

    @Test
    fun availabilityFollowsTtsContract() {
        assertEquals(0, LanguageMapper.availability("eng", "", "")) // LANG_AVAILABLE
        assertEquals(1, LanguageMapper.availability("eng", "USA", "")) // LANG_COUNTRY_AVAILABLE
        assertEquals(2, LanguageMapper.availability("eng", "USA", "POSIX")) // LANG_COUNTRY_VAR_AVAILABLE
        assertEquals(-2, LanguageMapper.availability("zzz", "", "")) // LANG_NOT_SUPPORTED
        assertEquals(-2, LanguageMapper.availability(null, null, null))
    }

    @Test
    fun speakable() {
        assertTrue(isSpeakable("Hello."))
        assertTrue(isSpeakable("1984"))
        assertFalse(isSpeakable(" ... "))
        assertFalse(isSpeakable(""))
    }

    @Test
    fun normalizesServerUrls() {
        assertEquals("http://192.168.1.5:3950", normalizeServerUrl("192.168.1.5"))
        assertEquals("http://192.168.1.5:8000", normalizeServerUrl(" 192.168.1.5:8000/ "))
        assertEquals("http://pc.tailnet.ts.net:3950", normalizeServerUrl("http://pc.tailnet.ts.net"))
        assertEquals("https://relay.example:443", normalizeServerUrl("https://relay.example:443/"))
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
