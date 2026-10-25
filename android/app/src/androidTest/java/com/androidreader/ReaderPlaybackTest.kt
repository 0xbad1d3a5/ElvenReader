package com.androidreader

import android.content.ComponentName
import android.net.Uri
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.androidreader.library.BookPosition
import com.androidreader.library.Library
import com.androidreader.net.PronEntry
import com.androidreader.net.ServerClient
import com.androidreader.net.ServerSettings
import com.androidreader.player.PlaybackHub
import com.androidreader.player.PlaybackService
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Reads a real EPUB (Alice's Adventures in Wonderland, Project Gutenberg) aloud
 * through the playback service against the PC server the app is paired with.
 *
 *   adb shell am instrument -w -e class com.androidreader.ReaderPlaybackTest \
 *     com.androidreader.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class ReaderPlaybackTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private var controller: MediaController? = null

    @Before
    fun setUp() {
        assumeTrue("the app is not paired with a PC", ServerSettings(context).isConfigured)
        val token = SessionToken(context, ComponentName(context, PlaybackService::class.java))
        val future = instrumentation.runOnMainSyncFuture { MediaController.Builder(context, token).buildAsync() }
        controller = future.get(20, TimeUnit.SECONDS)
        waitFor(20) { PlaybackHub.player != null }
    }

    @After
    fun tearDown() {
        instrumentation.runOnMainSync {
            PlaybackHub.pause()
            controller?.release()
        }
    }

    private fun <T> android.app.Instrumentation.runOnMainSyncFuture(block: () -> com.google.common.util.concurrent.ListenableFuture<T>): com.google.common.util.concurrent.ListenableFuture<T> {
        var f: com.google.common.util.concurrent.ListenableFuture<T>? = null
        runOnMainSync { f = block() }
        return f!!
    }

    private fun waitFor(seconds: Int, condition: () -> Boolean) {
        val deadline = System.currentTimeMillis() + seconds * 1000L
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return
            Thread.sleep(200)
        }
        throw AssertionError("condition not met in ${seconds}s; state=${PlaybackHub.state.value}")
    }

    private fun importAlice(): String {
        val file = File(context.cacheDir, "alice.epub")
        InstrumentationRegistry.getInstrumentation().context.assets.open("alice.epub").use { input ->
            file.outputStream().use { input.copyTo(it) }
        }
        return runBlocking { Library.get(context).import(Uri.fromFile(file)) }.id
    }

    @Test
    fun readsTheMadTeaPartyAloudAndFollowsPronunciationChanges() {
        val bookId = importAlice()
        val library = Library.get(context)
        val book = library.content(bookId)
        assertTrue(book.meta.title, book.meta.title.startsWith("Alice"))
        val chapter = book.chapters.indexOfFirst { it.title.contains("Mad Tea-Party") }
        assertTrue("chapter list: ${book.chapters.map { it.title }}", chapter >= 0)

        val t0 = System.currentTimeMillis()
        instrumentation.runOnMainSync { PlaybackHub.open(bookId, BookPosition(chapter, 1, 0), play = true) }
        waitFor(240) { PlaybackHub.state.value.playing }
        val firstAudioMs = System.currentTimeMillis() - t0
        val start = PlaybackHub.state.value.index
        waitFor(120) { PlaybackHub.state.value.index >= start + 3 }
        assertNull(PlaybackHub.state.value.error)
        assertEquals(chapter, PlaybackHub.state.value.chapter)

        // A pronunciation saved mid-chapter re-synthesizes what is buffered; playback continues.
        val client = ServerClient.get(context)
        val entry = client.savePronunciation(PronEntry("", "Dormouse", "door mouse", "", false, bookId))
        try {
            instrumentation.runOnMainSync { PlaybackHub.invalidateUpcoming() }
            val after = PlaybackHub.state.value.index
            waitFor(120) { PlaybackHub.state.value.index >= after + 2 }
            assertNull(PlaybackHub.state.value.error)
            assertTrue(PlaybackHub.state.value.generation > 0)
        } finally {
            client.deletePronunciation(entry.id)
        }

        // The position is saved per book.
        val saved = library.get(bookId)!!.position
        assertEquals(chapter, saved.chapter)
        assertTrue(saved.paragraph >= 1)
        println("ReaderPlaybackTest: first audio after ${firstAudioMs}ms, readyAhead=${PlaybackHub.state.value.readyAhead}")
    }
}
