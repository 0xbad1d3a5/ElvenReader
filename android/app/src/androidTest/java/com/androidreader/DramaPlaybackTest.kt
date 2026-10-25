package com.androidreader

import android.content.ComponentName
import android.net.Uri
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.androidreader.drama.Dramatization
import com.androidreader.library.BookPosition
import com.androidreader.library.Library
import com.androidreader.net.ServerSettings
import com.androidreader.player.PlaybackHub
import com.androidreader.player.PlaybackService
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Dramatized reading end to end: upload Alice to the PC, wait for the LLM analysis,
 * create character voices, then read the Mad Tea-Party with per-character voices.
 * Slow (the whole book is analysed); run on demand:
 *
 *   adb shell am instrument -w -e class com.androidreader.DramaPlaybackTest \
 *     com.androidreader.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class DramaPlaybackTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    private fun waitFor(seconds: Int, what: String, condition: () -> Boolean) {
        val deadline = System.currentTimeMillis() + seconds * 1000L
        while (System.currentTimeMillis() < deadline) {
            if (condition()) return
            Thread.sleep(1000)
        }
        throw AssertionError("$what not done in ${seconds}s; state=${PlaybackHub.state.value}")
    }

    @Test
    fun analysesCastsAndReadsWithCharacterVoices() {
        assumeTrue("the app is not paired with a PC", ServerSettings(context).isConfigured)
        val file = File(context.cacheDir, "alice.epub")
        instrumentation.context.assets.open("alice.epub").use { input -> file.outputStream().use { input.copyTo(it) } }
        val library = Library.get(context)
        val bookId = runBlocking { library.import(Uri.fromFile(file)) }.id
        library.update(library.get(bookId)!!.copy(dramatized = true))
        Dramatization.init(context)

        val t0 = System.currentTimeMillis()
        Dramatization.ensureAnalysis(bookId)
        waitFor(3 * 3600, "analysis") { Dramatization.refresh(bookId)?.status == "done" }
        val analysisMin = (System.currentTimeMillis() - t0) / 60000.0
        val script = Dramatization.refresh(bookId)!!
        val names = script.characters.map { it.name.lowercase() }
        assertTrue(names.toString(), names.any { "hatter" in it } && names.any { "alice" in it } && names.any { "dormouse" in it })

        Dramatization.castVoices(bookId, null)
        waitFor(3600, "casting") { Dramatization.refresh(bookId)?.status != "casting" }
        val cast = Dramatization.refresh(bookId)!!
        val hatter = cast.characters.first { "hatter" in it.name.lowercase() }
        val alice = cast.characters.first { it.name.lowercase().startsWith("alice") }
        assertNotNull(hatter.voice)
        assertTrue("Alice and the Hatter share a voice", hatter.voice != alice.voice)

        val content = library.content(bookId)
        val chapter = content.chapters.indexOfFirst { it.title.contains("Mad Tea-Party") }
        val token = SessionToken(context, ComponentName(context, PlaybackService::class.java))
        var future: com.google.common.util.concurrent.ListenableFuture<MediaController>? = null
        instrumentation.runOnMainSync { future = MediaController.Builder(context, token).buildAsync() }
        val controller = future!!.get(20, TimeUnit.SECONDS)
        try {
            waitFor(20, "player") { PlaybackHub.player != null }
            instrumentation.runOnMainSync { PlaybackHub.open(bookId, BookPosition(chapter, 3, 0), play = true) }
            assertTrue(PlaybackHub.state.value.dramatized)
            // ¶3 is the March Hare's "Have some wine," — voiced by the March Hare, not the narrator.
            val first = PlaybackHub.requestFor(chapter, PlaybackHub.state.value.index)!!
            val hare = cast.characters.first { "march hare" in it.name.lowercase() }
            assertEquals(hare.voice, first.voice)
            val start = PlaybackHub.state.value.index
            waitFor(600, "playback") { PlaybackHub.state.value.index >= start + 6 }
            assertNull(PlaybackHub.state.value.error)
            println("DramaPlaybackTest: analysis ${"%.1f".format(analysisMin)} min, ${cast.characters.size} characters, ${cast.lines.size} lines")
        } finally {
            instrumentation.runOnMainSync { PlaybackHub.pause(); controller.release() }
        }
    }
}
