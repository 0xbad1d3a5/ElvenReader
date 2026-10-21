package com.androidvoicestudio.tts

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * End-to-end test of the engine on a device against a running relay + VoiceStudio.
 * Binds to this engine explicitly (the system default engine is left alone) and uses
 * synthesizeToFile so nothing is played aloud.
 *
 *   gradlew connectedDebugAndroidTest
 *     -Pandroid.testInstrumentationRunnerArguments.relayUrl=http://192.168.1.20:3950
 *     -Pandroid.testInstrumentationRunnerArguments.relayToken=<token>
 */
@RunWith(AndroidJUnit4::class)
class EngineInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val settings = TtsSettings(context)
    private lateinit var tts: TextToSpeech
    private lateinit var saved: Pair<String, String>

    private class Outcome {
        val latch = CountDownLatch(1)
        @Volatile var error: Int? = null
        @Volatile var stopped = false
    }

    @Before
    fun setUp() {
        val args = InstrumentationRegistry.getArguments()
        val url = args.getString("relayUrl")
        val token = args.getString("relayToken")
        assumeTrue("relayUrl/relayToken instrumentation arguments not given", url != null && token != null)
        saved = settings.serverUrl to settings.token
        settings.serverUrl = url!!
        settings.token = token!!

        val ready = CountDownLatch(1)
        var status = TextToSpeech.ERROR
        instrumentation.runOnMainSync {
            tts = TextToSpeech(context, { status = it; ready.countDown() }, context.packageName)
        }
        assertTrue("engine did not bind", ready.await(15, TimeUnit.SECONDS))
        assertEquals(TextToSpeech.SUCCESS, status)
    }

    @After
    fun tearDown() {
        if (::tts.isInitialized) tts.shutdown()
        if (::saved.isInitialized) {
            settings.serverUrl = saved.first
            settings.token = saved.second
            settings.recordError(null) // don't leave test failures in the user's "Last error"
        }
    }

    private fun synthesize(text: String, file: File, id: String): Outcome {
        val outcome = Outcome()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) = outcome.latch.countDown()
            override fun onStop(utteranceId: String?, interrupted: Boolean) {
                outcome.stopped = true
                outcome.latch.countDown()
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) = onError(utteranceId, TextToSpeech.ERROR)
            override fun onError(utteranceId: String?, errorCode: Int) {
                outcome.error = errorCode
                outcome.latch.countDown()
            }
        })
        assertEquals(TextToSpeech.SUCCESS, tts.synthesizeToFile(text, Bundle(), file, id))
        return outcome
    }

    @Test
    fun reportsLanguagesAsAvailable() {
        assertTrue(tts.isLanguageAvailable(Locale.US) >= TextToSpeech.LANG_AVAILABLE)
        assertTrue(tts.isLanguageAvailable(Locale.GERMANY) >= TextToSpeech.LANG_AVAILABLE)
        assertTrue(tts.setLanguage(Locale.UK) >= TextToSpeech.LANG_AVAILABLE)
        assertTrue(tts.voices.orEmpty().isNotEmpty())
    }

    @Test
    fun synthesizesThroughRelay() {
        tts.setLanguage(Locale.US)
        val file = File(context.cacheDir, "engine-test.wav").apply { delete() }
        val outcome = synthesize("Hello from the VoiceStudio engine test on this phone.", file, "ok")
        assertTrue("timed out", outcome.latch.await(90, TimeUnit.SECONDS))
        assertNull("error ${outcome.error}: ${settings.lastError}", outcome.error)
        // WAV header + at least one second of 24 kHz 16-bit mono audio.
        assertTrue("only ${file.length()} bytes", file.length() > 44 + 48000)
    }

    @Test
    fun stopCancelsPromptly() {
        val longText = (1..15).joinToString(" ") { "This is sentence number $it of a long paragraph." }
        val outcome = synthesize(longText, File(context.cacheDir, "engine-stop.wav"), "stop")
        Thread.sleep(1500)
        assertEquals("finished before stop", 1L, outcome.latch.count)
        val t0 = System.nanoTime()
        tts.stop()
        assertTrue("stop did not complete", outcome.latch.await(5, TimeUnit.SECONDS))
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertTrue("not reported as stopped (error ${outcome.error})", outcome.stopped)
        assertTrue("stop took ${ms}ms", ms < 2000)
    }

    @Test
    fun rejectedTokenFailsCleanly() {
        settings.token = "wrong-token"
        val outcome = synthesize("This should be rejected.", File(context.cacheDir, "engine-401.wav"), "401")
        assertTrue("hung", outcome.latch.await(15, TimeUnit.SECONDS))
        assertEquals(TextToSpeech.ERROR_SERVICE, outcome.error)
    }

    @Test
    fun unreachablePcFailsCleanly() {
        settings.serverUrl = "http://192.0.2.1:3950" // TEST-NET-1: never routable
        val t0 = System.nanoTime()
        val outcome = synthesize("This should fail quickly.", File(context.cacheDir, "engine-fail.wav"), "fail")
        assertTrue("hung", outcome.latch.await(15, TimeUnit.SECONDS))
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertTrue("expected an error", outcome.error != null)
        assertTrue("error not recorded for the settings screen", settings.lastError.isNotBlank())
        assertTrue("took ${ms}ms", ms < 10_000)
    }
}
