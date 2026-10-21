package com.androidvoicestudio.tts

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Times any installed TTS engine through the standard Android API, sentence by
 * sentence like a reading app. Results go to logcat (tag EngineBenchmark) and the
 * instrumentation status output.
 *
 *   adb shell am instrument -w -r -e benchmarkEngine com.k2fsa.sherpa.onnx.tts.engine \
 *     -e class com.androidvoicestudio.tts.EngineBenchmark \
 *     com.androidvoicestudio.tts.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class EngineBenchmark {
    private val sentences = listOf(
        "The valley had been quiet for a hundred years.",
        "Tonight, for the first time, something stirred beneath the old stone bridge.",
        "The old man set down his lantern and looked out across the silent water.",
        "He had heard the stories as a boy, of course, but he had never believed them.",
        "Now, standing in the cold, he was no longer so sure.",
    )

    @Test
    fun benchmark() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val engine = InstrumentationRegistry.getArguments().getString("benchmarkEngine")
        assumeTrue("benchmarkEngine argument not given", engine != null)
        val context = instrumentation.targetContext

        val ready = CountDownLatch(1)
        var status = TextToSpeech.ERROR
        lateinit var tts: TextToSpeech
        instrumentation.runOnMainSync { tts = TextToSpeech(context, { status = it; ready.countDown() }, engine) }
        assertTrue("engine did not bind", ready.await(60, TimeUnit.SECONDS))
        assertEquals(TextToSpeech.SUCCESS, status)
        tts.setLanguage(Locale.UK)

        val report = StringBuilder("engine=$engine\n")
        var totalSynth = 0.0
        var totalAudio = 0.0
        var totalTtfa = 0.0
        try {
            sentences.forEachIndexed { i, text ->
                val file = File(context.cacheDir, "bench-$i.wav").apply { delete() }
                val done = CountDownLatch(1)
                var error: Int? = null
                var firstAudioNs = 0L
                tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onAudioAvailable(utteranceId: String?, audio: ByteArray?) {
                        if (firstAudioNs == 0L) firstAudioNs = System.nanoTime()
                    }
                    override fun onDone(utteranceId: String?) = done.countDown()
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) = onError(utteranceId, -1)
                    override fun onError(utteranceId: String?, errorCode: Int) { error = errorCode; done.countDown() }
                })
                val t0 = System.nanoTime()
                assertEquals(TextToSpeech.SUCCESS, tts.synthesizeToFile(text, Bundle(), file, "b$i"))
                assertTrue("timed out", done.await(120, TimeUnit.SECONDS))
                val secs = (System.nanoTime() - t0) / 1e9
                assertEquals("error", null, error)
                val audio = wavSeconds(file)
                val ttfa = if (firstAudioNs > 0) (firstAudioNs - t0) / 1e9 else Double.NaN
                if (i > 0) { // first sentence includes model warm-up
                    totalSynth += secs
                    totalAudio += audio
                    totalTtfa += ttfa
                }
                report.append(String.format(Locale.ROOT, "sentence %d: first audio %.2fs, %.2fs to synthesize %.2fs of audio%s\n",
                    i + 1, ttfa, secs, audio, if (i == 0) " (includes warm-up)" else ""))
            }
            report.append(String.format(Locale.ROOT, "warm average: first audio %.2fs, RTF %.3f (%.2fs per sentence)\n",
                totalTtfa / (sentences.size - 1), totalSynth / totalAudio, totalSynth / (sentences.size - 1)))
        } finally {
            tts.shutdown()
        }
        Log.i("EngineBenchmark", report.toString())
        instrumentation.sendStatus(0, Bundle().apply { putString("stream", report.toString()) })
    }

    /** Duration from a WAV header: data bytes / byte rate. */
    private fun wavSeconds(file: File): Double {
        val b = file.readBytes()
        fun le32(o: Int) = (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8) or
            ((b[o + 2].toInt() and 0xff) shl 16) or ((b[o + 3].toInt() and 0xff) shl 24)
        val byteRate = le32(28)
        var pos = 12
        while (pos + 8 <= b.size) {
            val id = String(b, pos, 4, Charsets.US_ASCII)
            val size = le32(pos + 4)
            if (id == "data") return minOf(size, b.size - pos - 8).toDouble() / byteRate
            pos += 8 + size
        }
        return (b.size - 44).toDouble() / byteRate
    }
}
