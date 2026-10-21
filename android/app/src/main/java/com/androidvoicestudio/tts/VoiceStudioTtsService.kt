package com.androidvoicestudio.tts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.util.Log
import okhttp3.Call
import java.io.IOException
import java.io.InterruptedIOException
import java.net.SocketTimeoutException
import java.util.Locale

/**
 * System text-to-speech engine backed by VoiceStudio running on a PC.
 *
 * Each utterance (ReadEra sends roughly one sentence at a time) is POSTed to the
 * relay, which streams PCM16 mono back from VoiceStudio's /ws/tts. Audio is passed
 * to the framework as it arrives, so playback starts after the first sentence
 * chunk is synthesized rather than after the whole utterance.
 */
class VoiceStudioTtsService : TextToSpeechService() {
    private lateinit var settings: TtsSettings

    @Volatile private var currentCall: Call? = null
    @Volatile private var stopRequested = false
    @Volatile private var language = arrayOf("eng", "", "")

    override fun onCreate() {
        // TextToSpeechService.onCreate() calls onLoadLanguage(), so settings must exist first.
        settings = TtsSettings(this)
        val default = Locale.getDefault()
        language = try {
            arrayOf(default.isO3Language, default.isO3Country, default.variant)
        } catch (_: Exception) {
            arrayOf("eng", "", "")
        }
        super.onCreate()
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int =
        LanguageMapper.availability(lang, country, variant)

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        val result = LanguageMapper.availability(lang, country, variant)
        if (result >= TextToSpeech.LANG_AVAILABLE) {
            language = arrayOf(lang.orEmpty(), country.orEmpty(), variant.orEmpty())
        }
        return result
    }

    override fun onGetLanguage(): Array<String> = language

    override fun onStop() {
        stopRequested = true
        currentCall?.cancel()
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        stopRequested = false
        val text = request.charSequenceText?.toString() ?: request.text.orEmpty()

        if (!isSpeakable(text)) {
            // Lone punctuation / whitespace: complete instantly with no audio.
            if (callback.start(DEFAULT_SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1) == TextToSpeech.SUCCESS) {
                callback.done()
            }
            return
        }
        if (!settings.isConfigured) {
            fail(callback, TextToSpeech.ERROR_NOT_INSTALLED_YET, "Not paired with a relay yet")
            return
        }

        val rate = request.speechRate.takeIf { it > 0 } ?: 100
        val params = SynthesisParams(
            text = text,
            voice = settings.voiceId,
            language = settings.languageHint(request.language),
            speed = (settings.speed * rate / 100f).coerceIn(0.5f, 2.5f),
            instruct = settings.instruct,
        )
        val call = RelayClient.create(this, settings.serverUrl, settings.token).ttsCall(params)
        currentCall = call
        if (stopRequested) call.cancel()

        try {
            call.execute().use { resp ->
                RelayClient.ensureSuccess(resp)
                val sampleRate = resp.header("X-Sample-Rate")?.toIntOrNull() ?: DEFAULT_SAMPLE_RATE
                if (callback.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) {
                    return
                }
                val chunker = PcmChunker(callback.maxBufferSize) { buf, off, len ->
                    callback.audioAvailable(buf, off, len) == TextToSpeech.SUCCESS
                }
                val input = resp.body?.byteStream() ?: throw IOException("empty response")
                val buf = ByteArray(READ_BUFFER)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    if (!chunker.feed(buf, n)) return // stopped by the framework
                }
                callback.done()
                if (settings.lastError.isNotEmpty()) settings.recordError(null)
            }
        } catch (e: IOException) {
            if (stopRequested || call.isCanceled()) return
            val code = when (e) {
                is RelayHttpException -> when (e.code) {
                    400, 413 -> TextToSpeech.ERROR_INVALID_REQUEST
                    504 -> TextToSpeech.ERROR_NETWORK_TIMEOUT
                    else -> TextToSpeech.ERROR_SERVICE
                }
                is SocketTimeoutException, is InterruptedIOException -> TextToSpeech.ERROR_NETWORK_TIMEOUT
                else -> TextToSpeech.ERROR_NETWORK
            }
            fail(callback, code, e.message ?: e.javaClass.simpleName)
        } finally {
            currentCall = null
        }
    }

    private fun fail(callback: SynthesisCallback, code: Int, message: String) {
        Log.w(TAG, "Synthesis failed ($code): $message")
        settings.recordError(message)
        callback.error(code)
        // done() is required after error(); without it the client never receives onError.
        callback.done()
    }

    companion object {
        private const val TAG = "VoiceStudioTTS"
        private const val DEFAULT_SAMPLE_RATE = 24000
        private const val READ_BUFFER = 8192
    }
}
