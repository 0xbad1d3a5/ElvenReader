package com.androidreader.tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import com.androidreader.R
import java.util.Locale

/**
 * Answers the framework's "is voice data installed?" check. Synthesis happens on the
 * PC, so there is nothing to download: report every language we accept as available.
 */
class CheckVoiceDataActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val available = ArrayList<String>()
        for (locale in Locale.getAvailableLocales()) {
            val tag = try {
                listOf(locale.isO3Language, locale.isO3Country).filter { it.isNotEmpty() }.joinToString("-")
            } catch (_: Exception) {
                continue
            }
            if (tag.isNotEmpty() && tag !in available &&
                LanguageMapper.availability(locale.isO3Language, locale.isO3Country, null) >= TextToSpeech.LANG_AVAILABLE
            ) {
                available.add(tag)
            }
        }
        val result = Intent()
            .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, available)
            .putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, ArrayList())
        setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, result)
        finish()
    }
}

/** Supplies the "Listen to an example" text in the system TTS settings. */
class GetSampleTextActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val lang = LanguageMapper.toLocale(intent?.getStringExtra("language"))?.language
        val text = SAMPLES[lang] ?: getString(R.string.sample_text)
        setResult(
            TextToSpeech.LANG_AVAILABLE,
            Intent().putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, text),
        )
        finish()
    }

    private companion object {
        val SAMPLES = mapOf(
            "de" to "Hallo! Hier liest AndroidReader mit einer Stimme von deinem Computer vor.",
            "es" to "\u00a1Hola! AndroidReader lee con una voz de tu ordenador.",
            "fr" to "Bonjour ! AndroidReader lit avec une voix venue de votre ordinateur.",
            "it" to "Ciao! AndroidReader legge con una voce dal tuo computer.",
            "nl" to "Hallo! AndroidReader leest voor met een stem van je computer.",
            "pt" to "Ol\u00e1! O AndroidReader l\u00ea com uma voz do seu computador.",
        )
    }
}
