package com.androidvoicestudio.tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
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
            "de" to "Hallo! Dies ist VoiceStudio, das von deinem Computer aus über dein Telefon spricht.",
            "es" to "¡Hola! Soy VoiceStudio, hablando desde tu ordenador a través de tu teléfono.",
            "fr" to "Bonjour ! Ici VoiceStudio, qui parle depuis votre ordinateur via votre téléphone.",
            "it" to "Ciao! Sono VoiceStudio e parlo dal tuo computer attraverso il tuo telefono.",
            "nl" to "Hallo! Dit is VoiceStudio, dat vanaf je computer via je telefoon spreekt.",
            "pt" to "Olá! Aqui é o VoiceStudio, falando do seu computador pelo seu telefone.",
        )
    }
}
