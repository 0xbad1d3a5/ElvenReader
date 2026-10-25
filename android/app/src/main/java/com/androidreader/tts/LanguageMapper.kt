package com.androidreader.tts

import android.speech.tts.TextToSpeech
import java.util.Locale
import java.util.MissingResourceException

/**
 * Maps the ISO 639 / ISO 3166 codes the TTS framework hands us to the
 * language names the TTS models expect, e.g. "eng" -> "English".
 *
 * The server's engines are zero-shot multilingual, so every real
 * language code is reported as available; unknown codes are rejected so
 * readers can still fall back to another engine for garbage input.
 */
object LanguageMapper {
    private val iso2Languages: Set<String> by lazy { Locale.getISOLanguages().toSet() }

    private val iso3ToIso2: Map<String, String> by lazy {
        Locale.getISOLanguages().mapNotNull { code ->
            try {
                Locale(code).isO3Language to code
            } catch (_: MissingResourceException) {
                null
            }
        }.toMap()
    }

    /** Normalises a 2- or 3-letter language code to a [Locale], or null if unknown. */
    fun toLocale(language: String?, country: String? = null, variant: String? = null): Locale? {
        val lang = language?.trim()?.lowercase(Locale.ROOT).orEmpty()
        if (lang.isEmpty()) return null
        val base = when {
            lang.length == 2 && lang in iso2Languages -> lang
            lang.length == 3 -> iso3ToIso2[lang] ?: lang.takeIf { isKnownThreeLetter(it) }
            else -> null
        } ?: return null
        return Locale(base, country?.trim().orEmpty(), variant?.trim().orEmpty())
    }

    private fun isKnownThreeLetter(code: String): Boolean {
        val name = Locale(code).getDisplayLanguage(Locale.ENGLISH)
        return name.isNotBlank() && !name.equals(code, ignoreCase = true)
    }

    /** Result for [android.speech.tts.TextToSpeechService.onIsLanguageAvailable]. */
    fun availability(language: String?, country: String?, variant: String?): Int {
        toLocale(language) ?: return TextToSpeech.LANG_NOT_SUPPORTED
        return when {
            country.isNullOrBlank() -> TextToSpeech.LANG_AVAILABLE
            variant.isNullOrBlank() -> TextToSpeech.LANG_COUNTRY_AVAILABLE
            else -> TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE
        }
    }

    /** English display name the server understands, e.g. "English", "German". */
    fun displayName(language: String?): String? =
        toLocale(language)?.getDisplayLanguage(Locale.ENGLISH)?.takeIf { it.isNotBlank() }
}

/** True when there is anything worth synthesizing (at least one letter or digit). */
fun isSpeakable(text: CharSequence): Boolean = text.any { it.isLetterOrDigit() }
