package com.androidvoicestudio.tts

import android.speech.tts.TextToSpeech
import java.util.Locale
import java.util.MissingResourceException

/**
 * Maps the ISO 639 / ISO 3166 codes the TTS framework hands us to the
 * language names VoiceStudio (OmniVoice) expects, e.g. "eng" -> "English".
 *
 * OmniVoice is zero-shot across hundreds of languages, so every real
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

    /** English display name VoiceStudio understands, e.g. "English", "German". */
    fun displayName(language: String?): String? =
        toLocale(language)?.getDisplayLanguage(Locale.ENGLISH)?.takeIf { it.isNotBlank() }
}

/** True when there is anything worth synthesizing (at least one letter or digit). */
fun isSpeakable(text: CharSequence): Boolean = text.any { it.isLetterOrDigit() }

/**
 * Accepts what a user types ("192.168.1.5", "pc:3950", "http://pc:3950/")
 * and returns a base URL like "http://192.168.1.5:3950".
 */
fun normalizeServerUrl(input: String, defaultPort: Int = 3950): String {
    var s = input.trim().trimEnd('/')
    if (s.isEmpty()) return ""
    if (!s.contains("://")) s = "http://$s"
    val schemeEnd = s.indexOf("://") + 3
    val hostPart = s.substring(schemeEnd).substringBefore('/')
    val hasPort = if (hostPart.startsWith("[")) hostPart.contains("]:") else hostPart.contains(':')
    val path = s.substring(schemeEnd + hostPart.length)
    val host = if (hasPort) hostPart else "$hostPart:$defaultPort"
    return s.substring(0, schemeEnd) + host + path
}
