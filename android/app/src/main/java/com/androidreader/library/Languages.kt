package com.androidreader.library

import java.util.Locale
import java.util.MissingResourceException

/**
 * Maps an EPUB's language code (ISO 639-1/639-2, e.g. "en", "eng", "en-GB") to the
 * English language name the server's models expect, e.g. "English".
 */
object Languages {
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
    fun toLocale(language: String?): Locale? {
        val lang = language?.trim()?.substringBefore('-')?.substringBefore('_')?.lowercase(Locale.ROOT).orEmpty()
        if (lang.isEmpty()) return null
        val base = when {
            lang.length == 2 && lang in iso2Languages -> lang
            lang.length == 3 -> iso3ToIso2[lang] ?: lang.takeIf { isKnownThreeLetter(it) }
            else -> null
        } ?: return null
        return Locale(base)
    }

    private fun isKnownThreeLetter(code: String): Boolean {
        val name = Locale(code).getDisplayLanguage(Locale.ENGLISH)
        return name.isNotBlank() && !name.equals(code, ignoreCase = true)
    }

    /** English display name the server understands, e.g. "English", "German". */
    fun displayName(language: String?): String? =
        toLocale(language)?.getDisplayLanguage(Locale.ENGLISH)?.takeIf { it.isNotBlank() }
}
