package com.androidreader.net

import android.content.Context
import androidx.core.content.edit

/** Persistent app settings: the connection to the PC and the reader's preferences. */
class ServerSettings(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("androidreader", Context.MODE_PRIVATE)

    // ------------------------------------------------------------ connection

    /** Server URLs in failover order, e.g. the home network first, then the PC's address on your VPN. */
    var urls: List<String>
        get() = prefs.getString(KEY_URLS, "").orEmpty().split('\n').filter { it.isNotBlank() }
        set(value) = prefs.edit { putString(KEY_URLS, value.joinToString("\n")) }

    var token: String
        get() = prefs.getString(KEY_TOKEN, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_TOKEN, value.trim()) }

    val isConfigured: Boolean get() = urls.isNotEmpty() && token.isNotBlank()

    // ------------------------------------------------------------ reader

    var readerFontSp: Float
        get() = prefs.getFloat(KEY_FONT, 19f)
        set(value) = prefs.edit { putFloat(KEY_FONT, value) }

    /** "system", "light", "dark" or "sepia". */
    var readerTheme: String
        get() = prefs.getString(KEY_THEME, "system").orEmpty()
        set(value) = prefs.edit { putString(KEY_THEME, value) }

    var readerSpeed: Float
        get() = prefs.getFloat(KEY_READER_SPEED, 1.0f)
        set(value) = prefs.edit { putFloat(KEY_READER_SPEED, value) }

    /** Silence after a sentence; paragraph ends get a little more. */
    var sentencePauseMs: Int
        get() = prefs.getInt(KEY_SENTENCE_PAUSE, DEFAULT_SENTENCE_PAUSE_MS)
        set(value) = prefs.edit { putInt(KEY_SENTENCE_PAUSE, value) }

    /** Sentences synthesized ahead of the one playing. Quality models need a deep buffer. */
    var prefetchCount: Int
        get() = prefs.getInt(KEY_PREFETCH, 6)
        set(value) = prefs.edit { putInt(KEY_PREFETCH, value) }

    /** Engine/voice used for books that don't choose their own; empty = the server's default. */
    var readerEngineId: String
        get() = prefs.getString(KEY_READER_ENGINE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_READER_ENGINE, value) }

    var readerVoiceId: String
        get() = prefs.getString(KEY_READER_VOICE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_READER_VOICE, value) }

    companion object {
        const val DEFAULT_SENTENCE_PAUSE_MS = 300

        private const val KEY_URLS = "server_urls"
        private const val KEY_TOKEN = "token"
        private const val KEY_SENTENCE_PAUSE = "sentence_pause_ms"
        private const val KEY_FONT = "reader_font_sp"
        private const val KEY_THEME = "reader_theme"
        private const val KEY_READER_SPEED = "reader_speed"
        private const val KEY_PREFETCH = "prefetch_count"
        private const val KEY_READER_ENGINE = "reader_engine"
        private const val KEY_READER_VOICE = "reader_voice"
    }
}
