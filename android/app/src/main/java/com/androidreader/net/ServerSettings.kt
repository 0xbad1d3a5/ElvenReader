package com.androidreader.net

import android.content.Context
import androidx.core.content.edit

/** Persistent settings shared by the reader, the system TTS engine and the settings screens. */
class ServerSettings(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("androidreader", Context.MODE_PRIVATE)

    // ------------------------------------------------------------ connection

    /** Server URLs in failover order, e.g. LAN first, then Tailscale. */
    var urls: List<String>
        get() = prefs.getString(KEY_URLS, "").orEmpty().split('\n').filter { it.isNotBlank() }
        set(value) = prefs.edit { putString(KEY_URLS, value.joinToString("\n")) }

    var token: String
        get() = prefs.getString(KEY_TOKEN, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_TOKEN, value.trim()) }

    val isConfigured: Boolean get() = urls.isNotEmpty() && token.isNotBlank()

    // ------------------------------------------------------------ system TTS engine (ReadEra etc.)

    /** Engine id for the system engine; empty = the server's default engine. */
    var engineId: String
        get() = prefs.getString(KEY_ENGINE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_ENGINE, value) }

    var voiceId: String
        get() = prefs.getString(KEY_VOICE_ID, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_VOICE_ID, value) }

    var voiceName: String
        get() = prefs.getString(KEY_VOICE_NAME, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_VOICE_NAME, value) }

    /** Optional delivery/style for every system-engine sentence, e.g. "calm, warm". */
    var delivery: String
        get() = prefs.getString(KEY_DELIVERY, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_DELIVERY, value) }

    /** Lightly dramatize quoted exclamations/questions for the system engine. */
    var autoDelivery: Boolean
        get() = prefs.getBoolean(KEY_AUTO_DELIVERY, true)
        set(value) = prefs.edit { putBoolean(KEY_AUTO_DELIVERY, value) }

    var languageMode: String
        get() = prefs.getString(KEY_LANGUAGE_MODE, LANGUAGE_FROM_READER).orEmpty()
        set(value) = prefs.edit { putString(KEY_LANGUAGE_MODE, value) }

    var fixedLanguage: String
        get() = prefs.getString(KEY_FIXED_LANGUAGE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_FIXED_LANGUAGE, value) }

    /** Multiplier applied on top of the reading app's own speech rate. */
    var speed: Float
        get() = prefs.getFloat(KEY_SPEED, 1.0f)
        set(value) = prefs.edit { putFloat(KEY_SPEED, value) }

    var sentencePauseMs: Int
        get() = prefs.getInt(KEY_SENTENCE_PAUSE, DEFAULT_SENTENCE_PAUSE_MS)
        set(value) = prefs.edit { putInt(KEY_SENTENCE_PAUSE, value) }

    val lastError: String
        get() = prefs.getString(KEY_LAST_ERROR, "").orEmpty()

    fun recordError(message: String?) = prefs.edit { putString(KEY_LAST_ERROR, message.orEmpty()) }

    fun languageHint(requestLanguage: String?): String? = when (languageMode) {
        LANGUAGE_AUTO -> null
        LANGUAGE_FIXED -> fixedLanguage.trim().ifEmpty { null }
        else -> com.androidreader.tts.LanguageMapper.displayName(requestLanguage)
    }

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

    /** Sentences synthesized ahead of the one playing. Quality models need a deep buffer. */
    var prefetchCount: Int
        get() = prefs.getInt(KEY_PREFETCH, 6)
        set(value) = prefs.edit { putInt(KEY_PREFETCH, value) }

    /** Engine/voice used for books that don't choose their own. */
    var readerEngineId: String
        get() = prefs.getString(KEY_READER_ENGINE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_READER_ENGINE, value) }

    var readerVoiceId: String
        get() = prefs.getString(KEY_READER_VOICE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_READER_VOICE, value) }

    companion object {
        const val LANGUAGE_FROM_READER = "reader"
        const val LANGUAGE_AUTO = "auto"
        const val LANGUAGE_FIXED = "fixed"
        const val DEFAULT_SENTENCE_PAUSE_MS = 300

        private const val KEY_URLS = "server_urls"
        private const val KEY_TOKEN = "token"
        private const val KEY_ENGINE = "engine_id"
        private const val KEY_VOICE_ID = "voice_id"
        private const val KEY_VOICE_NAME = "voice_name"
        private const val KEY_DELIVERY = "delivery"
        private const val KEY_AUTO_DELIVERY = "auto_delivery"
        private const val KEY_LANGUAGE_MODE = "language_mode"
        private const val KEY_FIXED_LANGUAGE = "fixed_language"
        private const val KEY_SPEED = "speed"
        private const val KEY_SENTENCE_PAUSE = "sentence_pause_ms"
        private const val KEY_LAST_ERROR = "last_error"
        private const val KEY_FONT = "reader_font_sp"
        private const val KEY_THEME = "reader_theme"
        private const val KEY_READER_SPEED = "reader_speed"
        private const val KEY_PREFETCH = "prefetch_count"
        private const val KEY_READER_ENGINE = "reader_engine"
        private const val KEY_READER_VOICE = "reader_voice"
    }
}
