package com.androidvoicestudio.tts

import android.content.Context
import androidx.core.content.edit

/** Persistent engine settings, shared by the settings screen and the TTS service. */
class TtsSettings(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("voicestudio_tts", Context.MODE_PRIVATE)

    var serverUrl: String
        get() = prefs.getString(KEY_URL, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_URL, value) }

    var token: String
        get() = prefs.getString(KEY_TOKEN, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_TOKEN, value.trim()) }

    /** VoiceStudio voice profile id; empty means the engine's default voice. */
    var voiceId: String
        get() = prefs.getString(KEY_VOICE_ID, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_VOICE_ID, value) }

    var voiceName: String
        get() = prefs.getString(KEY_VOICE_NAME, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_VOICE_NAME, value) }

    var languageMode: String
        get() = prefs.getString(KEY_LANGUAGE_MODE, LANGUAGE_FROM_READER).orEmpty()
        set(value) = prefs.edit { putString(KEY_LANGUAGE_MODE, value) }

    var fixedLanguage: String
        get() = prefs.getString(KEY_FIXED_LANGUAGE, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_FIXED_LANGUAGE, value) }

    /** Multiplier applied on top of the reader's own speech rate. */
    var speed: Float
        get() = prefs.getFloat(KEY_SPEED, 1.0f)
        set(value) = prefs.edit { putFloat(KEY_SPEED, value) }

    /** Optional OmniVoice style instruction, e.g. "calm, low pitch". */
    var instruct: String
        get() = prefs.getString(KEY_INSTRUCT, "").orEmpty()
        set(value) = prefs.edit { putString(KEY_INSTRUCT, value) }

    val lastError: String
        get() = prefs.getString(KEY_LAST_ERROR, "").orEmpty()

    fun recordError(message: String?) =
        prefs.edit { putString(KEY_LAST_ERROR, message.orEmpty()) }

    val isConfigured: Boolean get() = serverUrl.isNotBlank() && token.isNotBlank()

    /** Language hint sent to VoiceStudio for a request in [requestLanguage] (ISO 639-2). */
    fun languageHint(requestLanguage: String?): String? = when (languageMode) {
        LANGUAGE_AUTO -> null
        LANGUAGE_FIXED -> fixedLanguage.trim().ifEmpty { null }
        else -> LanguageMapper.displayName(requestLanguage)
    }

    companion object {
        const val LANGUAGE_FROM_READER = "reader"
        const val LANGUAGE_AUTO = "auto"
        const val LANGUAGE_FIXED = "fixed"

        private const val KEY_URL = "server_url"
        private const val KEY_TOKEN = "token"
        private const val KEY_VOICE_ID = "voice_id"
        private const val KEY_VOICE_NAME = "voice_name"
        private const val KEY_LANGUAGE_MODE = "language_mode"
        private const val KEY_FIXED_LANGUAGE = "fixed_language"
        private const val KEY_SPEED = "speed"
        private const val KEY_INSTRUCT = "instruct"
        private const val KEY_LAST_ERROR = "last_error"
    }
}
