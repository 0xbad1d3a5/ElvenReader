package com.androidvoicestudio.tts

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Locale

/** Engine settings, opened from the launcher, the system TTS settings gear, or a pairing link. */
class SettingsActivity : ComponentActivity() {
    private val pairing = mutableStateOf<Pair<String, String>?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        handlePairingIntent(intent)
        setContent { AppTheme { SettingsScreen(TtsSettings(this), pairing) } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handlePairingIntent(intent)
    }

    private fun handlePairingIntent(intent: Intent?) {
        val data = intent?.data ?: return
        if (data.scheme != "vstts" || data.host != "pair") return
        val url = data.getQueryParameter("url")
        val token = data.getQueryParameter("token")
        if (!url.isNullOrBlank() && !token.isNullOrBlank()) pairing.value = url to token
    }
}

@Composable
private fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val colors = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> darkColorScheme()
        else -> lightColorScheme()
    }
    MaterialTheme(colorScheme = colors) {
        Surface(modifier = Modifier.fillMaxSize(), content = content)
    }
}

private fun describeHealth(h: JSONObject): String {
    val vs = h.optJSONObject("voicestudio") ?: return "Relay OK"
    if (!vs.optBoolean("reachable")) {
        return "Relay OK, but VoiceStudio is not reachable on the PC.\n${vs.optString("detail")}"
    }
    val parts = mutableListOf("Connected. VoiceStudio ${vs.optString("version")}")
    vs.optString("engine").takeIf { it.isNotBlank() }?.let { parts += "engine: $it" }
    vs.optString("model_status").takeIf { it.isNotBlank() }?.let { status ->
        parts += "model: $status"
        if (status.lowercase(Locale.ROOT) == "loading") {
            parts += vs.optString("model_detail").ifBlank { "still loading; synthesis will fail until it finishes" }
        }
    }
    return parts.joinToString("\n")
}

private fun isDefaultEngine(context: Context): Boolean =
    Settings.Secure.getString(context.contentResolver, "tts_default_synth") == context.packageName

@Composable
private fun SettingsScreen(settings: TtsSettings, pairing: MutableState<Pair<String, String>?>) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var serverUrl by remember { mutableStateOf(settings.serverUrl) }
    var token by remember { mutableStateOf(settings.token) }
    var connection by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var voices by remember { mutableStateOf<List<RelayVoice>>(emptyList()) }
    var voiceId by remember { mutableStateOf(settings.voiceId) }
    var voiceName by remember { mutableStateOf(settings.voiceName) }
    var languageMode by remember { mutableStateOf(settings.languageMode) }
    var fixedLanguage by remember { mutableStateOf(settings.fixedLanguage) }
    var speed by remember { mutableFloatStateOf(settings.speed) }
    var sentencePause by remember { mutableIntStateOf(settings.sentencePauseMs) }
    var instruct by remember { mutableStateOf(settings.instruct) }
    var sample by remember { mutableStateOf(context.getString(R.string.sample_text)) }
    var testStatus by remember { mutableStateOf<String?>(null) }
    var lastError by remember { mutableStateOf(settings.lastError) }
    var defaultEngine by remember { mutableStateOf(isDefaultEngine(context)) }

    fun connect() {
        settings.serverUrl = normalizeServerUrl(serverUrl).also { serverUrl = it }
        settings.token = token
        if (!settings.isConfigured) {
            connection = "Enter the server URL and token shown by the relay on your PC."
            return
        }
        busy = true
        connection = "Connecting…"
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val client = RelayClient.create(context, settings.serverUrl, settings.token)
                    val health = client.health()
                    val list = runCatching { client.voices() }.getOrDefault(emptyList())
                    health to list
                }
            }
            result.onSuccess { (health, list) ->
                connection = describeHealth(health)
                voices = list
                // Never leave the engine without a reference voice: OmniVoice would
                // then invent a different speaker for every sentence.
                if (list.none { it.id == voiceId }) {
                    list.firstOrNull { it.stable }?.let { v ->
                        voiceId = v.id
                        voiceName = v.name
                        settings.voiceId = v.id
                        settings.voiceName = v.name
                    }
                }
            }.onFailure { e ->
                connection = when ((e as? RelayHttpException)?.code) {
                    401 -> "The relay rejected the token. Re-pair or check the token."
                    else -> "Could not reach the relay: ${e.message ?: e.javaClass.simpleName}\n" +
                        "Check the PC is on, the relay is running, and the firewall allows port 3950."
                }
            }
            busy = false
        }
    }

    LaunchedEffect(pairing.value) {
        val pair = pairing.value ?: return@LaunchedEffect
        serverUrl = pair.first
        token = pair.second
        pairing.value = null
        connect()
    }
    LaunchedEffect(Unit) { if (settings.isConfigured) connect() }
    LifecycleResumeEffect(Unit) {
        defaultEngine = isDefaultEngine(context)
        lastError = settings.lastError
        onPauseOrDispose { }
    }

    val ttsHolder = remember { arrayOfNulls<TextToSpeech>(1) }
    DisposableEffect(Unit) { onDispose { ttsHolder[0]?.shutdown() } }

    fun speak(tts: TextToSpeech) {
        val main = ContextCompat.getMainExecutor(context)
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = main.execute { testStatus = "Speaking…" }
            override fun onDone(utteranceId: String?) = main.execute { testStatus = "Done." }
            override fun onStop(utteranceId: String?, interrupted: Boolean) = main.execute { testStatus = "Stopped." }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) = onError(utteranceId, TextToSpeech.ERROR)

            override fun onError(utteranceId: String?, errorCode: Int) = main.execute {
                testStatus = "Failed (code $errorCode): ${settings.lastError.ifBlank { "unknown error" }}"
                lastError = settings.lastError
            }
        })
        tts.language = Locale.getDefault()
        tts.speak(sample, TextToSpeech.QUEUE_FLUSH, null, "voicestudio-test")
    }

    fun testVoice() {
        testStatus = "Synthesizing on the PC…"
        ttsHolder[0]?.let { speak(it); return }
        var created: TextToSpeech? = null
        created = TextToSpeech(context, { status ->
            val engine = created
            if (status == TextToSpeech.SUCCESS && engine != null) {
                ttsHolder[0] = engine
                speak(engine)
            } else {
                testStatus = "Could not start the engine (status $status)"
            }
        }, context.packageName)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("VoiceStudio TTS", style = MaterialTheme.typography.headlineSmall)

        Section("Connection to your PC") {
            OutlinedTextField(
                value = serverUrl,
                onValueChange = { serverUrl = it },
                label = { Text("Server URL (e.g. http://192.168.1.20:3950)") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedTextField(
                value = token,
                onValueChange = { token = it },
                label = { Text("Token") },
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                modifier = Modifier.fillMaxWidth(),
            )
            Button(onClick = { connect() }, enabled = !busy) { Text("Save & test connection") }
            connection?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
        }

        Section("Voice") {
            val options = voices.ifEmpty {
                listOf(RelayVoice(voiceId, voiceName.ifBlank { "Relay default voice" }, null))
            }
            options.forEach { v ->
                ChoiceRow(
                    selected = v.id == voiceId,
                    label = v.name + (v.language?.let { " ($it)" } ?: ""),
                ) {
                    voiceId = v.id
                    voiceName = v.name
                    settings.voiceId = v.id
                    settings.voiceName = v.name
                }
            }
            if (voices.isEmpty()) Text("Connect to load the voice profiles saved in VoiceStudio.")
            if (voices.isNotEmpty() && voices.none { it.stable }) {
                Text(
                    "No saved voice has reference audio, so the speaker will change every sentence. " +
                        "Clone a voice (or lock a designed one) in VoiceStudio, then reconnect.",
                    color = MaterialTheme.colorScheme.error,
                )
            }
            OutlinedTextField(
                value = instruct,
                onValueChange = { instruct = it; settings.instruct = it },
                label = { Text("Style instruction (optional, e.g. \"calm, low pitch\")") },
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Section("Language") {
            ChoiceRow(languageMode == TtsSettings.LANGUAGE_FROM_READER, "Follow the reading app's language") {
                languageMode = TtsSettings.LANGUAGE_FROM_READER; settings.languageMode = languageMode
            }
            ChoiceRow(languageMode == TtsSettings.LANGUAGE_AUTO, "Let VoiceStudio detect it") {
                languageMode = TtsSettings.LANGUAGE_AUTO; settings.languageMode = languageMode
            }
            ChoiceRow(languageMode == TtsSettings.LANGUAGE_FIXED, "Always use:") {
                languageMode = TtsSettings.LANGUAGE_FIXED; settings.languageMode = languageMode
            }
            if (languageMode == TtsSettings.LANGUAGE_FIXED) {
                OutlinedTextField(
                    value = fixedLanguage,
                    onValueChange = { fixedLanguage = it; settings.fixedLanguage = it },
                    label = { Text("Language name, e.g. English") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        Section("Speed and pacing") {
            Text(String.format(Locale.ROOT, "%.2f× (multiplied by the reading app's speed)", speed))
            Slider(
                value = speed,
                onValueChange = { speed = (it * 20).toInt() / 20f },
                onValueChangeFinished = { settings.speed = speed },
                valueRange = 0.5f..2.0f,
            )
            Text("Pause after sentences: $sentencePause ms (a third of it after commas)")
            Slider(
                value = sentencePause.toFloat(),
                onValueChange = { sentencePause = (it / 50).toInt() * 50 },
                onValueChangeFinished = { settings.sentencePauseMs = sentencePause },
                valueRange = 0f..1000f,
            )
        }

        Section("Test") {
            OutlinedTextField(
                value = sample,
                onValueChange = { sample = it },
                modifier = Modifier.fillMaxWidth(),
            )
            Button(onClick = { testVoice() }) { Text("Test voice") }
            testStatus?.let { Text(it) }
            if (lastError.isNotBlank()) {
                Text("Last error: $lastError", color = MaterialTheme.colorScheme.error)
            }
        }

        Section("System") {
            Text(
                if (defaultEngine) "VoiceStudio TTS is the preferred engine."
                else "Set VoiceStudio TTS as the preferred engine so ReadEra can use it.",
            )
            OutlinedButton(onClick = {
                try {
                    context.startActivity(Intent("com.android.settings.TTS_SETTINGS"))
                } catch (_: ActivityNotFoundException) {
                    context.startActivity(Intent(Settings.ACTION_SETTINGS))
                }
            }) { Text("Open text-to-speech settings") }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun ChoiceRow(selected: Boolean, label: String, onClick: () -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .selectable(selected = selected, onClick = onClick, role = Role.RadioButton),
    ) {
        RadioButton(selected = selected, onClick = null)
        Text(label, modifier = Modifier.padding(start = 8.dp))
    }
}
