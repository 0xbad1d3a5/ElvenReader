package com.androidreader.ui

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
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
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import com.androidreader.R
import com.androidreader.net.EngineInfo
import com.androidreader.net.ServerClient
import com.androidreader.net.ServerHttpException
import com.androidreader.net.ServerSettings
import com.androidreader.net.ServerVoice
import com.androidreader.net.parseServerUrls
import com.androidreader.net.urlKind
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Locale

private fun describeHealth(h: JSONObject, url: String?): String {
    val engine = h.optJSONObject("engine")
    val parts = mutableListOf("Connected via ${url ?: "?"} (${url?.let(::urlKind) ?: ""}). Server ${h.optString("version")}")
    parts += "Default model: ${h.optString("default_engine_label", h.optString("default_engine"))}"
    engine?.let { e ->
        val active = e.optString("active").takeIf { it.isNotBlank() && it != "null" }
        parts += when (e.optString("state")) {
            "loading" -> "Loading ${e.optString("loading")}\u2026"
            "error" -> "Model error: ${e.optString("detail")}"
            else -> if (active != null) "Loaded on the GPU: $active" else "No model loaded yet (loads on first use)"
        }
    }
    parts += "${h.optInt("voices")} voices"
    return parts.joinToString("\n")
}

private fun isDefaultEngine(context: Context): Boolean =
    Settings.Secure.getString(context.contentResolver, "tts_default_synth") == context.packageName

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(pairing: MutableState<Pair<List<String>, String>?>, onBack: () -> Unit) {
    val context = LocalContext.current
    val settings = remember { ServerSettings(context) }
    val client = remember { ServerClient.get(context) }
    val scope = rememberCoroutineScope()

    var urls by remember { mutableStateOf(settings.urls.joinToString("\n")) }
    var token by remember { mutableStateOf(settings.token) }
    var connection by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var engines by remember { mutableStateOf<List<EngineInfo>>(emptyList()) }
    var voices by remember { mutableStateOf<List<ServerVoice>>(emptyList()) }
    var engineId by remember { mutableStateOf(settings.engineId) }
    var readerEngineId by remember { mutableStateOf(settings.readerEngineId) }
    var voiceId by remember { mutableStateOf(settings.voiceId) }
    var readerVoiceId by remember { mutableStateOf(settings.readerVoiceId) }
    var delivery by remember { mutableStateOf(settings.delivery) }
    var autoDelivery by remember { mutableStateOf(settings.autoDelivery) }
    var languageMode by remember { mutableStateOf(settings.languageMode) }
    var fixedLanguage by remember { mutableStateOf(settings.fixedLanguage) }
    var speed by remember { mutableFloatStateOf(settings.speed) }
    var sentencePause by remember { mutableIntStateOf(settings.sentencePauseMs) }
    var sample by remember { mutableStateOf(context.getString(R.string.sample_text)) }
    var testStatus by remember { mutableStateOf<String?>(null) }
    var lastError by remember { mutableStateOf(settings.lastError) }
    var defaultEngine by remember { mutableStateOf(isDefaultEngine(context)) }

    fun connect() {
        settings.urls = parseServerUrls(urls).also { urls = it.joinToString("\n") }
        settings.token = token
        if (!settings.isConfigured) {
            connection = "Enter the server URL(s) and the token shown by the server on your PC."
            return
        }
        busy = true
        connection = "Connecting\u2026"
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    client.probe()
                    val health = client.health()
                    Triple(health, client.engines().first, client.voices())
                }
            }
            result.onSuccess { (health, e, v) ->
                connection = describeHealth(health, client.activeUrl)
                engines = e
                voices = v.first
                if (voices.none { it.id == voiceId }) {
                    v.second?.let { def -> voiceId = def; settings.voiceId = def }
                }
            }.onFailure { e ->
                connection = when ((e as? ServerHttpException)?.code) {
                    401 -> "The server rejected the token. Pair again or check the token."
                    else -> "Could not reach the PC: ${e.message ?: e.javaClass.simpleName}\n" +
                        "Check the PC is on, the server is running, the firewall allows port 3950, " +
                        "and (away from home) that Tailscale is connected."
                }
            }
            busy = false
        }
    }

    LaunchedEffect(pairing.value) {
        val pair = pairing.value ?: return@LaunchedEffect
        urls = pair.first.joinToString("\n")
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
            override fun onStart(utteranceId: String?) = main.execute { testStatus = "Speaking\u2026" }
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
        tts.speak(sample, TextToSpeech.QUEUE_FLUSH, null, "androidreader-test")
    }

    fun testVoice() {
        testStatus = "Synthesizing on the PC\u2026"
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

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .imePadding()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Section("Connection to your PC") {
                Text(
                    "Scan the QR code on http://127.0.0.1:3950/ on the PC, or enter the addresses below, one per " +
                        "line: the home network address first, then the Tailscale one. The app uses whichever answers.",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedTextField(
                    value = urls,
                    onValueChange = { urls = it },
                    label = { Text("Server URLs (e.g. http://192.168.1.20:3950)") },
                    minLines = 2,
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

            if (engines.isNotEmpty()) {
                Section("Voice models on the PC") {
                    engines.forEach { e ->
                        Text(
                            e.label + when {
                                !e.installed -> " \u2014 not installed"
                                e.active -> " \u2014 loaded"
                                else -> ""
                            },
                            style = MaterialTheme.typography.titleSmall,
                        )
                        Text(
                            e.description + (e.rtf?.let { String.format(Locale.ROOT, "\nMeasured: %.2f\u00d7 real time", 1 / it) } ?: "") +
                                "\nDelivery control: ${e.delivery.ifBlank { "none" }} \u00b7 Licence: ${e.licence}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
                Section("Reader defaults") {
                    Text("Model for books (each book can override it)", style = MaterialTheme.typography.titleSmall)
                    EngineChoices(engines, readerEngineId) { readerEngineId = it; settings.readerEngineId = it }
                    Text("Voice for books", style = MaterialTheme.typography.titleSmall)
                    VoiceChoices(voices, readerVoiceId) { readerVoiceId = it; settings.readerVoiceId = it }
                }
            }

            Section("System voice (ReadEra and other apps)") {
                Text(
                    if (defaultEngine) "AndroidReader is the preferred text-to-speech engine."
                    else "Set AndroidReader as the preferred engine so ReadEra can use it.",
                )
                OutlinedButton(onClick = {
                    try {
                        context.startActivity(Intent("com.android.settings.TTS_SETTINGS"))
                    } catch (_: ActivityNotFoundException) {
                        context.startActivity(Intent(Settings.ACTION_SETTINGS))
                    }
                }) { Text("Open text-to-speech settings") }
                if (engines.isNotEmpty()) {
                    Text("Model", style = MaterialTheme.typography.titleSmall)
                    EngineChoices(engines, engineId) { engineId = it; settings.engineId = it }
                }
                Text("Voice", style = MaterialTheme.typography.titleSmall)
                if (voices.isEmpty()) Text("Connect to load the voices saved on the PC.")
                VoiceChoices(voices, voiceId) { id ->
                    voiceId = id
                    settings.voiceId = id
                    settings.voiceName = voices.firstOrNull { it.id == id }?.name.orEmpty()
                }
                OutlinedTextField(
                    value = delivery,
                    onValueChange = { delivery = it; settings.delivery = it },
                    label = { Text("Delivery for every sentence (optional, e.g. \"calm, warm\")") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Expressive quotes")
                        Text(
                            "Reading apps send one sentence at a time with no book context, so only simple cues are " +
                                "used: an exclamation inside quotes, or \u201cwhispered\u201d/\u201cshouted\u201d next to a quote. " +
                                "Models with delivery control (Fish S2 Pro, IndexTTS) act on it.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    Switch(autoDelivery, { autoDelivery = it; settings.autoDelivery = it })
                }
                Text("Language", style = MaterialTheme.typography.titleSmall)
                ChoiceRow(languageMode == ServerSettings.LANGUAGE_FROM_READER, "Follow the reading app's language") {
                    languageMode = ServerSettings.LANGUAGE_FROM_READER; settings.languageMode = languageMode
                }
                ChoiceRow(languageMode == ServerSettings.LANGUAGE_AUTO, "Let the model detect it") {
                    languageMode = ServerSettings.LANGUAGE_AUTO; settings.languageMode = languageMode
                }
                ChoiceRow(languageMode == ServerSettings.LANGUAGE_FIXED, "Always use:") {
                    languageMode = ServerSettings.LANGUAGE_FIXED; settings.languageMode = languageMode
                }
                if (languageMode == ServerSettings.LANGUAGE_FIXED) {
                    OutlinedTextField(
                        value = fixedLanguage,
                        onValueChange = { fixedLanguage = it; settings.fixedLanguage = it },
                        label = { Text("Language name, e.g. English") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Text(String.format(Locale.ROOT, "Speed %.2f\u00d7 (multiplied by the reading app's speed)", speed))
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
                OutlinedTextField(value = sample, onValueChange = { sample = it }, modifier = Modifier.fillMaxWidth())
                Button(onClick = { testVoice() }) { Text("Test system voice") }
                testStatus?.let { Text(it) }
                if (lastError.isNotBlank()) Text("Last error: $lastError", color = MaterialTheme.colorScheme.error)
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun EngineChoices(engines: List<EngineInfo>, selected: String, onSelect: (String) -> Unit) {
    ChoiceRow(selected.isBlank(), "Server default") { onSelect("") }
    engines.filter { it.installed }.forEach { e -> ChoiceRow(selected == e.id, e.label) { onSelect(e.id) } }
}

@Composable
private fun VoiceChoices(voices: List<ServerVoice>, selected: String, onSelect: (String) -> Unit) {
    if (voices.isNotEmpty()) ChoiceRow(selected.isBlank(), "Server default") { onSelect("") }
    voices.forEach { v ->
        ChoiceRow(selected == v.id, v.name + (v.language?.let { " ($it)" } ?: "")) { onSelect(v.id) }
    }
}
