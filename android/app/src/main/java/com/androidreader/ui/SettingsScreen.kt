package com.androidreader.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.media3.common.util.UnstableApi
import com.androidreader.net.EngineInfo
import com.androidreader.net.ServerClient
import com.androidreader.net.ServerHttpException
import com.androidreader.net.ServerSettings
import com.androidreader.net.ServerVoice
import com.androidreader.net.parseServerUrls
import com.androidreader.net.urlKind
import com.androidreader.player.PlaybackHub
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Locale

private fun describeHealth(h: JSONObject, url: String?): String {
    val engine = h.optJSONObject("engine")
    val via = url?.let { "$it (${if (urlKind(it) == "lan") "local network" else "remote"})" } ?: "?"
    val parts = mutableListOf("Connected via $via. Server ${h.optString("version")}")
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

@androidx.annotation.OptIn(UnstableApi::class)
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
    var readerEngineId by remember { mutableStateOf(settings.readerEngineId) }
    var readerVoiceId by remember { mutableStateOf(settings.readerVoiceId) }
    var sentencePause by remember { mutableIntStateOf(settings.sentencePauseMs) }

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
                    Triple(client.health(), client.engines().first, client.voices().first)
                }
            }
            result.onSuccess { (health, e, v) ->
                connection = describeHealth(health, client.activeUrl)
                engines = e
                voices = v
            }.onFailure { e ->
                connection = when ((e as? ServerHttpException)?.code) {
                    401 -> "The server rejected the token. Pair again or check the token."
                    else -> "Could not reach the PC: ${e.message ?: e.javaClass.simpleName}\n" +
                        "Check the PC is on, the server is running, the firewall allows port 3950, " +
                        "and (away from home) that your VPN is connected."
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
                    "Scan the QR code on http://127.0.0.1:3950/ on the PC, or enter addresses below, one per " +
                        "line: the home network address first, then the PC's address on your VPN for when " +
                        "you're away. The app uses whichever answers.",
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
            }

            Section("Reading defaults") {
                Text("Each book can override the model and voice in its reading options.", style = MaterialTheme.typography.bodySmall)
                if (engines.isNotEmpty()) {
                    Text("Model", style = MaterialTheme.typography.titleSmall)
                    ChoiceRow(readerEngineId.isBlank(), "Server default") { readerEngineId = ""; settings.readerEngineId = "" }
                    engines.filter { it.installed }.forEach { e ->
                        ChoiceRow(readerEngineId == e.id, e.label) { readerEngineId = e.id; settings.readerEngineId = e.id }
                    }
                }
                Text("Voice", style = MaterialTheme.typography.titleSmall)
                if (voices.isEmpty()) Text("Connect to load the voices saved on the PC.")
                if (voices.isNotEmpty()) ChoiceRow(readerVoiceId.isBlank(), "Server default") { readerVoiceId = ""; settings.readerVoiceId = "" }
                voices.forEach { v ->
                    ChoiceRow(readerVoiceId == v.id, v.name + (v.language?.let { " ($it)" } ?: "")) {
                        readerVoiceId = v.id; settings.readerVoiceId = v.id
                    }
                }
                Text("Pause after sentences: $sentencePause ms (paragraph ends get more)")
                Slider(
                    value = sentencePause.toFloat(),
                    onValueChange = { sentencePause = (it / 50).toInt() * 50 },
                    onValueChangeFinished = {
                        settings.sentencePauseMs = sentencePause
                        PlaybackHub.invalidateUpcoming()
                    },
                    valueRange = 0f..1000f,
                )
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}
