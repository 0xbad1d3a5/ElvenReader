package com.androidreader.ui

import android.content.Context
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.androidreader.drama.Character
import com.androidreader.drama.Dramatization
import com.androidreader.library.Utterance
import com.androidreader.net.ServerClient
import com.androidreader.net.ServerVoice
import com.androidreader.player.PlaybackHub
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

object DramaUi {
    private val editing = mutableStateOf<Utterance?>(null)

    fun ensureAnalysis(context: Context, bookId: String) {
        Dramatization.init(context)
        Dramatization.ensureAnalysis(bookId)
    }

    fun editLine(u: Utterance) {
        editing.value = u
    }

    @Composable
    fun AnalysisStatus(bookId: String) {
        val context = LocalContext.current
        val scripts by Dramatization.scripts.collectAsState()
        LaunchedEffect(bookId) {
            withContext(Dispatchers.IO) { runCatching { Dramatization.refresh(bookId) } }
            Dramatization.follow(bookId)
        }
        val s = scripts[bookId]
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            when (s?.status) {
                null, "none" -> {
                    Text("Not analysed yet.", style = MaterialTheme.typography.bodySmall)
                    OutlinedButton(onClick = { ensureAnalysis(context, bookId) }) { Text("Analyse on the PC") }
                }
                "queued", "running" -> {
                    Text("Analysing: ${(s.progress * 100).toInt()}% \u00b7 ${s.detail}", style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth())
                    Text("Lines already analysed are voiced; the rest is read by the narrator.", style = MaterialTheme.typography.bodySmall)
                }
                "casting" -> {
                    Text("Creating character voices on the PC\u2026 ${s.detail}", style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                "error" -> {
                    Text("Analysis failed: ${s.detail}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                    OutlinedButton(onClick = { ensureAnalysis(context, bookId) }) { Text("Retry") }
                }
                else -> Text(
                    "${s.characters.size} characters, ${s.lines.size} lines" + (s.model.takeIf { it.isNotBlank() }?.let { " \u00b7 $it" } ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }

    /** Dialog for correcting who says a quotation and how. */
    @Composable
    fun LineEditorHost(bookId: String) {
        val u = editing.value ?: return
        val scripts by Dramatization.scripts.collectAsState()
        val script = scripts[bookId]
        val qid = u.quoteId ?: return
        val line = script?.lines?.get(qid)
        var speaker by remember(qid) { mutableStateOf(line?.speaker) }
        var delivery by remember(qid) { mutableStateOf(line?.delivery.orEmpty()) }
        var error by remember { mutableStateOf<String?>(null) }
        val scope = rememberCoroutineScope()
        AlertDialog(
            onDismissRequest = { editing.value = null },
            title = { Text("Who says this?") },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("\u201c${u.text.trim('\u201c', '\u201d', '"')}\u201d", style = MaterialTheme.typography.bodyMedium)
                    OutlinedTextField(delivery, { delivery = it }, label = { Text("Delivery (e.g. shouting, urgent)") },
                        singleLine = true, modifier = Modifier.fillMaxWidth())
                    ChoiceRow(speaker == null, "Narrator") { speaker = null }
                    script?.characters?.sortedByDescending { it.lines }?.forEach { c ->
                        ChoiceRow(speaker == c.id, c.name) { speaker = c.id }
                    }
                    error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    scope.launch {
                        withContext(Dispatchers.IO) { runCatching { Dramatization.updateLine(bookId, qid, speaker, delivery.trim()) } }
                            .onSuccess { editing.value = null; PlaybackHub.invalidateUpcoming() }
                            .onFailure { error = it.message }
                    }
                }) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { editing.value = null }) { Text("Cancel") } },
        )
    }
}

/** Characters found in the book: rename, merge aliases, assign voices. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CastScreen(bookId: String, onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scripts by Dramatization.scripts.collectAsState()
    val script = scripts[bookId]
    var voices by remember { mutableStateOf<List<ServerVoice>>(emptyList()) }
    var selected by remember { mutableStateOf<Character?>(null) }
    var busy by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(bookId) {
        Dramatization.init(context)
        withContext(Dispatchers.IO) {
            runCatching { Dramatization.refresh(bookId) }
            runCatching { voices = ServerClient.get(context).getJson("/api/voices?all=1").let { j ->
                val arr = j.getJSONArray("voices")
                (0 until arr.length()).map { i -> arr.getJSONObject(i).let { v -> ServerVoice(v.getString("id"), v.optString("name"), null, v.optString("description"), v.optString("source")) } }
            } }
        }
    }

    fun act(label: String, block: () -> Unit) {
        busy = label
        scope.launch {
            val r = withContext(Dispatchers.IO) { runCatching(block) }
            busy = r.exceptionOrNull()?.let { "Failed: ${it.message}" }
            PlaybackHub.invalidateUpcoming()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Cast") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            )
        },
    ) { padding ->
        LazyColumn(Modifier.padding(padding)) {
            item {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    DramaUi.AnalysisStatus(bookId)
                    Button(onClick = { act("Creating voices on the PC\u2026") { Dramatization.castVoices(bookId, null) } }, enabled = busy == null) {
                        Text("Create voices for characters without one")
                    }
                    busy?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
                }
            }
            // Grouped like the PC viewer: main, supporting, extras (by lines within each).
            val tiers = listOf("main" to "Main", "supporting" to "Supporting", "extra" to "Extras", null to "Not sorted yet")
            val chars = script?.characters ?: emptyList()
            for ((tier, label) in tiers) {
                val members = chars.filter { c -> if (tier == null) c.role !in setOf("main", "supporting", "extra") else c.role == tier }
                    .sortedByDescending { it.lines }
                if (members.isEmpty()) continue
                item(key = "tier-$label") {
                    Text("$label (${members.size})", style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(start = 16.dp, top = 12.dp, bottom = 4.dp))
                }
                items(members, key = { it.id }) { c ->
                ListItem(
                    headlineContent = { Text(c.name) },
                    supportingContent = {
                        Text(listOf("${c.lines} lines", c.gender, c.age, c.description, voices.firstOrNull { it.id == c.voice }?.name?.let { "voice: $it" })
                            .filter { !it.isNullOrBlank() }.joinToString(" \u00b7 "))
                    },
                    modifier = Modifier.clickable { selected = c },
                )
                HorizontalDivider()
                }
            }
        }
    }

    selected?.let { c ->
        var name by remember(c.id) { mutableStateOf(c.name) }
        var voice by remember(c.id) { mutableStateOf(c.voice) }
        var mergeInto by remember(c.id) { mutableStateOf<String?>(null) }
        AlertDialog(
            onDismissRequest = { selected = null },
            title = { Text(c.name) },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    OutlinedTextField(name, { name = it }, label = { Text("Name") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                    if (c.aliases.isNotEmpty()) Text("Also called: ${c.aliases.joinToString()}", style = MaterialTheme.typography.bodySmall)
                    Text("Voice", style = MaterialTheme.typography.titleSmall)
                    voices.forEach { v -> ChoiceRow(voice == v.id, v.name + if (v.source == "character") " (character)" else "") { voice = v.id } }
                    Text("Same person as\u2026", style = MaterialTheme.typography.titleSmall)
                    ChoiceRow(mergeInto == null, "Nobody else") { mergeInto = null }
                    script?.characters?.filter { it.id != c.id }?.forEach { o -> ChoiceRow(mergeInto == o.id, o.name) { mergeInto = o.id } }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    val changes = JSONObject().put("name", name.trim())
                    voice?.let { changes.put("voice", it) }
                    mergeInto?.let { changes.put("merge_into", it) }
                    selected = null
                    act("Saving\u2026") { Dramatization.updateCharacter(bookId, c.id, changes) }
                }) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { selected = null }) { Text("Cancel") } },
        )
    }
}
