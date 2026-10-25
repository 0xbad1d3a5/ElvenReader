package com.androidreader.ui

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
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.media3.common.util.UnstableApi
import com.androidreader.library.Library
import com.androidreader.net.PronEntry
import com.androidreader.net.ServerClient
import com.androidreader.net.TtsRequest
import com.androidreader.player.PlaybackHub
import com.androidreader.player.SpeechAudio
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

/**
 * Edit or create a pronunciation. [sentence] is the text the word was picked from;
 * Preview synthesizes it with the unsaved entry and the book's current voice.
 */
@androidx.annotation.OptIn(UnstableApi::class)
@Composable
fun PronunciationDialog(
    initial: PronEntry,
    bookId: String?,
    sentence: String?,
    onDismiss: () -> Unit,
    onSaved: (PronEntry) -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val client = remember { ServerClient.get(context) }
    var term by remember { mutableStateOf(initial.term) }
    var respelling by remember { mutableStateOf(initial.respelling) }
    var phonemes by remember { mutableStateOf(initial.phonemes) }
    var caseSensitive by remember { mutableStateOf(initial.caseSensitive) }
    var forBook by remember { mutableStateOf(initial.book != null || (initial.id.isEmpty() && bookId != null)) }
    var status by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    val sample = sentence ?: term

    fun entry() = initial.copy(
        term = term.trim(), respelling = respelling.trim(), phonemes = phonemes.trim(), caseSensitive = caseSensitive,
        book = if (forBook && bookId != null) bookId else null,
    )

    fun preview() {
        busy = true
        status = "Synthesizing on the PC\u2026"
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val meta = bookId?.let { Library.get(context).get(it) }
                    val base = PlaybackHub.state.value.let { s ->
                        if (s.bookId == bookId) PlaybackHub.requestFor(s.chapter, s.index) else null
                    }
                    val req = TtsRequest(
                        text = sample,
                        engine = base?.engine ?: meta?.engine?.ifBlank { null },
                        voice = base?.voice ?: meta?.voice?.ifBlank { null },
                        language = base?.language,
                        book = bookId,
                        extraPronunciations = listOf(entry()),
                        cache = false,
                        sentencePauseMs = 0,
                    )
                    client.tts(req).use { resp ->
                        val rate = resp.header("X-Sample-Rate")?.toIntOrNull() ?: 24000
                        val out = ByteArrayOutputStream()
                        resp.body?.byteStream()?.copyTo(out)
                        SpeechAudio(rate, out.toByteArray())
                    }
                }
            }
            result.onSuccess { PlaybackHub.pause(); ClipPlayer.play(it); status = null }
                .onFailure { status = "Preview failed: ${it.message}" }
            busy = false
        }
    }

    fun save() {
        busy = true
        scope.launch {
            val result = withContext(Dispatchers.IO) { runCatching { client.savePronunciation(entry()) } }
            busy = false
            result.onSuccess {
                PlaybackHub.invalidateUpcoming()
                onSaved(it)
            }.onFailure { status = "Could not save: ${it.message}" }
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (initial.id.isEmpty()) "Fix pronunciation" else "Edit pronunciation") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(term, { term = it }, label = { Text("Word or name") }, singleLine = true,
                    modifier = Modifier.fillMaxWidth())
                OutlinedTextField(respelling, { respelling = it }, label = { Text("Say it like (e.g. her-MY-oh-nee)") },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(phonemes, { phonemes = it }, label = { Text("Phonemes, optional (CMU: HH ER0 M AY1 AH0 N IY0)") },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                Text(
                    "The respelling works with every voice model. Phonemes are used by OmniVoice and IndexTTS " +
                        "when given; plurals and possessives (\u201c${term.ifBlank { "Name" }}\u2019s\u201d) use the respelling.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(caseSensitive, { caseSensitive = it })
                    Text("Match case exactly")
                }
                if (bookId != null) {
                    ChoiceRow(forBook, "Only in this book") { forBook = true }
                    ChoiceRow(!forBook, "In all books") { forBook = false }
                }
                OutlinedButton(onClick = { preview() }, enabled = !busy && term.isNotBlank()) { Text("Preview") }
                status?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }
        },
        confirmButton = {
            TextButton(onClick = { save() }, enabled = !busy && term.isNotBlank() && (respelling.isNotBlank() || phonemes.isNotBlank())) {
                Text("Save")
            }
        },
        dismissButton = { TextButton(onClick = { ClipPlayer.stop(); onDismiss() }) { Text("Cancel") } },
    )
}

/** Lists global entries and, when opened from a book, that book's entries. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PronunciationsScreen(bookId: String?, onBack: () -> Unit) {
    val context = LocalContext.current
    val client = remember { ServerClient.get(context) }
    val scope = rememberCoroutineScope()
    var entries by remember { mutableStateOf<List<PronEntry>>(emptyList()) }
    var error by remember { mutableStateOf<String?>(null) }
    var editing by remember { mutableStateOf<PronEntry?>(null) }
    val bookTitle = bookId?.let { Library.get(context).get(it)?.title }

    fun reload() {
        scope.launch {
            withContext(Dispatchers.IO) { runCatching { client.pronunciations(bookId) } }
                .onSuccess { entries = it; error = null }
                .onFailure { error = "Could not load from the PC: ${it.message}" }
        }
    }
    LaunchedEffect(bookId) { reload() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Pronunciations") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { editing = PronEntry("", "", "", "", false, bookId) }) {
                Icon(Icons.Default.Add, "Add")
            }
        },
    ) { padding ->
        LazyColumn(Modifier.padding(padding)) {
            error?.let { item { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(16.dp)) } }
            item {
                Text(
                    "Corrections are applied on the PC before synthesis, with every voice model. " +
                        "Long-press a word while reading to add one.",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(16.dp),
                )
            }
            val groups = listOfNotNull(
                bookId?.let { "This book" + (bookTitle?.let { t -> " \u2014 $t" } ?: "") to entries.filter { e -> e.book == bookId } },
                "All books" to entries.filter { it.isGlobal },
            )
            for ((title, list) in groups) {
                item { Text(title, style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(16.dp, 12.dp, 16.dp, 4.dp)) }
                if (list.isEmpty()) item { Text("None yet.", modifier = Modifier.padding(horizontal = 16.dp)) }
                items(list, key = { it.id }) { e ->
                    ListItem(
                        headlineContent = { Text(e.term) },
                        supportingContent = {
                            Text(listOf(e.respelling, e.phonemes.takeIf { it.isNotBlank() }?.let { "[$it]" }, if (e.caseSensitive) "match case" else null)
                                .filterNotNull().filter { it.isNotBlank() }.joinToString("  \u00b7  "))
                        },
                        trailingContent = {
                            IconButton(onClick = {
                                scope.launch {
                                    withContext(Dispatchers.IO) { runCatching { client.deletePronunciation(e.id) } }
                                    PlaybackHub.invalidateUpcoming()
                                    reload()
                                }
                            }) { Icon(Icons.Default.Delete, "Delete") }
                        },
                        modifier = Modifier.clickable { editing = e },
                    )
                    HorizontalDivider()
                }
            }
        }
    }

    editing?.let { e ->
        PronunciationDialog(e, bookId, null, onDismiss = { editing = null }, onSaved = { editing = null; reload() })
    }
}
