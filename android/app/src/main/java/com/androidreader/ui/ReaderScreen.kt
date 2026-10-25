package com.androidreader.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Groups
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SkipNext
import androidx.compose.material.icons.filled.SkipPrevious
import androidx.compose.material.icons.filled.Spellcheck
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.FastForward
import androidx.compose.material.icons.filled.FastRewind
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.androidreader.library.BookContent
import com.androidreader.library.Library
import com.androidreader.library.Utterance
import com.androidreader.net.EngineInfo
import com.androidreader.net.PronEntry
import com.androidreader.net.ServerClient
import com.androidreader.net.ServerSettings
import com.androidreader.net.ServerVoice
import com.androidreader.player.PlaybackHub
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/** What a long-press picked: a word inside an utterance. */
private data class Pick(val word: String, val utterance: Utterance)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReaderScreen(bookId: String, onBack: () -> Unit, onPronunciations: () -> Unit, onCast: () -> Unit) {
    val context = LocalContext.current
    val library = remember { Library.get(context) }
    val settings = remember { ServerSettings(context) }
    val hub by PlaybackHub.state.collectAsState()
    val books by library.books.collectAsState()
    val meta = books.firstOrNull { it.id == bookId }
    var content by remember { mutableStateOf<BookContent?>(null) }
    var fontSp by remember { mutableFloatStateOf(settings.readerFontSp) }
    var theme by remember { mutableStateOf(settings.readerTheme) }
    var showToc by remember { mutableStateOf(false) }
    var showOptions by remember { mutableStateOf(false) }
    var pick by remember { mutableStateOf<Pick?>(null) }
    var fixing by remember { mutableStateOf<Pick?>(null) }
    var browseChapter by remember { mutableStateOf<Int?>(null) }

    LaunchedEffect(bookId) {
        content = withContext(Dispatchers.IO) { library.content(bookId) }
        if (PlaybackHub.state.value.bookId != bookId) PlaybackHub.open(bookId)
    }
    val book = content ?: run {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
        return
    }
    val active = hub.bookId == bookId
    val chapter = (browseChapter ?: if (active) hub.chapter else meta?.position?.chapter ?: 0).coerceIn(0, book.chapters.size - 1)
    val dramatized = if (active) hub.dramatized else meta?.dramatized == true
    val utterances = remember(book, chapter, dramatized) { book.utterances(chapter, dramatized) }
    val current = if (active && hub.chapter == chapter) utterances.getOrNull(hub.index) else null
    val colors = pageColors(theme)
    val listState = rememberLazyListState()

    // Follow the voice: keep the paragraph being read on screen.
    LaunchedEffect(current?.paragraph, chapter) {
        val p = current?.paragraph ?: return@LaunchedEffect
        val visible = listState.layoutInfo.visibleItemsInfo
        val target = p + 1 // item 0 is the chapter title
        if (visible.none { it.index == target } || visible.lastOrNull()?.index == target) {
            listState.animateScrollToItem(target, -200)
        }
    }
    LaunchedEffect(browseChapter) { if (browseChapter != null) listState.scrollToItem(0) }

    Scaffold(
        containerColor = colors.background,
        topBar = {
            TopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(containerColor = colors.background, titleContentColor = colors.text),
                title = {
                    Column {
                        Text(book.chapters[chapter].title, maxLines = 1, style = MaterialTheme.typography.titleMedium)
                        Text(book.meta.title, maxLines = 1, style = MaterialTheme.typography.bodySmall, color = colors.muted)
                    }
                },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = colors.text) } },
                actions = {
                    IconButton(onClick = { showToc = true }) { Icon(Icons.AutoMirrored.Filled.List, "Contents", tint = colors.text) }
                    IconButton(onClick = onPronunciations) { Icon(Icons.Default.Spellcheck, "Pronunciations", tint = colors.text) }
                    if (dramatized) IconButton(onClick = onCast) { Icon(Icons.Default.Groups, "Cast", tint = colors.text) }
                    IconButton(onClick = { showOptions = true }) { Icon(Icons.Default.Tune, "Reading options", tint = colors.text) }
                },
            )
        },
        bottomBar = { PlayerBar(hub, active, colors) },
    ) { padding ->
        LazyColumn(
            state = listState,
            modifier = Modifier.padding(padding).fillMaxSize().background(colors.background),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy((fontSp * 0.6f).dp),
        ) {
            item {
                Text(book.chapters[chapter].title, color = colors.text, fontSize = (fontSp * 1.3f).sp, fontWeight = FontWeight.Bold)
            }
            itemsIndexed(book.chapters[chapter].paragraphs) { p, para ->
                val inPara = remember(utterances, p) { utterances.filter { it.paragraph == p } }
                ParagraphText(
                    text = para.text,
                    heading = para.heading,
                    highlight = current?.takeIf { it.paragraph == p }?.let { it.start until it.end },
                    colors = colors,
                    fontSp = fontSp,
                    onTap = { offset ->
                        inPara.firstOrNull { offset < it.end }?.let { u ->
                            browseChapter = null
                            if (!active) PlaybackHub.open(bookId)
                            PlaybackHub.playFrom(chapter, u.index)
                        }
                    },
                    onLongPress = { offset ->
                        val u = inPara.firstOrNull { offset < it.end } ?: return@ParagraphText
                        wordAt(para.text, offset)?.let { pick = Pick(it, u) }
                    },
                )
            }
            item {
                if (chapter + 1 < book.chapters.size) {
                    TextButton(onClick = { browseChapter = chapter + 1 }) { Text("Next chapter: ${book.chapters[chapter + 1].title}") }
                }
                Spacer(Modifier.size(80.dp))
            }
        }
    }

    if (showToc) {
        ModalBottomSheet(onDismissRequest = { showToc = false }) {
            LazyColumn {
                itemsIndexed(book.chapters) { i, c ->
                    ListItem(
                        headlineContent = { Text(c.title, fontWeight = if (i == chapter) FontWeight.Bold else null) },
                        modifier = Modifier.clickable {
                            showToc = false
                            browseChapter = null
                            if (active && hub.playing) PlaybackHub.playFrom(i, 0)
                            else PlaybackHub.open(bookId, com.androidreader.library.BookPosition(i, 0, 0), play = false)
                        },
                    )
                }
            }
        }
    }

    if (showOptions) {
        ModalBottomSheet(onDismissRequest = { showOptions = false }) {
            ReadingOptions(
                bookId = bookId,
                fontSp = fontSp,
                onFont = { fontSp = it; settings.readerFontSp = it },
                theme = theme,
                onTheme = { theme = it; settings.readerTheme = it },
            )
        }
    }

    pick?.let { p ->
        AlertDialog(
            onDismissRequest = { pick = null },
            title = { Text("\u201c${p.word}\u201d") },
            text = {
                Column {
                    TextButton(onClick = { fixing = p; pick = null }) { Text("Fix pronunciation") }
                    TextButton(onClick = { pick = null; PlaybackHub.playFrom(chapter, p.utterance.index) }) { Text("Read from here") }
                    if (dramatized && p.utterance.quote != null) {
                        TextButton(onClick = { pick = null; DramaUi.editLine(p.utterance) }) { Text("Change speaker or delivery") }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { pick = null }) { Text("Close") } },
        )
    }
    fixing?.let { p ->
        PronunciationDialog(
            initial = PronEntry("", p.word, "", "", false, bookId),
            bookId = bookId,
            sentence = p.utterance.text,
            onDismiss = { fixing = null },
            onSaved = { fixing = null },
        )
    }
    DramaUi.LineEditorHost(bookId)
}

@Composable
private fun ParagraphText(
    text: String,
    heading: Boolean,
    highlight: IntRange?,
    colors: PageColors,
    fontSp: Float,
    onTap: (Int) -> Unit,
    onLongPress: (Int) -> Unit,
) {
    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }
    val tap by androidx.compose.runtime.rememberUpdatedState(onTap)
    val longPress by androidx.compose.runtime.rememberUpdatedState(onLongPress)
    val annotated = remember(text, highlight, colors) {
        buildAnnotatedString {
            if (highlight == null || highlight.first >= text.length) {
                append(text)
            } else {
                append(text.substring(0, highlight.first))
                withStyle(SpanStyle(background = colors.highlight)) {
                    append(text.substring(highlight.first, (highlight.last + 1).coerceAtMost(text.length)))
                }
                append(text.substring((highlight.last + 1).coerceAtMost(text.length)))
            }
        }
    }
    Text(
        annotated,
        color = colors.text,
        fontSize = (if (heading) fontSp * 1.15f else fontSp).sp,
        lineHeight = (fontSp * 1.55f).sp,
        fontWeight = if (heading) FontWeight.SemiBold else null,
        onTextLayout = { layout = it },
        modifier = Modifier.fillMaxWidth().pointerInput(text) {
            detectTapGestures(
                onTap = { pos -> layout?.let { tap(it.getOffsetForPosition(pos)) } },
                onLongPress = { pos -> layout?.let { longPress(it.getOffsetForPosition(pos)) } },
            )
        },
    )
}

/** The word (letters, digits, apostrophes, inner hyphens) around [offset]. */
fun wordAt(text: String, offset: Int): String? {
    if (text.isEmpty()) return null
    fun isWord(c: Char) = c.isLetterOrDigit() || c == '\'' || c == '\u2019' || c == '-'
    var i = offset.coerceIn(0, text.length - 1)
    if (!isWord(text[i]) && i > 0 && isWord(text[i - 1])) i--
    if (!isWord(text[i])) return null
    var s = i
    var e = i + 1
    while (s > 0 && isWord(text[s - 1])) s--
    while (e < text.length && isWord(text[e])) e++
    return text.substring(s, e).trim('\'', '\u2019', '-').removeSuffix("\u2019s").removeSuffix("'s").ifBlank { null }
}

@Composable
private fun PlayerBar(hub: PlaybackHub.State, active: Boolean, colors: PageColors) {
    BottomAppBar(containerColor = colors.background, modifier = Modifier.navigationBarsPadding()) {
        Column(Modifier.fillMaxWidth()) {
            val status = when {
                !active -> null
                hub.error != null -> "Problem: ${hub.error}"
                hub.buffering -> "Waiting for the PC to synthesize\u2026"
                hub.playing -> "${hub.readyAhead} ahead ready" + (hub.sleepEndsAt?.let { " \u00b7 sleep ${((it - System.currentTimeMillis()) / 60000) + 1} min" }
                    ?: if (hub.sleepEndOfChapter) " \u00b7 sleep at chapter end" else "")
                else -> null
            }
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text(status.orEmpty(), color = if (hub.error != null) MaterialTheme.colorScheme.error else colors.muted,
                    style = MaterialTheme.typography.bodySmall, maxLines = 1, modifier = Modifier.weight(1f).padding(start = 8.dp))
                if (hub.error != null) TextButton(onClick = { PlaybackHub.retry() }) { Text("Retry") }
            }
            Row(horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                IconButton(onClick = { PlaybackHub.previousChapter() }) { Icon(Icons.Default.SkipPrevious, "Previous chapter", tint = colors.text) }
                IconButton(onClick = { PlaybackHub.skip(-1) }) { Icon(Icons.Default.FastRewind, "Previous sentence", tint = colors.text) }
                FilledIconButton(onClick = { PlaybackHub.togglePlay() }, modifier = Modifier.size(52.dp)) {
                    if (hub.buffering && hub.playing.not() && active && hub.error == null) {
                        CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                    } else {
                        Icon(if (hub.playing) Icons.Default.Pause else Icons.Default.PlayArrow, if (hub.playing) "Pause" else "Play")
                    }
                }
                IconButton(onClick = { PlaybackHub.skip(1) }) { Icon(Icons.Default.FastForward, "Next sentence", tint = colors.text) }
                IconButton(onClick = { PlaybackHub.nextChapter() }) { Icon(Icons.Default.SkipNext, "Next chapter", tint = colors.text) }
            }
        }
    }
}

@Composable
private fun ReadingOptions(bookId: String, fontSp: Float, onFont: (Float) -> Unit, theme: String, onTheme: (String) -> Unit) {
    val context = LocalContext.current
    val library = remember { Library.get(context) }
    val settings = remember { ServerSettings(context) }
    val client = remember { ServerClient.get(context) }
    val scope = rememberCoroutineScope()
    val books by library.books.collectAsState()
    val meta = books.firstOrNull { it.id == bookId } ?: return
    var speed by remember { mutableFloatStateOf(settings.readerSpeed) }
    var prefetch by remember { mutableFloatStateOf(settings.prefetchCount.toFloat()) }
    var engines by remember { mutableStateOf<List<EngineInfo>>(emptyList()) }
    var voices by remember { mutableStateOf<List<ServerVoice>>(emptyList()) }
    var error by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) { runCatching { client.engines().first to client.voices().first } }
            .onSuccess { (e, v) -> engines = e.filter { it.installed }; voices = v }
            .onFailure { error = "Could not reach the PC: ${it.message}" }
    }

    fun updateBook(change: (com.androidreader.library.BookMeta) -> com.androidreader.library.BookMeta, reload: Boolean = false) {
        library.update(change(meta))
        if (PlaybackHub.state.value.bookId == bookId) {
            if (reload) PlaybackHub.reloadBook() else PlaybackHub.invalidateUpcoming()
        }
    }

    Column(
        Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text("Text size ${fontSp.toInt()}", style = MaterialTheme.typography.titleSmall)
        Slider(fontSp, onValueChange = { onFont(it) }, valueRange = 14f..32f)
        Text("Theme", style = MaterialTheme.typography.titleSmall)
        Row {
            listOf("system" to "System", "light" to "Light", "sepia" to "Sepia", "dark" to "Dark").forEach { (k, label) ->
                TextButton(onClick = { onTheme(k) }) { Text(label, fontWeight = if (theme == k) FontWeight.Bold else null) }
            }
        }
        Text(String.format(Locale.ROOT, "Speed %.2f\u00d7", speed), style = MaterialTheme.typography.titleSmall)
        Slider(speed, onValueChange = { speed = (it * 20).toInt() / 20f }, onValueChangeFinished = { PlaybackHub.setSpeed(speed) },
            valueRange = 0.5f..2.5f)
        Text("Sleep timer", style = MaterialTheme.typography.titleSmall)
        Row {
            TextButton(onClick = { PlaybackHub.setSleepMinutes(null) }) { Text("Off") }
            listOf(15, 30, 60).forEach { m -> TextButton(onClick = { PlaybackHub.setSleepMinutes(m) }) { Text("$m min") } }
            TextButton(onClick = { PlaybackHub.setSleepEndOfChapter() }) { Text("Chapter end") }
        }
        HorizontalDivider()
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Dramatized reading", style = MaterialTheme.typography.titleSmall)
                Text("Characters get their own voices and lines are performed (\u201cStop!\u201d she shouted). " +
                    "The PC analyses the book first.", style = MaterialTheme.typography.bodySmall)
            }
            Switch(meta.dramatized, onCheckedChange = { on ->
                updateBook({ it.copy(dramatized = on) }, reload = true)
                if (on) DramaUi.ensureAnalysis(context, bookId)
            })
        }
        if (meta.dramatized) DramaUi.AnalysisStatus(bookId)
        HorizontalDivider()
        Text("Voice model for this book", style = MaterialTheme.typography.titleSmall)
        error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        ChoiceRow(meta.engine.isBlank(), "Default (${settings.readerEngineId.ifBlank { "server default" }})") { updateBook({ it.copy(engine = "") }) }
        engines.forEach { e ->
            ChoiceRow(meta.engine == e.id, e.label + (e.rtf?.let { String.format(Locale.ROOT, "  \u00b7 %.1f\u00d7 real time", 1 / it) } ?: "")) {
                updateBook({ it.copy(engine = e.id) })
            }
        }
        Text(if (meta.dramatized) "Narrator voice" else "Voice", style = MaterialTheme.typography.titleSmall)
        ChoiceRow(meta.voice.isBlank(), "Default") { updateBook({ it.copy(voice = "") }) }
        voices.forEach { v ->
            ChoiceRow(meta.voice == v.id, v.name + (v.description?.let { "  \u00b7 $it" } ?: "")) { updateBook({ it.copy(voice = v.id) }) }
        }
        HorizontalDivider()
        Text("Sentences prepared ahead: ${prefetch.toInt()}", style = MaterialTheme.typography.titleSmall)
        Text("Slower, higher-quality models need more.", style = MaterialTheme.typography.bodySmall)
        Slider(prefetch, onValueChange = { prefetch = it.toInt().toFloat() }, onValueChangeFinished = { settings.prefetchCount = prefetch.toInt() },
            valueRange = 2f..20f)
        Spacer(Modifier.size(24.dp))
    }
}
