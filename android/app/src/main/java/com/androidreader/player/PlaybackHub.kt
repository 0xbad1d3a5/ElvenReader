package com.androidreader.player

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import com.androidreader.library.BookContent
import com.androidreader.library.BookMeta
import com.androidreader.library.BookPosition
import com.androidreader.library.Library
import com.androidreader.library.Utterance
import com.androidreader.net.ServerSettings
import com.androidreader.net.TtsRequest
import com.androidreader.tts.LanguageMapper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.IOException

/** How a dramatized book voices one utterance (from the server's script). */
data class Voicing(val voice: String?, val delivery: String?, val speaker: String? = null)

/**
 * In-process owner of the reading session: which book and utterance is playing,
 * the ExoPlayer playlist (one item per utterance), prefetching and the sleep timer.
 * The [PlaybackService] creates the player; the UI talks to this object directly
 * (same process) and the MediaSession exposes it to the lock screen and Bluetooth.
 */
@androidx.annotation.OptIn(UnstableApi::class)
object PlaybackHub {
    data class State(
        val bookId: String? = null,
        val chapter: Int = 0,
        val index: Int = 0,
        val dramatized: Boolean = false,
        val playing: Boolean = false,
        val buffering: Boolean = false,
        val error: String? = null,
        val sleepEndsAt: Long? = null,
        val sleepEndOfChapter: Boolean = false,
        /** Utterances ready ahead of the current one (for the buffer indicator). */
        val readyAhead: Int = 0,
        val generation: Int = 0,
    )

    private val _state = MutableStateFlow(State())
    val state: StateFlow<State> = _state

    private lateinit var appContext: Context
    private lateinit var settings: ServerSettings
    private lateinit var fetcher: SpeechFetcher
    private val main = Handler(Looper.getMainLooper())

    @Volatile var player: ExoPlayer? = null
        private set
    @Volatile private var content: BookContent? = null
    @Volatile private var generation = 0
    private val chaptersInPlaylist = sortedSetOf<Int>()
    private var pendingOpen: (() -> Unit)? = null
    private var sleepRunnable: Runnable? = null

    /** Provided by the dramatization feature: voice and delivery for an utterance. */
    @Volatile var voicing: ((BookMeta, Utterance) -> Voicing?)? = null

    fun init(context: Context) {
        if (::appContext.isInitialized) return
        appContext = context.applicationContext
        settings = ServerSettings(appContext)
        fetcher = SpeechFetcher(appContext)
        fetcher.listener = { _, _ -> main.post { updateReadyAhead() } }
    }

    // ------------------------------------------------------------ service side

    fun attach(exo: ExoPlayer) {
        player = exo
        exo.addListener(listener)
        exo.setPlaybackSpeed(settings.readerSpeed)
        pendingOpen?.let { it(); pendingOpen = null }
    }

    fun detach() {
        player?.removeListener(listener)
        player = null
        cancelSleep()
    }

    private val listener = object : Player.Listener {
        override fun onMediaItemTransition(mediaItem: MediaItem?, reason: Int) {
            val (c, i) = parse(mediaItem?.mediaId) ?: return
            val previous = _state.value
            if (previous.sleepEndOfChapter && c != previous.chapter && reason == Player.MEDIA_ITEM_TRANSITION_REASON_AUTO) {
                player?.pause()
                _state.value = _state.value.copy(sleepEndOfChapter = false)
            }
            _state.value = _state.value.copy(chapter = c, index = i, error = null)
            savePosition(c, i)
            extendPlaylist(c, i)
            prefetch(c, i)
        }

        override fun onIsPlayingChanged(isPlaying: Boolean) {
            _state.value = _state.value.copy(playing = isPlaying)
        }

        override fun onPlaybackStateChanged(playbackState: Int) {
            _state.value = _state.value.copy(buffering = playbackState == Player.STATE_BUFFERING)
        }

        override fun onPlayerError(error: PlaybackException) {
            var cause: Throwable? = error
            while (cause?.cause != null && cause !is IOException) cause = cause.cause
            _state.value = _state.value.copy(error = cause?.message ?: error.errorCodeName, playing = false)
        }
    }

    // ------------------------------------------------------------ UI side

    /** Opens a book at its saved position (or [position]) and optionally starts playing. */
    fun open(bookId: String, position: BookPosition? = null, play: Boolean = false) {
        val action = {
            val library = Library.get(appContext)
            val book = library.content(bookId)
            val meta = library.get(bookId) ?: book.meta
            library.update(meta.copy(lastOpened = System.currentTimeMillis()))
            val pos = position ?: meta.position
            content = book
            fetcher.clear()
            val index = book.utteranceAt(pos, meta.dramatized)
            _state.value = State(bookId = bookId, chapter = pos.chapter, index = index, dramatized = meta.dramatized,
                generation = generation)
            loadChapter(pos.chapter.coerceIn(0, book.chapters.size - 1), index, play)
        }
        if (player == null) pendingOpen = action else action()
    }

    val currentMeta: BookMeta?
        get() = content?.let { Library.get(appContext).get(it.meta.id) ?: it.meta }

    val currentContent: BookContent? get() = content

    fun playFrom(chapter: Int, index: Int) {
        val book = content ?: return
        if (chapter !in book.chapters.indices) return
        val p = player ?: return
        val itemIndex = (0 until p.mediaItemCount).firstOrNull { parse(p.getMediaItemAt(it).mediaId) == (chapter to index) }
        if (itemIndex != null) {
            p.seekTo(itemIndex, 0)
            _state.value = _state.value.copy(chapter = chapter, index = index, error = null)
            prefetch(chapter, index)
            if (!p.isPlaying) p.play()
        } else {
            loadChapter(chapter, index, play = true)
        }
    }

    fun togglePlay() {
        val p = player ?: return
        if (p.isPlaying) {
            p.pause()
        } else {
            if (_state.value.error != null || p.playbackState == Player.STATE_IDLE) p.prepare()
            if (p.playbackState == Player.STATE_ENDED) nextChapter() else p.play()
        }
    }

    fun pause() = player?.pause()

    fun retry() {
        _state.value = _state.value.copy(error = null)
        player?.let { it.prepare(); it.play() }
    }

    fun skip(delta: Int) {
        val s = _state.value
        val book = content ?: return
        val count = book.utterances(s.chapter, s.dramatized).size
        val target = s.index + delta
        when {
            target in 0 until count -> playFrom(s.chapter, target)
            target >= count && s.chapter + 1 < book.chapters.size -> playFrom(s.chapter + 1, 0)
            target < 0 && s.chapter > 0 -> playFrom(s.chapter - 1, 0)
        }
    }

    fun nextChapter() {
        val s = _state.value
        if (content?.chapters?.size?.let { s.chapter + 1 < it } == true) playFrom(s.chapter + 1, 0)
    }

    fun previousChapter() {
        val s = _state.value
        playFrom(if (s.index > 2 || s.chapter == 0) s.chapter else s.chapter - 1, 0)
    }

    fun setSpeed(speed: Float) {
        settings.readerSpeed = speed
        player?.setPlaybackSpeed(speed)
    }

    fun setSleepMinutes(minutes: Int?) {
        cancelSleep()
        if (minutes == null || minutes <= 0) {
            _state.value = _state.value.copy(sleepEndsAt = null, sleepEndOfChapter = false)
            return
        }
        val r = Runnable {
            player?.pause()
            _state.value = _state.value.copy(sleepEndsAt = null)
        }
        sleepRunnable = r
        main.postDelayed(r, minutes * 60_000L)
        _state.value = _state.value.copy(sleepEndsAt = System.currentTimeMillis() + minutes * 60_000L, sleepEndOfChapter = false)
    }

    fun setSleepEndOfChapter() {
        cancelSleep()
        _state.value = _state.value.copy(sleepEndsAt = null, sleepEndOfChapter = true)
    }

    private fun cancelSleep() {
        sleepRunnable?.let { main.removeCallbacks(it) }
        sleepRunnable = null
    }

    /**
     * Something that changes the sound of upcoming utterances changed (a pronunciation,
     * the voice, the engine, a speaker assignment): drop buffered audio after the current
     * utterance and re-synthesize it.
     */
    fun invalidateUpcoming() {
        val p = player ?: return
        val book = content ?: return
        generation++
        val current = p.currentMediaItemIndex
        val keep = mutableSetOf<String>()
        p.currentMediaItem?.let { item -> parse(item.mediaId)?.let { (c, i) -> keep += key(c, i, currentGenerationOf(item)) } }
        fetcher.retainOnly(keep)
        if (current + 1 < p.mediaItemCount) {
            val items = (current + 1 until p.mediaItemCount).mapNotNull { idx ->
                parse(p.getMediaItemAt(idx).mediaId)?.let { (c, i) -> mediaItem(book, c, i) }
            }
            p.replaceMediaItems(current + 1, p.mediaItemCount, items)
        }
        _state.value = _state.value.copy(generation = generation)
        val s = _state.value
        prefetch(s.chapter, s.index)
    }

    /** Re-reads the book's settings (engine, voice, dramatized) and restarts at the current spot. */
    fun reloadBook() {
        val s = _state.value
        val book = content ?: return
        val wasPlaying = player?.isPlaying == true
        val u = book.utterances(s.chapter, s.dramatized).getOrNull(s.index)
        val pos = u?.let { BookPosition(it.chapter, it.paragraph, it.start) } ?: BookPosition(s.chapter)
        generation++
        open(book.meta.id, pos, wasPlaying)
    }

    // ------------------------------------------------------------ playlist

    private fun loadChapter(chapter: Int, index: Int, play: Boolean) {
        val p = player ?: return
        val book = content ?: return
        val dramatized = currentMeta?.dramatized == true
        val count = book.utterances(chapter, dramatized).size
        chaptersInPlaylist.clear()
        chaptersInPlaylist += chapter
        val items = (0 until count).map { mediaItem(book, chapter, it) }
        _state.value = _state.value.copy(chapter = chapter, index = index, dramatized = dramatized, error = null)
        p.setMediaItems(items, index.coerceIn(0, (count - 1).coerceAtLeast(0)), 0)
        p.prepare()
        prefetch(chapter, index)
        extendPlaylist(chapter, index)
        if (play) p.play()
    }

    /** Appends the next chapter when nearing the end, so playback continues without a gap. */
    private fun extendPlaylist(chapter: Int, index: Int) {
        val p = player ?: return
        val book = content ?: return
        val dramatized = _state.value.dramatized
        val count = book.utterances(chapter, dramatized).size
        val next = chapter + 1
        if (count - index <= 8 && next < book.chapters.size && next !in chaptersInPlaylist) {
            chaptersInPlaylist += next
            p.addMediaItems(book.utterances(next, dramatized).indices.map { mediaItem(book, next, it) })
        }
    }

    private fun mediaItem(book: BookContent, chapter: Int, index: Int): MediaItem {
        val dramatized = _state.value.dramatized
        val uri = Uri.Builder().scheme("areader").authority("u")
            .appendQueryParameter("b", book.meta.id)
            .appendQueryParameter("c", chapter.toString())
            .appendQueryParameter("i", index.toString())
            .appendQueryParameter("d", if (dramatized) "1" else "0")
            .appendQueryParameter("g", generation.toString())
            .build()
        val cover = Library.get(appContext).coverFile(book.meta.id)
        val metadata = MediaMetadata.Builder()
            .setTitle(book.chapters[chapter].title)
            .setArtist(book.meta.title)
            .setAlbumTitle(book.meta.title)
            .apply { if (cover.exists()) setArtworkUri(Uri.fromFile(cover)) }
            .build()
        return MediaItem.Builder().setMediaId("$chapter:$index").setUri(uri).setMediaMetadata(metadata).build()
    }

    private fun parse(mediaId: String?): Pair<Int, Int>? {
        val parts = mediaId?.split(':') ?: return null
        if (parts.size != 2) return null
        return (parts[0].toIntOrNull() ?: return null) to (parts[1].toIntOrNull() ?: return null)
    }

    private fun currentGenerationOf(item: MediaItem): Int =
        item.localConfiguration?.uri?.getQueryParameter("g")?.toIntOrNull() ?: generation

    // ------------------------------------------------------------ audio

    private fun key(chapter: Int, index: Int, gen: Int): String {
        val s = _state.value
        return "${s.bookId}|$chapter|$index|${if (s.dramatized) 1 else 0}|$gen"
    }

    /** The server request for one utterance. */
    fun requestFor(chapter: Int, index: Int): TtsRequest? {
        val book = content ?: return null
        val meta = currentMeta ?: book.meta
        val u = book.utterances(chapter, _state.value.dramatized).getOrNull(index) ?: return null
        val v = if (_state.value.dramatized) voicing?.invoke(meta, u) else null
        return TtsRequest(
            text = u.text,
            engine = meta.engine.ifBlank { settings.readerEngineId }.ifBlank { null },
            voice = v?.voice ?: meta.voice.ifBlank { settings.readerVoiceId }.ifBlank { null },
            language = LanguageMapper.displayName(meta.language?.substringBefore('-')),
            delivery = v?.delivery,
            book = meta.id,
            sentencePauseMs = pauseAfter(book, u),
        )
    }

    /** Full pause after a paragraph's last utterance, a short one between quote and narration. */
    private fun pauseAfter(book: BookContent, u: Utterance): Int {
        val base = settings.sentencePauseMs
        val list = book.utterances(u.chapter, _state.value.dramatized)
        val next = list.getOrNull(u.index + 1)
        return when {
            next == null || next.paragraph != u.paragraph -> (base * 1.6).toInt()
            else -> base
        }
    }

    /** Called on ExoPlayer's loading thread; blocks until the PC has synthesized the utterance. */
    fun awaitAudio(uri: Uri): SpeechAudio {
        val c = uri.getQueryParameter("c")?.toIntOrNull() ?: throw IOException("bad uri")
        val i = uri.getQueryParameter("i")?.toIntOrNull() ?: throw IOException("bad uri")
        val g = uri.getQueryParameter("g")?.toIntOrNull() ?: 0
        val req = requestFor(c, i) ?: throw IOException("utterance not found")
        return fetcher.await(key(c, i, g), req, 300_000)
    }

    private fun prefetch(chapter: Int, index: Int) {
        val book = content ?: return
        val dramatized = _state.value.dramatized
        val ahead = settings.prefetchCount.coerceIn(1, 30)
        val window = mutableListOf<Pair<Int, Int>>()
        var c = chapter
        var i = index
        while (window.size <= ahead && c < book.chapters.size) {
            val n = book.utterances(c, dramatized).size
            if (i < n) {
                window += c to i
                i++
            } else {
                c++
                i = 0
            }
        }
        val keys = window.map { (wc, wi) -> key(wc, wi, generation) }
        val previous = key(chapter, index - 1, generation)
        fetcher.retainOnly((keys + previous + currentKeys()).toSet())
        window.forEachIndexed { n, (wc, wi) ->
            requestFor(wc, wi)?.let { fetcher.request(keys[n], it, urgent = n == 0) }
        }
        updateReadyAhead()
    }

    /** Keys of items ExoPlayer may still be holding with an older generation. */
    private fun currentKeys(): List<String> {
        val item = player?.currentMediaItem ?: return emptyList()
        val (c, i) = parse(item.mediaId) ?: return emptyList()
        return listOf(key(c, i, currentGenerationOf(item)))
    }

    private fun updateReadyAhead() {
        val s = _state.value
        val book = content ?: return
        var n = 0
        var c = s.chapter
        var i = s.index + 1
        while (n < 30 && c < book.chapters.size) {
            val count = book.utterances(c, s.dramatized).size
            if (i >= count) {
                c++; i = 0; continue
            }
            if (!fetcher.isReady(key(c, i, generation))) break
            n++; i++
        }
        if (n != s.readyAhead) _state.value = s.copy(readyAhead = n)
    }

    private fun savePosition(chapter: Int, index: Int) {
        val book = content ?: return
        val u = book.utterances(chapter, _state.value.dramatized).getOrNull(index) ?: return
        Library.get(appContext).savePosition(book.meta.id, BookPosition(chapter, u.paragraph, u.start))
    }
}
