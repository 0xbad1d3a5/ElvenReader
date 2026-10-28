package com.androidreader.library

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.androidreader.epub.Chapter
import com.androidreader.epub.EpubParser
import com.androidreader.epub.Paragraph
import com.androidreader.epub.Quotes
import com.androidreader.epub.Segment
import com.androidreader.epub.Sentences
import com.androidreader.epub.TextSpan
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** Where the reader is: a character offset inside a paragraph, stable across segmentation modes. */
data class BookPosition(val chapter: Int = 0, val paragraph: Int = 0, val offset: Int = 0)

data class BookMeta(
    val id: String,
    val title: String,
    val author: String,
    val language: String?,
    val added: Long,
    val lastOpened: Long = 0,
    val chapterCount: Int = 0,
    val position: BookPosition = BookPosition(),
    val progress: Float = 0f,
    /** Engine/voice for this book; empty = the reader defaults. */
    val engine: String = "",
    val voice: String = "",
    /** Multi-voice, expressive reading from the server's dramatization script. */
    val dramatized: Boolean = false,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("id", id).put("title", title).put("author", author).put("language", language ?: JSONObject.NULL)
        .put("added", added).put("lastOpened", lastOpened).put("chapterCount", chapterCount)
        .put("chapter", position.chapter).put("paragraph", position.paragraph).put("offset", position.offset)
        .put("progress", progress.toDouble()).put("engine", engine).put("voice", voice).put("dramatized", dramatized)

    companion object {
        fun fromJson(j: JSONObject) = BookMeta(
            id = j.getString("id"),
            title = j.optString("title"),
            author = j.optString("author"),
            language = if (j.isNull("language")) null else j.optString("language").ifBlank { null },
            added = j.optLong("added"),
            lastOpened = j.optLong("lastOpened"),
            chapterCount = j.optInt("chapterCount"),
            position = BookPosition(j.optInt("chapter"), j.optInt("paragraph"), j.optInt("offset")),
            progress = j.optDouble("progress", 0.0).toFloat(),
            engine = j.optString("engine"),
            voice = j.optString("voice"),
            dramatized = j.optBoolean("dramatized"),
        )
    }
}

/** One spoken unit: a sentence, or (dramatized) part of a sentence split at quotation marks. */
data class Utterance(
    val chapter: Int,
    val index: Int,
    val paragraph: Int,
    val start: Int,
    val end: Int,
    val text: String,
    /** Start offset of the quotation this belongs to, or null for narration. */
    val quote: Int?,
) {
    /** Stable id used by the server's dramatization script: "c3p12o40". */
    val id: String get() = "c${chapter}p${paragraph}o$start"
    val quoteId: String? get() = quote?.let { "c${chapter}p${paragraph}q$it" }
}

class BookContent(val meta: BookMeta, val chapters: List<Chapter>) {
    private val sentenceCache = HashMap<Int, List<List<TextSpan>>>()
    private val utteranceCache = HashMap<Pair<Int, Boolean>, List<Utterance>>()

    fun sentences(chapter: Int): List<List<TextSpan>> = synchronized(this) {
        sentenceCache.getOrPut(chapter) { chapters[chapter].paragraphs.map { Sentences.split(it.text) } }
    }

    fun utterances(chapter: Int, dramatized: Boolean): List<Utterance> = synchronized(this) {
        utteranceCache.getOrPut(chapter to dramatized) {
            val paragraphs = chapters[chapter].paragraphs
            val sentences = sentences(chapter)
            val out = mutableListOf<Utterance>()
            paragraphs.forEachIndexed { p, para ->
                for (seg: Segment in Quotes.segments(para.text, sentences[p], dramatized)) {
                    out += Utterance(chapter, out.size, p, seg.start, seg.end, para.text.substring(seg.start, seg.end), seg.quote)
                }
            }
            out
        }
    }

    /** Index of the utterance containing (or following) a position. */
    fun utteranceAt(position: BookPosition, dramatized: Boolean): Int {
        if (position.chapter !in chapters.indices) return 0
        val list = utterances(position.chapter, dramatized)
        val idx = list.indexOfFirst {
            it.paragraph > position.paragraph || (it.paragraph == position.paragraph && it.end > position.offset)
        }
        return if (idx < 0) (list.size - 1).coerceAtLeast(0) else idx
    }

    fun progressOf(position: BookPosition): Float {
        val total = chapters.sumOf { c -> c.paragraphs.sumOf { it.text.length } }.coerceAtLeast(1)
        var before = 0
        for (c in 0 until position.chapter.coerceAtMost(chapters.size)) before += chapters[c].paragraphs.sumOf { it.text.length }
        chapters.getOrNull(position.chapter)?.paragraphs?.let { ps ->
            for (p in 0 until position.paragraph.coerceAtMost(ps.size)) before += ps[p].text.length
        }
        return (before + position.offset).toFloat() / total
    }
}

/** EPUBs imported into app storage: files/books/<id>/{book.epub, meta.json, content.json, cover.jpg}. */
class Library private constructor(context: Context) {
    private val root = File(context.filesDir, "books").apply { mkdirs() }
    private val resolver = context.contentResolver
    private val _books = MutableStateFlow(loadAll())
    val books: StateFlow<List<BookMeta>> = _books
    private var cached: BookContent? = null

    private fun dir(id: String) = File(root, id)
    fun coverFile(id: String) = File(dir(id), "cover.jpg")
    fun epubFile(id: String) = File(dir(id), "book.epub")

    private fun loadAll(): List<BookMeta> = (root.listFiles() ?: emptyArray()).mapNotNull { d ->
        runCatching { BookMeta.fromJson(JSONObject(File(d, "meta.json").readText())) }.getOrNull()
    }.sortedByDescending { maxOf(it.lastOpened, it.added) }

    fun get(id: String): BookMeta? = _books.value.firstOrNull { it.id == id }

    @Synchronized
    fun update(meta: BookMeta) {
        File(dir(meta.id), "meta.json").writeText(meta.toJson().toString())
        _books.value = (_books.value.filter { it.id != meta.id } + meta).sortedByDescending { maxOf(it.lastOpened, it.added) }
        cached?.let { if (it.meta.id == meta.id) cached = BookContent(meta, it.chapters).also { c -> cached = c } }
    }

    fun savePosition(id: String, position: BookPosition) {
        val meta = get(id) ?: return
        val content = cached?.takeIf { it.meta.id == id }
        update(meta.copy(position = position, progress = content?.progressOf(position) ?: meta.progress))
    }

    suspend fun import(uri: Uri): BookMeta = withContext(Dispatchers.IO) {
        val tmp = File.createTempFile("import", ".epub", root)
        try {
            resolver.openInputStream(uri)?.use { input -> tmp.outputStream().use { input.copyTo(it) } }
                ?: throw IllegalArgumentException("Could not open the file")
            val id = com.androidreader.drama.BookPayload.bookId(tmp)
            get(id)?.let { return@withContext it }
            val book = EpubParser.parse(tmp)
            val d = dir(id).apply { mkdirs() }
            tmp.copyTo(File(d, "book.epub"), overwrite = true)
            writeContent(File(d, "content.json"), book.title, book.author, book.language, book.chapters)
            book.cover?.let { saveCover(it, coverFile(id)) }
            val meta = BookMeta(
                id = id, title = book.title, author = book.author, language = book.language,
                added = System.currentTimeMillis(), chapterCount = book.chapters.size,
            )
            update(meta)
            meta
        } finally {
            tmp.delete()
        }
    }

    private val pictures = object : android.util.LruCache<String, Bitmap>(48 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.byteCount
    }

    /**
     * A picture from the book's EPUB, scaled down to about [maxWidth] pixels wide; null if it is missing,
     * unreadable, or tiny (ornaments and inline glyphs are left out).
     */
    fun picture(id: String, path: String, maxWidth: Int): Bitmap? {
        val key = "$id/$path/$maxWidth"
        pictures.get(key)?.let { return it }
        val bytes = runCatching {
            java.util.zip.ZipFile(File(dir(id), "book.epub")).use { zip ->
                zip.getEntry(path)?.let { e -> zip.getInputStream(e).use { it.readBytes() } }
            }
        }.getOrNull() ?: return null
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
        if (opts.outWidth < MIN_PICTURE_PX || opts.outHeight < MIN_PICTURE_PX) return null
        var sample = 1
        while (opts.outWidth / (sample * 2) >= maxWidth) sample *= 2
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: return null
        pictures.put(key, bmp)
        return bmp
    }

    fun delete(id: String) {
        dir(id).deleteRecursively()
        _books.value = _books.value.filter { it.id != id }
        if (cached?.meta?.id == id) cached = null
    }

    @Synchronized
    fun content(id: String): BookContent {
        cached?.let { if (it.meta.id == id) return it }
        var meta = get(id) ?: throw IllegalArgumentException("Unknown book $id")
        val file = File(dir(id), "content.json")
        val json = JSONObject(file.readText())
        var chapters = json.getJSONArray("chapters").let { arr ->
            (0 until arr.length()).map { i ->
                val c = arr.getJSONObject(i)
                val ps = c.getJSONArray("p")
                Chapter(c.getString("title"), (0 until ps.length()).map { k ->
                    val p = ps.getJSONArray(k)
                    Paragraph(p.getString(0), p.optInt(1) == 1, strings(p.optJSONArray(2)))
                }, strings(c.optJSONArray("end")))
            }
        }
        val epub = File(dir(id), "book.epub")
        if (json.optInt("parser", 1) < EpubParser.VERSION && epub.exists()) {
            // Parsed by an older version that split the book differently: parse again and keep the place.
            runCatching { EpubParser.parse(epub) }.getOrNull()?.let { book ->
                val (c, p) = EpubParser.remap(chapters, book.chapters, meta.position.chapter, meta.position.paragraph)
                chapters = book.chapters
                writeContent(file, book.title, book.author, book.language, chapters)
                meta = meta.copy(chapterCount = chapters.size, position = meta.position.copy(chapter = c, paragraph = p))
                update(meta)
            }
        }
        return BookContent(meta, chapters).also { cached = it }
    }

    companion object {
        private const val MIN_PICTURE_PX = 64
        @Volatile private var instance: Library? = null
        fun get(context: Context): Library = instance ?: synchronized(this) {
            instance ?: Library(context.applicationContext).also { instance = it }
        }

        fun writeContent(file: File, title: String, author: String, language: String?, chapters: List<Chapter>) {
            val arr = JSONArray()
            for (c in chapters) {
                val ps = JSONArray()
                c.paragraphs.forEach { p ->
                    val row = JSONArray().put(p.text).put(if (p.heading) 1 else 0)
                    if (p.images.isNotEmpty()) row.put(JSONArray(p.images))
                    ps.put(row)
                }
                val ch = JSONObject().put("title", c.title).put("p", ps)
                if (c.endImages.isNotEmpty()) ch.put("end", JSONArray(c.endImages))
                arr.put(ch)
            }
            file.writeText(
                JSONObject().put("title", title).put("author", author).put("language", language ?: JSONObject.NULL)
                    .put("parser", EpubParser.VERSION).put("chapters", arr).toString(),
            )
        }

        private fun strings(arr: JSONArray?): List<String> = if (arr == null) emptyList() else (0 until arr.length()).map { arr.getString(it) }

        private fun saveCover(bytes: ByteArray, dest: File) {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            var sample = 1
            while (opts.outWidth / (sample * 2) >= 300 && opts.outHeight / (sample * 2) >= 300) sample *= 2
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
                ?: return
            dest.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        }
    }
}
