package com.androidreader.drama

import android.content.Context
import com.androidreader.epub.Quotes
import com.androidreader.library.BookMeta
import com.androidreader.library.Library
import com.androidreader.library.Utterance
import com.androidreader.net.ServerClient
import com.androidreader.net.ServerHttpException
import com.androidreader.net.optStringOrNull
import com.androidreader.player.PlaybackHub
import com.androidreader.player.Voicing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class Character(
    val id: String,
    val name: String,
    val aliases: List<String>,
    val gender: String,
    val age: String,
    val description: String,
    val voice: String?,
    val lines: Int,
)

data class Line(val speaker: String?, val delivery: String?, val corrected: Boolean)

/** The server's analysis of a book: who says each quotation and how. */
data class BookScript(
    val status: String = "none",
    val progress: Float = 0f,
    val detail: String = "",
    val characters: List<Character> = emptyList(),
    val lines: Map<String, Line> = emptyMap(),
    val model: String = "",
    /** (chapter, first paragraph, narrating character id or null), in reading order. */
    val narration: List<Triple<Int, Int, String?>> = emptyList(),
) {
    fun character(id: String?) = characters.firstOrNull { it.id == id }

    /** Who narrates at a position (first-person books); null = the book's narrator voice. */
    fun narratorAt(chapter: Int, paragraph: Int): String? =
        narration.lastOrNull { (c, p, _) -> c < chapter || (c == chapter && p <= paragraph) }?.third

    companion object {
        fun fromJson(j: JSONObject): BookScript {
            val chars = j.optJSONArray("characters") ?: JSONArray()
            val lines = j.optJSONObject("lines") ?: JSONObject()
            val narration = j.optJSONArray("narration") ?: JSONArray()
            return BookScript(
                status = j.optString("status", "none"),
                progress = j.optDouble("progress", 0.0).toFloat(),
                detail = j.optString("detail"),
                model = j.optString("model"),
                narration = (0 until narration.length()).mapNotNull { i ->
                    narration.optJSONArray(i)?.let { e ->
                        Triple(e.optInt(0), e.optInt(1), if (e.isNull(2)) null else e.optString(2))
                    }
                },
                characters = (0 until chars.length()).map { i ->
                    val c = chars.getJSONObject(i)
                    val aliases = c.optJSONArray("aliases") ?: JSONArray()
                    Character(
                        id = c.getString("id"),
                        name = c.optString("name"),
                        aliases = (0 until aliases.length()).map { aliases.getString(it) },
                        gender = c.optString("gender"),
                        age = c.optString("age"),
                        description = c.optString("description"),
                        voice = c.optStringOrNull("voice"),
                        lines = c.optInt("lines"),
                    )
                },
                lines = lines.keys().asSequence().associateWith { k ->
                    val l = lines.getJSONObject(k)
                    Line(l.optStringOrNull("speaker"), l.optStringOrNull("delivery"), l.optBoolean("corrected"))
                },
            )
        }
    }
}

/**
 * Uploads books for analysis, follows progress and answers [PlaybackHub.voicing]:
 * a quotation is spoken in its character's voice with its delivery; narration
 * keeps the book's narrator voice.
 */
object Dramatization {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _scripts = MutableStateFlow<Map<String, BookScript>>(emptyMap())
    val scripts: StateFlow<Map<String, BookScript>> = _scripts
    private val polls = HashMap<String, Job>()
    private lateinit var appContext: Context

    fun init(context: Context) {
        if (::appContext.isInitialized) return
        appContext = context.applicationContext
        PlaybackHub.voicing = ::voicing
    }

    private fun cacheFile(bookId: String) = File(File(appContext.filesDir, "books/$bookId"), "script.json")

    fun script(bookId: String): BookScript? = _scripts.value[bookId] ?: runCatching {
        BookScript.fromJson(JSONObject(cacheFile(bookId).readText())).also { put(bookId, it) }
    }.getOrNull()

    private fun put(bookId: String, script: BookScript, raw: JSONObject? = null) {
        _scripts.value = _scripts.value + (bookId to script)
        if (raw != null) runCatching { cacheFile(bookId).writeText(raw.toString()) }
    }

    fun voicing(meta: BookMeta, u: Utterance): Voicing? {
        val script = script(meta.id) ?: return null
        val qid = u.quoteId
        if (qid == null) {
            // First-person books: narration in the narrating character's own voice.
            val narrator = script.character(script.narratorAt(u.chapter, u.paragraph)) ?: return null
            return narrator.voice?.let { Voicing(voice = it, delivery = null, speaker = narrator.name) }
        }
        val line = script.lines[qid] ?: return null
        val character = script.character(line.speaker)
        return Voicing(voice = character?.voice, delivery = line.delivery, speaker = character?.name)
    }

    /** The book's text and quotation offsets, exactly as the phone segments them (plus the cover). */
    fun bookPayload(bookId: String): JSONObject {
        val library = Library.get(appContext)
        val content = library.content(bookId)
        val payload = BookPayload.build(content.meta.title, content.meta.author, content.meta.language, content.chapters)
        library.coverFile(bookId).takeIf { it.exists() }?.let {
            payload.put("cover", android.util.Base64.encodeToString(it.readBytes(), android.util.Base64.NO_WRAP))
        }
        return payload
    }

    /** Uploads the book if the server doesn't have it and starts (or resumes) the analysis. */
    fun ensureAnalysis(bookId: String) {
        scope.launch {
            val client = ServerClient.get(appContext)
            runCatching {
                val current = refresh(bookId)
                if (current == null || current.status == "none" || current.status == "error") {
                    client.postJson("/api/books/$bookId", bookPayload(bookId).put("analyze", true), slow = true)
                }
                follow(bookId)
            }.onFailure { e ->
                put(bookId, (script(bookId) ?: BookScript()).copy(status = "error", detail = e.message ?: "upload failed"))
            }
        }
    }

    /** Polls the server while the analysis runs; new lines re-voice what is buffered. */
    fun follow(bookId: String) {
        synchronized(polls) {
            if (polls[bookId]?.isActive == true) return
            polls[bookId] = scope.launch {
                var lastLines = -1
                while (isActive) {
                    val s = runCatching { refresh(bookId) }.getOrNull()
                    if (s != null && s.lines.size != lastLines) {
                        if (lastLines >= 0 && PlaybackHub.state.value.bookId == bookId) {
                            withContext(Dispatchers.Main) { PlaybackHub.invalidateUpcoming() }
                        }
                        lastLines = s.lines.size
                    }
                    if (s == null || s.status !in setOf("queued", "running", "casting")) break
                    delay(5000)
                }
            }
        }
    }

    fun refresh(bookId: String): BookScript? {
        val raw = try {
            ServerClient.get(appContext).getJson("/api/books/$bookId/script")
        } catch (e: ServerHttpException) {
            if (e.code == 404) return null else throw e
        }
        val s = BookScript.fromJson(raw)
        put(bookId, s, raw)
        return s
    }

    fun updateLine(bookId: String, quoteId: String, speaker: String?, delivery: String?) {
        val body = JSONObject().put("speaker", speaker ?: JSONObject.NULL).put("delivery", delivery ?: "")
        ServerClient.get(appContext).patchJson("/api/books/$bookId/lines/$quoteId", body)
        refresh(bookId)
    }

    fun updateCharacter(bookId: String, id: String, changes: JSONObject) {
        ServerClient.get(appContext).patchJson("/api/books/$bookId/characters/$id", changes)
        refresh(bookId)
    }

    /** Starts voice creation on the PC; progress arrives through [follow]. */
    fun castVoices(bookId: String, engine: String?) {
        val body = JSONObject()
        engine?.let { body.put("engine", it) }
        ServerClient.get(appContext).postJson("/api/books/$bookId/cast", body)
        refresh(bookId)
        follow(bookId)
    }
}
