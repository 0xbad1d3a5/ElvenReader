package com.androidreader.drama

import com.androidreader.epub.Chapter
import com.androidreader.epub.Quotes
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * What the server's dramatization needs: every paragraph plus each quotation's
 * stable ID and offsets. Pure JVM code, shared by the app and the PC-side
 * process-book tool so both produce identical IDs.
 */
object BookPayload {
    fun build(title: String, author: String, language: String?, chapters: List<Chapter>): JSONObject {
        val arr = JSONArray()
        chapters.forEachIndexed { ci, chapter ->
            val paragraphs = JSONArray()
            val quotes = JSONArray()
            chapter.paragraphs.forEachIndexed { pi, p ->
                paragraphs.put(p.text)
                for (r in Quotes.regions(p.text)) {
                    quotes.put(JSONArray().put("c${ci}p${pi}q${r.start}").put(pi).put(r.start).put(r.end))
                }
            }
            val chapterJson = JSONObject().put("title", chapter.title).put("paragraphs", paragraphs).put("quotes", quotes)
            // Pictures (zip paths in the EPUB) shown before a paragraph, and after the last one.
            val images = JSONObject()
            chapter.paragraphs.forEachIndexed { pi, p -> if (p.images.isNotEmpty()) images.put(pi.toString(), JSONArray(p.images)) }
            if (images.length() > 0) chapterJson.put("images", images)
            if (chapter.endImages.isNotEmpty()) chapterJson.put("end_images", JSONArray(chapter.endImages))
            arr.put(chapterJson)
        }
        return JSONObject()
            .put("title", title)
            .put("author", author)
            .put("language", language ?: JSONObject.NULL)
            .put("chapters", arr)
    }

    /** The library's book ID: the first 16 hex digits of the EPUB file's SHA-1. */
    fun bookId(file: File): String {
        val md = MessageDigest.getInstance("SHA-1")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }.take(16)
    }
}
