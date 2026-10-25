package com.androidreader.tools

import com.androidreader.drama.BookPayload
import com.androidreader.epub.EpubParser
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * PC-side exporter used by server/scripts/process_book.py. It runs the app's own
 * EPUB parser and quote segmentation on the JVM, so the book ID and quotation IDs
 * match what the phone produces when it imports the same file.
 *
 *   gradlew testDebugUnitTest --tests com.androidreader.tools.ExportBookTool \
 *     -PexportEpub=book.epub -PexportOut=payload.json
 *
 * Skipped during normal test runs.
 */
class ExportBookTool {
    @Test
    fun export() {
        val epub = System.getProperty("exportEpub")
        val out = System.getProperty("exportOut")
        assumeTrue("not an export run", !epub.isNullOrBlank() && !out.isNullOrBlank())
        val file = File(epub!!)
        val book = EpubParser.parse(file)
        val payload = BookPayload.build(book.title, book.author, book.language, book.chapters)
        payload.put("id", BookPayload.bookId(file))
        book.cover?.let { payload.put("cover", java.util.Base64.getEncoder().encodeToString(it)) }
        payload.put("words", book.chapters.sumOf { c -> c.paragraphs.sumOf { p -> p.text.split(' ').count { it.isNotBlank() } } })
        File(out!!).writeText(payload.toString(), Charsets.UTF_8)
    }
}
