package com.androidreader

import com.androidreader.epub.Chapter
import com.androidreader.epub.EpubParser
import com.androidreader.epub.Paragraph
import com.androidreader.epub.Quotes
import com.androidreader.epub.Sentences
import com.androidreader.library.BookContent
import com.androidreader.library.BookMeta
import com.androidreader.library.BookPosition
import com.androidreader.net.parseServerUrls
import com.androidreader.net.urlKind
import com.androidreader.ui.wordAt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class SentenceTest {
    private fun split(text: String) = Sentences.split(text).map { text.substring(it.start, it.end) }

    @Test
    fun splitsOnSentenceEndsAndKeepsClosingQuotes() {
        assertEquals(
            listOf("It rained.", "\u201cStop!\u201d she shouted.", "Nobody moved?", "Then\u2026 silence."),
            split("It rained. \u201cStop!\u201d she shouted. Nobody moved? Then\u2026 silence."),
        )
    }

    @Test
    fun keepsAbbreviationsInitialsAndDecimals() {
        assertEquals(
            listOf("Mr. Smith met Dr. Jones at 3.30 p.m. on St. James's Street.", "J. K. Rowling waved."),
            split("Mr. Smith met Dr. Jones at 3.30 p.m. on St. James's Street. J. K. Rowling waved."),
        )
        assertEquals(listOf("We saw e.g. cats and dogs."), split("We saw e.g. cats and dogs."))
    }

    @Test
    fun quoteEndingInQuestionStartsNewSentenceOnlyBeforeCapital() {
        assertEquals(listOf("\u201cWhy?\u201d asked Tom.", "He left."), split("\u201cWhy?\u201d asked Tom. He left."))
        assertEquals(listOf("\u201cWhy?\u201d", "Tom left."), split("\u201cWhy?\u201d Tom left."))
    }

    @Test
    fun splitsVeryLongSentencesAtClauses() {
        val clause = "the wind kept howling across the empty moor without pause or mercy"
        val text = (1..8).joinToString("; ") { clause } + "."
        val parts = Sentences.split(text)
        assertTrue(parts.size > 1)
        assertTrue(parts.all { it.length <= Sentences.MAX_CHARS })
        assertEquals(text.replace(" ", ""), parts.joinToString("") { text.substring(it.start, it.end) }.replace(" ", ""))
    }

    @Test
    fun offsetsMapBackToTheOriginalText() {
        val text = "  One.  Two!   \u201cThree?\u201d  "
        val spans = Sentences.split(text)
        assertEquals(listOf("One.", "Two!", "\u201cThree?\u201d"), spans.map { text.substring(it.start, it.end) })
    }
}

class QuoteTest {
    @Test
    fun findsCurlyStraightAndBritishQuotes() {
        val text = "\u201cStop!\u201d she said. He said \"no\" twice. \u2018Don\u2019t,\u2019 she begged. It\u2019s 6\" long."
        val quotes = Quotes.regions(text).map { text.substring(it.start, it.end) }
        assertEquals(listOf("\u201cStop!\u201d", "\"no\"", "\u2018Don\u2019t,\u2019"), quotes)
    }

    @Test
    fun unclosedQuoteRunsToParagraphEnd() {
        val text = "\u201cI went home. Then I slept."
        assertEquals(listOf(text), Quotes.regions(text).map { text.substring(it.start, it.end) })
    }

    @Test
    fun dramatizedSegmentsSplitSpeechFromNarration() {
        val text = "\u201cStop!\u201d she shouted. \u201cDon\u2019t move. Please.\u201d He froze."
        val segs = Quotes.segments(text, Sentences.split(text), splitQuotes = true)
        assertEquals(
            listOf("\u201cStop!\u201d", "she shouted.", "\u201cDon\u2019t move.", "Please.\u201d", "He froze."),
            segs.map { text.substring(it.start, it.end) },
        )
        // Both sentences of the second quotation share its id; narration has none.
        assertNull(segs[1].quote)
        assertEquals(segs[2].quote, segs[3].quote)
        assertNull(segs[4].quote)
        assertEquals(0, segs[0].quote)
    }

    @Test
    fun plainSegmentsAreSentences() {
        val text = "\u201cStop!\u201d she shouted. He froze."
        assertEquals(2, Quotes.segments(text, Sentences.split(text), splitQuotes = false).size)
    }

    @Test
    fun punctuationOnlyPiecesJoinNeighbours() {
        val text = "\u201cYes\u201d, he said."
        val segs = Quotes.segments(text, Sentences.split(text), splitQuotes = true)
        assertEquals(listOf("\u201cYes\u201d,", "he said."), segs.map { text.substring(it.start, it.end) })
    }
}

class ReaderLogicTest {
    @Test
    fun wordAtFindsWholeWordsAndDropsPossessives() {
        val text = "Hermione\u2019s wand, and Ron-like courage."
        assertEquals("Hermione", wordAt(text, 3))
        assertEquals("wand", wordAt(text, 12))
        assertEquals("Ron-like", wordAt(text, 22))
        assertNull(wordAt(text, 16))
    }

    @Test
    fun serverUrlsAreParsedAndClassified() {
        assertEquals(
            listOf("http://192.168.86.10:3950", "http://10.8.0.2:3950", "http://my-pc.vpn.example:3950"),
            parseServerUrls("192.168.86.10\n10.8.0.2, my-pc.vpn.example\n192.168.86.10"),
        )
        assertEquals("lan", urlKind("http://192.168.86.10:3950"))
        assertEquals("lan", urlKind("http://10.8.0.2:3950"))
        assertEquals("remote", urlKind("http://my-pc.vpn.example:3950"))
        assertEquals("remote", urlKind("http://100.101.102.103:3950"))
    }

    @Test
    fun utterancesAndPositionsAreStableAcrossModes() {
        val chapter = Chapter("One", listOf(
            Paragraph("Chapter One", heading = true),
            Paragraph("\u201cStop!\u201d she shouted. He froze. Then he ran."),
        ))
        val book = BookContent(BookMeta("b", "T", "A", "en", 0), listOf(chapter))
        val plain = book.utterances(0, dramatized = false)
        val drama = book.utterances(0, dramatized = true)
        assertEquals(listOf("Chapter One", "\u201cStop!\u201d she shouted.", "He froze.", "Then he ran."), plain.map { it.text })
        assertEquals(5, drama.size)
        assertEquals("c0p1q0", drama[1].quoteId)
        // "He froze." is found in both modes from the same saved position.
        val pos = BookPosition(0, 1, plain[2].start)
        assertEquals("He froze.", plain[book.utteranceAt(pos, false)].text)
        assertEquals("He froze.", drama[book.utteranceAt(pos, true)].text)
        assertTrue(book.progressOf(pos) in 0.3f..0.8f)
    }
}

class EpubParserTest {
    private fun epub(dir: File): File {
        val file = File(dir, "test.epub")
        ZipOutputStream(file.outputStream()).use { zip ->
            fun put(name: String, text: String) {
                zip.putNextEntry(ZipEntry(name))
                zip.write(text.toByteArray())
                zip.closeEntry()
            }
            put("mimetype", "application/epub+zip")
            put("META-INF/container.xml", """<?xml version="1.0"?><container><rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>""")
            put("OEBPS/content.opf", """<?xml version="1.0"?>
                <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
                <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                  <dc:title>The Test Book</dc:title><dc:creator>A. Writer</dc:creator><dc:language>en-GB</dc:language>
                </metadata>
                <manifest>
                  <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                  <item id="cover" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
                  <item id="c1" href="text/ch%201.xhtml" media-type="application/xhtml+xml"/>
                  <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
                  <item id="empty" href="text/blank.xhtml" media-type="application/xhtml+xml"/>
                </manifest>
                <spine><itemref idref="empty"/><itemref idref="c1"/><itemref idref="c2"/></spine></package>""")
            put("OEBPS/nav.xhtml", """<html><body><nav epub:type="toc"><ol>
                <li><a href="text/ch%201.xhtml#start">The Beginning</a></li>
                <li><a href="text/ch2.xhtml">The End</a></li></ol></nav></body></html>""")
            put("OEBPS/images/cover.jpg", "JPEGDATA")
            put("OEBPS/text/blank.xhtml", "<html><body><div><img src='x.png'/></div></body></html>")
            put("OEBPS/text/ch 1.xhtml", """<html><head><style>p{}</style></head><body>
                <h1>Chapter 1</h1>
                <p>It was a <i>dark</i> and stormy&nbsp;night.<sup><a href="#n1">1</a></sup></p>
                <p>Soft${"\u00ad"}hyphen here.<br/>New line.</p>
                <div>Loose text <p>inner para</p> tail</div>
                </body></html>""")
            put("OEBPS/text/ch2.xhtml", "<html><body><p>\u201cStop!\u201d she shouted.</p><p>   </p></body></html>")
        }
        return file
    }

    @Test
    fun parsesMetadataTocCoverAndParagraphs() {
        val dir = createTempDir()
        try {
            val book = EpubParser.parse(epub(dir))
            assertEquals("The Test Book", book.title)
            assertEquals("A. Writer", book.author)
            assertEquals("en-GB", book.language)
            assertEquals("JPEGDATA", book.cover?.toString(Charsets.UTF_8))
            assertEquals(listOf("The Beginning", "The End"), book.chapters.map { it.title })
            assertEquals(
                listOf("Chapter 1", "It was a dark and stormy night.", "Softhyphen here. New line.", "Loose text", "inner para", "tail"),
                book.chapters[0].paragraphs.map { it.text },
            )
            assertTrue(book.chapters[0].paragraphs[0].heading)
            assertEquals(listOf("\u201cStop!\u201d she shouted."), book.chapters[1].paragraphs.map { it.text })
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun resolvesRelativeHrefs() {
        assertEquals("OEBPS/text/ch 1.xhtml", EpubParser.resolve("OEBPS/nav.xhtml", "text/ch%201.xhtml#x"))
        assertEquals("OEBPS/images/a.jpg", EpubParser.resolve("OEBPS/text/c.xhtml", "../images/a.jpg"))
        assertEquals("c.xhtml", EpubParser.resolve("content.opf", "c.xhtml"))
    }
}

class BookScriptTest {
    @Test
    fun narratorFollowsSectionsInReadingOrder() {
        val script = com.androidreader.drama.BookScript.fromJson(org.json.JSONObject(
            """{"status":"done","characters":[],"lines":{},"narration":[[0,0,"sasaki"],[1,90,"neighbor"],[2,0,null]]}""",
        ))
        assertEquals("sasaki", script.narratorAt(0, 5))
        assertEquals("sasaki", script.narratorAt(1, 89))
        assertEquals("neighbor", script.narratorAt(1, 90))
        assertNull(script.narratorAt(2, 3))
        assertNull(com.androidreader.drama.BookScript().narratorAt(0, 0))
    }
}
