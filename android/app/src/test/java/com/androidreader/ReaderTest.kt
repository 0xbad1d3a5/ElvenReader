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
import org.junit.Assert.assertNotEquals
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

    @Test
    fun quotedTermsStayInTheNarration() {
        val text = "I\u2019d heard them say something about \u201cink,\u201d but I hadn\u2019t realized. \u201cStop!\u201d he said."
        val segs = Quotes.segments(text, Sentences.split(text), splitQuotes = true)
        assertEquals(
            listOf("I\u2019d heard them say something about \u201cink,\u201d but I hadn\u2019t realized.", "\u201cStop!\u201d", "he said."),
            segs.map { text.substring(it.start, it.end) },
        )
        assertNull(segs[0].quote)
        assertEquals(2, Quotes.regions(text).size) // quote ids are unchanged
        val old = "Alice considered a little, and then said \u201cThe fourth.\u201d"
        assertEquals(false, Quotes.isMention(old, old.indexOf('\u201c'), old.length))
        val bill = "then, saying to herself \u201cThis is Bill,\u201d she gave"
        assertEquals(false, Quotes.isMention(bill, bill.indexOf('\u201c'), bill.indexOf('\u201d') + 1))
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

    @Test
    fun filesMissingFromTheTocContinueThePreviousChapter() {
        val dir = createTempDir()
        try {
            val file = File(dir, "split.epub")
            ZipOutputStream(file.outputStream()).use { zip ->
                fun put(name: String, text: String) {
                    zip.putNextEntry(ZipEntry(name)); zip.write(text.toByteArray()); zip.closeEntry()
                }
                put("META-INF/container.xml", """<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>""")
                put("content.opf", """<package><metadata><dc:title>Split</dc:title></metadata><manifest>
                    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                    <item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>
                    <item id="a1" href="a-01.xhtml" media-type="application/xhtml+xml"/>
                    <item id="b" href="b.xhtml" media-type="application/xhtml+xml"/></manifest>
                    <spine><itemref idref="a"/><itemref idref="a1"/><itemref idref="b"/></spine></package>""")
                put("nav.xhtml", """<html><body><nav epub:type="toc"><ol><li><a href="a.xhtml">Part One</a></li>
                    <li><a href="b.xhtml">Part Two</a></li></ol></nav></body></html>""")
                put("a.xhtml", "<html><body><p>One.</p></body></html>")
                // after an illustration, the chapter continues in another file with a mid-chapter heading
                put("a-01.xhtml", "<html><body><p>Still one.</p><h3>(The Neighbor's POV)</h3><p>Hers.</p></body></html>")
                put("b.xhtml", "<html><body><p>Two.</p></body></html>")
            }
            val book = EpubParser.parse(file)
            assertEquals(listOf("Part One", "Part Two"), book.chapters.map { it.title })
            assertEquals(listOf("One.", "Still one.", "(The Neighbor's POV)", "Hers."), book.chapters[0].paragraphs.map { it.text })
            // a position saved with the old one-chapter-per-file split lands on the same paragraph
            val old = listOf(Chapter("Part One", book.chapters[0].paragraphs.take(1)),
                Chapter("(The Neighbor's POV)", book.chapters[0].paragraphs.drop(1)), book.chapters[1])
            assertEquals(0 to 3, EpubParser.remap(old, book.chapters, 1, 2))
            assertEquals(1 to 0, EpubParser.remap(old, book.chapters, 2, 0))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun picturesAttachToTheNextParagraphWithoutChangingTheText() {
        val dir = createTempDir()
        try {
            val file = File(dir, "pics.epub")
            ZipOutputStream(file.outputStream()).use { zip ->
                fun put(name: String, text: String) {
                    zip.putNextEntry(ZipEntry(name)); zip.write(text.toByteArray()); zip.closeEntry()
                }
                put("META-INF/container.xml", """<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>""")
                put("OEBPS/content.opf", """<package><metadata><dc:title>Pics</dc:title></metadata><manifest>
                    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                    <item id="i0" href="text/insert0.xhtml" media-type="application/xhtml+xml"/>
                    <item id="a" href="text/a.xhtml" media-type="application/xhtml+xml"/>
                    <item id="i1" href="text/insert1.xhtml" media-type="application/xhtml+xml"/>
                    <item id="b" href="text/b.xhtml" media-type="application/xhtml+xml"/></manifest>
                    <spine><itemref idref="i0"/><itemref idref="a"/><itemref idref="i1"/><itemref idref="b"/></spine></package>""")
                put("OEBPS/nav.xhtml", """<html><body><nav epub:type="toc"><ol><li><a href="text/a.xhtml">One</a></li>
                    <li><a href="text/b.xhtml">Two</a></li></ol></nav></body></html>""")
                // a colour page before the first chapter, an <svg><image> insert page at a chapter's end
                put("OEBPS/text/insert0.xhtml", """<html><body><div><img src="../images/color.jpg"/></div></body></html>""")
                put("OEBPS/text/a.xhtml", """<html><body><p>First.</p><p class="pic"><img src="../images/mid.png" alt=""/></p>
                    <p>Second.</p></body></html>""")
                put("OEBPS/text/insert1.xhtml", """<html><body><svg xmlns:xlink="http://www.w3.org/1999/xlink">
                    <image xlink:href="../images/end.jpg"/></svg></body></html>""")
                put("OEBPS/text/b.xhtml", "<html><body><p>Third.</p></body></html>")
            }
            val book = EpubParser.parse(file)
            assertEquals(listOf("One", "Two"), book.chapters.map { it.title })
            assertEquals(listOf("First.", "Second."), book.chapters[0].paragraphs.map { it.text })
            assertEquals(listOf("OEBPS/images/color.jpg"), book.chapters[0].paragraphs[0].images)
            assertEquals(listOf("OEBPS/images/mid.png"), book.chapters[0].paragraphs[1].images)
            assertEquals(listOf("OEBPS/images/end.jpg"), book.chapters[0].endImages)
            assertEquals(emptyList<String>(), book.chapters[1].paragraphs[0].images)
        } finally {
            dir.deleteRecursively()
        }
    }
}

class BookScriptTest {
    @Test
    fun descriptiveNamesAreShownCapitalised() {
        val d = { n: String -> com.androidreader.drama.displayName(n) }
        assertEquals("Prisoner", d("a prisoner"))
        assertEquals("Tattoo Artist", d("the tattoo artist"))
        assertEquals("Man in the Hat", d("man in the hat"))
        assertEquals("The Neighbor", d("the Neighbor"))
        assertEquals("Type Twelve", d("Type Twelve"))
        assertEquals("Mr. Akutsu", d("Mr. Akutsu"))
    }

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

    @Test
    fun editedVoicesMakeTheScriptDiffer() {
        fun parse(version: String) = com.androidreader.drama.BookScript.fromJson(org.json.JSONObject(
            """{"status":"done","characters":[],"lines":{},"voice_versions":{"v1":"$version"}}""",
        ))
        assertEquals(mapOf("v1" to "aaa"), parse("aaa").voiceVersions)
        assertEquals(parse("aaa"), parse("aaa"))
        assertNotEquals(parse("aaa"), parse("bbb"))  // the reader drops prepared audio when this changes
    }
}
