package com.androidreader.epub

import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import org.jsoup.nodes.Element
import org.jsoup.nodes.Node
import org.jsoup.nodes.TextNode
import org.jsoup.parser.Parser
import java.io.File
import java.util.zip.ZipFile

/** A paragraph of text; [images] are the pictures (zip paths in the EPUB) shown just before it. */
data class Paragraph(val text: String, val heading: Boolean = false, val images: List<String> = emptyList())

/** [endImages]: pictures after the chapter's last paragraph. */
data class Chapter(val title: String, val paragraphs: List<Paragraph>, val endImages: List<String> = emptyList())

/** One XHTML file: its paragraphs (pictures attached to the paragraph after them) and pictures after the last one. */
class ParsedDocument(val paragraphs: List<Paragraph>, val trailingImages: List<String>)

data class EpubBook(
    val title: String,
    val author: String,
    val language: String?,
    val cover: ByteArray?,
    val chapters: List<Chapter>,
)

class EpubException(message: String) : Exception(message)

/** Reads an EPUB 2/3 file into plain paragraphs per chapter of its table of contents. */
object EpubParser {
    /** Bumped when parsing changes how a book is split; books parsed earlier are re-parsed. */
    const val VERSION = 3 // 3: pictures (text and paragraph numbering unchanged from 2)

    /**
     * Where [position] (in a book split as [old]) is in the same text split as [new]: the books have
     * the same paragraphs in the same order, only the chapter boundaries differ.
     */
    fun remap(old: List<Chapter>, new: List<Chapter>, chapter: Int, paragraph: Int): Pair<Int, Int> {
        var flat = 0
        for (c in 0 until chapter.coerceAtMost(old.size)) flat += old[c].paragraphs.size
        flat += paragraph
        for ((c, ch) in new.withIndex()) {
            if (flat < ch.paragraphs.size) return c to flat
            flat -= ch.paragraphs.size
        }
        return (new.size - 1).coerceAtLeast(0) to 0
    }

    private val BLOCKS = setOf(
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dt", "dd",
        "section", "article", "header", "footer", "aside", "figure", "figcaption", "table", "tr", "td", "th",
        "ul", "ol", "dl", "hr", "body", "main", "center", "address",
    )
    private val SKIP = setOf("script", "style", "nav", "head", "svg", "math", "rt", "rp", "template", "noscript")
    private val HEADINGS = setOf("h1", "h2", "h3", "h4", "h5", "h6")
    private val WHITESPACE = Regex("[\\s\u00a0\u2007\u202f]+")

    fun parse(file: File): EpubBook = ZipFile(file).use { zip ->
        fun read(path: String): ByteArray? = zip.getEntry(path)?.let { e -> zip.getInputStream(e).use { it.readBytes() } }
        fun readText(path: String): String? = read(path)?.toString(Charsets.UTF_8)

        val container = readText("META-INF/container.xml") ?: throw EpubException("Not an EPUB: no container.xml")
        val opfPath = Jsoup.parse(container, "", Parser.xmlParser()).selectFirst("rootfile")?.attr("full-path")
            ?.takeIf { it.isNotBlank() } ?: throw EpubException("EPUB has no package document")
        val opf = Jsoup.parse(readText(opfPath) ?: throw EpubException("Missing $opfPath"), "", Parser.xmlParser())

        val title = meta(opf, "title") ?: file.nameWithoutExtension
        val author = meta(opf, "creator").orEmpty()
        val language = meta(opf, "language")

        data class Item(val href: String, val type: String, val props: String)
        val manifest = opf.select("manifest > item").associate { el ->
            el.attr("id") to Item(resolve(opfPath, el.attr("href")), el.attr("media-type"), el.attr("properties"))
        }

        val cover = findCover(opf, manifest.mapValues { it.value.href to (it.value.type to it.value.props) })?.let(::read)

        val toc = mutableMapOf<String, String>()
        manifest.values.firstOrNull { "nav" in it.props.split(' ') }?.let { nav ->
            readText(nav.href)?.let { toc.putAll(navTitles(nav.href, it)) }
        }
        if (toc.isEmpty()) {
            val ncxId = opf.selectFirst("spine")?.attr("toc")
            val ncx = manifest[ncxId] ?: manifest.values.firstOrNull { it.type == "application/x-dtbncx+xml" }
            ncx?.let { item -> readText(item.href)?.let { toc.putAll(ncxTitles(item.href, it)) } }
        }

        val chapters = mutableListOf<Chapter>()
        // Publishers often split one chapter into several files (at every illustration). Only files the
        // table of contents points to start a chapter; the others continue the one before them.
        val tocStarts = toc.keys.toSet()
        val groupByToc = tocStarts.size >= 2
        // Pictures waiting for the next paragraph; true = they belong to the chapter before it (shown at its
        // end if a new chapter starts first). Picture-only pages never become chapters, so numbering is stable.
        val carry = mutableListOf<Pair<String, Boolean>>()
        fun withImages(paras: List<Paragraph>, images: List<String>) =
            if (images.isEmpty()) paras else listOf(paras[0].copy(images = images + paras[0].images)) + paras.drop(1)
        for (ref in opf.select("spine > itemref")) {
            if (ref.attr("linear") == "no") continue
            val item = manifest[ref.attr("idref")] ?: continue
            if (!item.type.contains("html")) continue
            val html = readText(item.href) ?: continue
            val doc = extractDocument(html, item.href)
            val continues = groupByToc && item.href !in tocStarts && chapters.isNotEmpty()
            if (doc.paragraphs.isEmpty()) {
                carry += doc.trailingImages.map { it to continues }
                continue
            }
            if (continues) {
                val last = chapters.removeAt(chapters.size - 1)
                chapters += last.copy(paragraphs = last.paragraphs + withImages(doc.paragraphs, carry.map { it.first }))
            } else {
                val before = carry.filter { it.second }.map { it.first }
                if (before.isNotEmpty()) chapters[chapters.size - 1] = chapters.last().let { it.copy(endImages = it.endImages + before) }
                val chapterTitle = toc[item.href]
                    ?: doc.paragraphs.firstOrNull { it.heading }?.text?.take(80)
                    ?: "Section ${chapters.size + 1}"
                chapters += Chapter(chapterTitle, withImages(doc.paragraphs, carry.filterNot { it.second }.map { it.first }))
            }
            carry.clear()
            carry += doc.trailingImages.map { it to true }
        }
        if (chapters.isEmpty()) throw EpubException("No readable text in this EPUB")
        if (carry.isNotEmpty()) chapters[chapters.size - 1] = chapters.last().let { it.copy(endImages = it.endImages + carry.map { c -> c.first }) }
        EpubBook(title.trim(), author.trim(), language?.trim(), cover, chapters)
    }

    private fun meta(opf: Document, name: String): String? {
        val metadata = opf.selectFirst("metadata") ?: opf
        return metadata.allElements.firstOrNull { el ->
            val tag = el.tagName().lowercase()
            tag == "dc:$name" || tag.endsWith(":$name") || tag == name
        }?.text()?.takeIf { it.isNotBlank() }
    }

    private fun findCover(opf: Document, manifest: Map<String, Pair<String, Pair<String, String>>>): String? {
        manifest.values.firstOrNull { "cover-image" in it.second.second.split(' ') }?.let { return it.first }
        val metaCover = opf.select("meta").firstOrNull { it.attr("name") == "cover" }?.attr("content")
        manifest[metaCover]?.takeIf { it.second.first.startsWith("image/") }?.let { return it.first }
        return manifest.entries.firstOrNull { (id, v) ->
            v.second.first.startsWith("image/") && (id.contains("cover", true) || v.first.contains("cover", true))
        }?.value?.first
    }

    /** Resolves [href] relative to the file at [base] (both zip paths). */
    fun resolve(base: String, href: String): String {
        val clean = href.substringBefore('#')
        if (clean.isEmpty()) return base
        val decoded = try {
            java.net.URLDecoder.decode(clean.replace("+", "%2B"), "UTF-8")
        } catch (_: Exception) {
            clean
        }
        val parts = ArrayList<String>()
        if (!decoded.startsWith("/")) base.substringBeforeLast('/', "").split('/').filter { it.isNotEmpty() }.forEach { parts += it }
        for (seg in decoded.split('/')) {
            when (seg) {
                "", "." -> {}
                ".." -> if (parts.isNotEmpty()) parts.removeAt(parts.size - 1)
                else -> parts += seg
            }
        }
        return parts.joinToString("/")
    }

    private fun navTitles(navPath: String, html: String): Map<String, String> {
        val doc = Jsoup.parse(html)
        val nav = doc.select("nav").firstOrNull { it.attr("epub:type").contains("toc") } ?: doc.selectFirst("nav") ?: return emptyMap()
        val out = linkedMapOf<String, String>()
        for (a in nav.select("a[href]")) {
            val target = resolve(navPath, a.attr("href"))
            val text = a.text().trim()
            if (text.isNotEmpty()) out.putIfAbsent(target, text)
        }
        return out
    }

    private fun ncxTitles(ncxPath: String, xml: String): Map<String, String> {
        val doc = Jsoup.parse(xml, "", Parser.xmlParser())
        val out = linkedMapOf<String, String>()
        for (point in doc.select("navPoint")) {
            val src = point.selectFirst("content")?.attr("src") ?: continue
            val text = point.selectFirst("navLabel")?.text()?.trim().orEmpty()
            if (text.isNotEmpty()) out.putIfAbsent(resolve(ncxPath, src), text)
        }
        return out
    }

    /** Block-level text of an XHTML document, one entry per paragraph or heading. */
    fun extractParagraphs(html: String): List<Paragraph> = extractDocument(html).paragraphs

    /** Paragraphs plus pictures (`<img>`, `<svg><image>`), resolved against [base], the file's zip path. */
    fun extractDocument(html: String, base: String = ""): ParsedDocument {
        val body = Jsoup.parse(html).body() ?: return ParsedDocument(emptyList(), emptyList())
        val walker = Walker(base)
        walker.walk(body)
        return ParsedDocument(walker.out, walker.pending)
    }

    private fun isBlock(el: Element) = el.tagName().lowercase() in BLOCKS

    private fun skip(el: Element): Boolean {
        val tag = el.tagName().lowercase()
        if (tag in SKIP) return true
        val type = el.attr("epub:type")
        if (type.contains("noteref") || type.contains("pagebreak")) return true
        if (el.attr("role") == "doc-noteref" || el.attr("role") == "doc-pagebreak") return true
        // Footnote markers such as <sup><a href="#n1">1</a></sup>.
        if (tag == "sup" && el.text().trim().let { t -> t.length <= 3 && t.none { it.isLetter() } }) return true
        return false
    }

    private class Walker(val base: String) {
        val out = mutableListOf<Paragraph>()
        val pending = mutableListOf<String>()

        private fun picture(el: Element) {
            val refs = when (el.tagName().lowercase()) {
                "img" -> listOf(el.attr("src"))
                "svg" -> el.select("image").map { it.attr("xlink:href").ifBlank { it.attr("href") } }
                else -> emptyList()
            }
            for (ref in refs) if (ref.isNotBlank() && !ref.startsWith("data:")) pending += if (base.isEmpty()) ref else resolve(base, ref)
        }

        fun walk(el: Element) {
            picture(el)
            if (skip(el)) return
            val hasBlockChild = el.children().any { isBlock(it) && !skip(it) }
            if (!hasBlockChild) {
                emit(inlineText(el), el.tagName().lowercase() in HEADINGS)
                return
            }
            // Mixed content: text runs between block children become their own paragraphs.
            val run = StringBuilder()
            for (node in el.childNodes()) {
                if (node is Element && isBlock(node)) {
                    emit(run.toString(), false)
                    run.clear()
                    walk(node)
                } else {
                    run.append(nodeText(node))
                }
            }
            emit(run.toString(), false)
        }

        private fun inlineText(el: Element): String {
            val sb = StringBuilder()
            for (node in el.childNodes()) sb.append(nodeText(node))
            return sb.toString()
        }

        private fun nodeText(node: Node): String = when (node) {
            is TextNode -> node.wholeText
            is Element -> {
                picture(node)
                when {
                    skip(node) -> ""
                    node.tagName().equals("br", true) -> " "
                    node.tagName().equals("img", true) -> ""
                    else -> inlineText(node)
                }
            }
            else -> ""
        }

        /** A text paragraph takes the pictures seen since the previous one; picture-only blocks emit nothing. */
        private fun emit(raw: String, heading: Boolean) {
            val text = clean(raw)
            if (text.any { it.isLetterOrDigit() }) {
                out += Paragraph(text, heading, pending.toList())
                pending.clear()
            }
        }
    }

    /** Collapses whitespace and removes soft hyphens / zero-width characters. */
    fun clean(raw: String): String =
        WHITESPACE.replace(raw.filterNot { it == '\u00ad' || it == '\u200b' || it == '\u200c' || it == '\u200d' || it == '\u2060' || it == '\ufeff' }, " ").trim()
}
