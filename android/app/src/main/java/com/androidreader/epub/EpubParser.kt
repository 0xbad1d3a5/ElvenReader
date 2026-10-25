package com.androidreader.epub

import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import org.jsoup.nodes.Element
import org.jsoup.nodes.Node
import org.jsoup.nodes.TextNode
import org.jsoup.parser.Parser
import java.io.File
import java.util.zip.ZipFile

data class Paragraph(val text: String, val heading: Boolean = false)

data class Chapter(val title: String, val paragraphs: List<Paragraph>)

data class EpubBook(
    val title: String,
    val author: String,
    val language: String?,
    val cover: ByteArray?,
    val chapters: List<Chapter>,
)

class EpubException(message: String) : Exception(message)

/** Reads an EPUB 2/3 file into plain paragraphs per spine document. */
object EpubParser {
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
        for (ref in opf.select("spine > itemref")) {
            if (ref.attr("linear") == "no") continue
            val item = manifest[ref.attr("idref")] ?: continue
            if (!item.type.contains("html")) continue
            val html = readText(item.href) ?: continue
            val paragraphs = extractParagraphs(html)
            if (paragraphs.isEmpty()) continue
            val chapterTitle = toc[item.href]
                ?: paragraphs.firstOrNull { it.heading }?.text?.take(80)
                ?: "Section ${chapters.size + 1}"
            chapters += Chapter(chapterTitle, paragraphs)
        }
        if (chapters.isEmpty()) throw EpubException("No readable text in this EPUB")
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
    fun extractParagraphs(html: String): List<Paragraph> {
        val doc = Jsoup.parse(html)
        val body = doc.body() ?: return emptyList()
        val out = mutableListOf<Paragraph>()
        walk(body, out)
        return out
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

    private fun walk(el: Element, out: MutableList<Paragraph>) {
        if (skip(el)) return
        val hasBlockChild = el.children().any { isBlock(it) && !skip(it) }
        if (!hasBlockChild) {
            emit(inlineText(el), el.tagName().lowercase() in HEADINGS, out)
            return
        }
        // Mixed content: text runs between block children become their own paragraphs.
        val run = StringBuilder()
        for (node in el.childNodes()) {
            if (node is Element && isBlock(node)) {
                emit(run.toString(), false, out)
                run.clear()
                walk(node, out)
            } else {
                run.append(nodeText(node))
            }
        }
        emit(run.toString(), false, out)
    }

    private fun inlineText(el: Element): String {
        val sb = StringBuilder()
        for (node in el.childNodes()) sb.append(nodeText(node))
        return sb.toString()
    }

    private fun nodeText(node: Node): String = when (node) {
        is TextNode -> node.wholeText
        is Element -> when {
            skip(node) -> ""
            node.tagName().equals("br", true) -> " "
            node.tagName().equals("img", true) -> ""
            else -> inlineText(node)
        }
        else -> ""
    }

    private fun emit(raw: String, heading: Boolean, out: MutableList<Paragraph>) {
        val text = clean(raw)
        if (text.any { it.isLetterOrDigit() }) out += Paragraph(text, heading)
    }

    /** Collapses whitespace and removes soft hyphens / zero-width characters. */
    fun clean(raw: String): String =
        WHITESPACE.replace(raw.filterNot { it == '\u00ad' || it == '\u200b' || it == '\u200c' || it == '\u200d' || it == '\u2060' || it == '\ufeff' }, " ").trim()
}
