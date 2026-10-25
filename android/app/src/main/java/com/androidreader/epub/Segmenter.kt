package com.androidreader.epub

/** Half-open character range [start, end) within a paragraph. */
data class TextSpan(val start: Int, val end: Int) {
    val length: Int get() = end - start
    operator fun contains(offset: Int) = offset in start until end
}

/**
 * A piece of a paragraph spoken as one utterance. [quote] is the start offset of
 * the quotation it belongs to (all pieces of one quotation share it), or null for
 * narration.
 */
data class Segment(val start: Int, val end: Int, val quote: Int? = null)

/** Rule-based sentence splitting that never changes the text (offsets map back to the EPUB). */
object Sentences {
    private const val ENDERS = ".!?\u2026"
    private const val CLOSERS = "\"'\u201d\u2019\u00bb)]}\u300d\u300f"
    private val ABBREVIATIONS = setOf(
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "mt", "no", "vol", "ch", "fig",
        "gen", "col", "capt", "lt", "sgt", "rev", "hon", "messrs", "mme", "mlle", "co", "corp", "inc", "ltd",
        "approx", "dept", "est", "p", "pp", "cf", "ca", "op", "ed", "eds", "jan", "feb", "mar", "apr", "jun",
        "jul", "aug", "sep", "sept", "oct", "nov", "dec", "ave", "blvd", "rd", "esq", "maj", "adm", "cmdr", "gov",
        "pres", "sen", "rep", "supt", "fr", "bros", "mx",
    )
    const val MAX_CHARS = 360

    fun split(text: String): List<TextSpan> {
        val spans = mutableListOf<TextSpan>()
        val n = text.length
        var start = skipSpace(text, 0)
        var i = start
        while (i < n) {
            if (text[i] !in ENDERS) {
                i++
                continue
            }
            var j = i + 1
            while (j < n && text[j] in ENDERS) j++
            while (j < n && text[j] in CLOSERS) j++
            if (j >= n) break
            if (!text[j].isWhitespace()) {
                i = j
                continue
            }
            val k = skipSpace(text, j)
            if (k >= n) break
            val next = text[k]
            val boundary = when {
                next.isLowerCase() -> false
                text[i] == '.' && j == i + 1 && isAbbreviation(text, i) -> false
                else -> true
            }
            if (boundary) {
                spans += TextSpan(start, j)
                start = k
            }
            i = k
        }
        if (start < n) spans += TextSpan(start, trimEnd(text, n))
        return spans.filter { it.length > 0 }.flatMap { splitLong(text, it) }
    }

    private fun skipSpace(text: String, from: Int): Int {
        var k = from
        while (k < text.length && text[k].isWhitespace()) k++
        return k
    }

    private fun trimEnd(text: String, end: Int): Int {
        var e = end
        while (e > 0 && text[e - 1].isWhitespace()) e--
        return e
    }

    private fun isAbbreviation(text: String, dot: Int): Boolean {
        var s = dot
        while (s > 0 && (text[s - 1].isLetter() || text[s - 1] == '.')) s--
        val word = text.substring(s, dot)
        if (word.isEmpty()) return false
        if (word.length == 1 && word[0].isUpperCase()) return true // initials: "J. K. Rowling"
        if (word.contains('.')) return true // "e.g", "U.S"
        return word.lowercase() in ABBREVIATIONS
    }

    /** Over-long sentences are split at ; : or dashes, then at commas, so each stays a natural phrase. */
    private fun splitLong(text: String, span: TextSpan): List<TextSpan> {
        if (span.length <= MAX_CHARS) return listOf(span)
        for (marks in listOf(";:\u2014", ",")) {
            val cuts = mutableListOf<Int>()
            var last = span.start
            for (p in span.start until span.end - 1) {
                if (text[p] in marks && text[p + 1].isWhitespace() && p + 1 - last >= 40) {
                    cuts += p + 1
                    last = p + 1
                }
            }
            if (cuts.isEmpty()) continue
            val out = mutableListOf<TextSpan>()
            var s = span.start
            for (c in cuts) {
                if (c - s >= 40 && span.end - c >= 20) {
                    out += TextSpan(s, c)
                    s = skipSpace(text, c)
                }
            }
            out += TextSpan(s, span.end)
            if (out.size > 1) return out.flatMap { if (it.length > MAX_CHARS && marks == ",") listOf(it) else splitLong(text, it) }
        }
        return listOf(span)
    }
}

/** Finds quotations and cuts sentences into narration and speech segments. */
object Quotes {
    private val PAIRS = mapOf('\u201c' to '\u201d', '\u00ab' to '\u00bb', '\u201e' to '\u201c', '\u2018' to '\u2019', '"' to '"')
    private const val TRAILING_PUNCT = ",;:.!?\u2026\u2014\u2013"

    /** Quotation ranges, including the quote marks. An unclosed quote runs to the paragraph end. */
    fun regions(text: String): List<TextSpan> {
        val out = mutableListOf<TextSpan>()
        var open = -1
        var closer = ' '
        var i = 0
        while (i < text.length) {
            val c = text[i]
            if (open < 0) {
                val close = PAIRS[c]
                if (close != null && opensHere(text, i, c)) {
                    open = i
                    closer = close
                }
            } else if (c == closer && closesHere(text, i, c)) {
                out += TextSpan(open, i + 1)
                open = -1
            }
            i++
        }
        if (open >= 0) out += TextSpan(open, text.length)
        return out
    }

    private fun opensHere(text: String, i: Int, c: Char): Boolean {
        if (c != '\u2018' && c != '"') return true
        // ' and " also appear as apostrophes or inch marks: require a word start.
        val prev = if (i > 0) text[i - 1] else ' '
        val next = if (i + 1 < text.length) text[i + 1] else ' '
        return (prev.isWhitespace() || prev in "([\u2014-") && !next.isWhitespace()
    }

    private fun closesHere(text: String, i: Int, c: Char): Boolean {
        if (c != '\u2019' && c != '"') return true
        val next = if (i + 1 < text.length) text[i + 1] else ' '
        return !next.isLetter() // "don’t" keeps the quote open
    }

    /**
     * Utterances for a paragraph. Plain mode: one per sentence. Dramatized mode:
     * sentences are further cut where quotations start and end, so "“Stop!” she
     * shouted." becomes the quote “Stop!” and the narration "she shouted.".
     */
    fun segments(text: String, sentences: List<TextSpan>, splitQuotes: Boolean): List<Segment> {
        if (!splitQuotes) return sentences.map { Segment(it.start, it.end) }
        val regions = regions(text)
        val out = mutableListOf<Segment>()
        for (s in sentences) {
            val cuts = sortedSetOf(s.start, s.end)
            for (r in regions) {
                if (r.start in s.start + 1 until s.end) cuts += r.start
                if (r.end in s.start + 1 until s.end) cuts += r.end
            }
            val bounds = cuts.toList()
            val pieces = mutableListOf<Segment>()
            for (k in 0 until bounds.size - 1) {
                var a = bounds[k]
                var b = bounds[k + 1]
                while (a < b && text[a].isWhitespace()) a++
                // Punctuation right after a closing quote (", he said") belongs to the quote.
                if (pieces.isNotEmpty() && pieces.last().end == bounds[k]) {
                    var j = a
                    while (j < b && text[j] in TRAILING_PUNCT) j++
                    if (j > a) {
                        pieces[pieces.size - 1] = pieces.last().copy(end = j)
                        a = j
                        while (a < b && text[a].isWhitespace()) a++
                    }
                }
                while (b > a && text[b - 1].isWhitespace()) b--
                if (a >= b) continue
                val quote = regions.firstOrNull { a >= it.start && a < it.end }?.start
                pieces += Segment(a, b, quote)
            }
            // Punctuation-only pieces (a stray comma after a quote) join their neighbour.
            for (p in pieces) {
                val speakable = text.substring(p.start, p.end).any { it.isLetterOrDigit() }
                if (!speakable && out.isNotEmpty() && out.last().end <= p.start) {
                    out[out.size - 1] = out.last().copy(end = p.end)
                } else if (speakable) {
                    out += p
                }
            }
        }
        return out
    }
}
