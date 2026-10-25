package com.androidreader.tts

/**
 * Lightweight, per-sentence guess of how quoted speech should be delivered.
 *
 * Used where there is no book-level analysis (the system TTS engine that ReadEra
 * calls, or a book that has not been dramatized). It only looks at the sentence
 * itself: a quote ending in "!" or a speech verb such as "whispered" next to a
 * quote. Narration without quotes always stays neutral.
 */
object DeliveryHints {
    private val QUOTE = Regex("[\"\u201c\u201d\u00ab\u00bb]([^\"\u201c\u201d\u00ab\u00bb]{1,400})[\"\u201c\u201d\u00ab\u00bb]")
    private val VERBS = listOf(
        Regex("\\b(whisper(ed|s|ing)?|murmur(ed|s)?|breathed)\\b", RegexOption.IGNORE_CASE) to "whispering",
        Regex("\\b(shout(ed|s|ing)?|yell(ed|s|ing)?|scream(ed|s|ing)?|bellow(ed|s)?|roar(ed|s)?)\\b", RegexOption.IGNORE_CASE) to "shouting",
        Regex("\\b(sob(bed|s|bing)?|wept|weep(s|ing)?)\\b", RegexOption.IGNORE_CASE) to "sad, crying",
        Regex("\\b(laugh(ed|s|ing)?|giggl(ed|es|ing)|chuckl(ed|es|ing))\\b", RegexOption.IGNORE_CASE) to "laughing",
        Regex("\\b(hiss(ed|es)?|snarl(ed|s)?|growl(ed|s)?)\\b", RegexOption.IGNORE_CASE) to "angry",
        Regex("\\b(stammer(ed|s)?|stutter(ed|s)?)\\b", RegexOption.IGNORE_CASE) to "nervous, hesitant",
    )

    fun guess(sentence: String): String? {
        val quotes = QUOTE.findAll(sentence).map { it.groupValues[1].trim() }.toList()
        if (quotes.isEmpty()) return null
        val narration = QUOTE.replace(sentence, " ")
        VERBS.firstOrNull { (re, _) -> re.containsMatchIn(narration) }?.let { return it.second }
        val spoken = quotes.joinToString(" ")
        val letters = spoken.filter { it.isLetter() }
        if (letters.length >= 3 && letters.all { it.isUpperCase() }) return "shouting"
        if (quotes.any { it.trimEnd('\u2014', '-', ' ').endsWith("!") }) {
            return if (spoken.split(' ').size <= 4) "shouting, urgent" else "excited, emphatic"
        }
        return null
    }
}
