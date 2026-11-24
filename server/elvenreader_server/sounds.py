"""Sound marks: the non-verbal sounds written into a book's dialogue ("Ahaha, that's fine.", "Haaah... Really.",
"Kyaaah!", "Ngh... I'm fine.") marked in a line's spoken form ("say") so each speech engine can perform them its
own way. A mark is `{kind|written}`, the kind from KINDS and the sound as the book writes it:

    "{laugh|Ahaha}, that's fine."    "{sigh|Haaah}... Really."    "{scream|Kyaaah}! What's going on?!"

The page text never changes. The analysis (the LLM) names the written sounds of each quotation and their kind
(`mark` wraps them); an engine gets `render(text, its tags)`: a sound it can make becomes its own tag (VoxCPM2:
"[laugh-haha]"), any other is read as written. Braces never occur in the library's books (0 in 24); any that
do are turned into parentheses before the marks are read (`protect`), so a mark is never mistaken.

Kinds come from the library (47,304 quotations scanned: Ummm x135, Haaah x34, Huh?! x128, Ngh x31, Kyaaah, Hmph,
Eheheh, Shhh, creature noises) and the caption standards' sound vocabulary.
"""
from __future__ import annotations

import re

# kind -> what it covers (for the labelling prompt)
KINDS: dict[str, str] = {
    "laugh": "laughter or giggling written out (Ahaha, Hahaha, Ehehe, Heh-heh, Fufu, Kukuku)",
    "sigh": "a sigh or long breath out (Haaah, Hah..., Phew, Whew)",
    "gasp": "a sharp breath in, a hiccup (Hk!, Hic)",
    "um": "a hesitation or thinking noise (Um, Ummm, Uh, Er, Hmm, Mmm)",
    "hmph": "a displeased, scoffing or dismissive noise (Hmph, Tch, Tsk, Pfft)",
    "whoa": "a cry of surprise or wonder (Whoa, Wah, Hwah, Wooow, Oooh)",
    "huh": "a questioning or bewildered noise (Huh?, Eh?, Bwuh?, Wha?, Hm?)",
    "shh": "shushing (Shh, Shhh)",
    "sob": "crying or whimpering (Uuu, Waaah, Hic-hic)",
    "groan": "a groan or grunt of effort or pain (Ngh, Nngh, Ugh, Urk, Oof, Guh)",
    "cough": "a cough or throat-clearing (Ahem, Cough, Khh)",
    "sniff": "a sniff or sniffle (Sniff, Snf)",
    "yawn": "a yawn (Fwaaah, Yawn)",
    "scream": "a scream or shriek (Kyaaah, Gyaaah, Eek, Aaaaah!)",
    "yell": "a battle cry or shout of effort (Hyaaah, Raaah, Haaa!)",
    "growl": "an animal or creature noise (Grrr, Krrr, Kyuu, Woof, Quack)",
}
MARK = re.compile(r"\{([a-z]+)\|([^{}|]{1,60})\}")
_STRAY = str.maketrans("{}", "()")


def protect(text: str) -> str:
    """Braces that are not a sound mark become parentheses."""
    out, last = [], 0
    for m in MARK.finditer(text):
        if m.group(1) not in KINDS:
            continue
        out += [text[last:m.start()].translate(_STRAY), m.group(0)]
        last = m.end()
    return "".join(out) + text[last:].translate(_STRAY)


def mark(text: str, sounds) -> str:
    """`text` with the written sounds named ([{"written": "Ahaha", "sound": "laugh"}]) marked wherever they occur
    as whole words ("Gyaaah! Gyaaah!": both). Sounds not in the text, unknown kinds and text already inside a
    mark are left alone."""
    if not text or not isinstance(sounds, list):
        return text or ""
    for s in sounds:
        if not isinstance(s, dict) or s.get("sound") not in KINDS:
            continue
        written = " ".join(str(s.get("written") or "").split()).strip("\"'\u201c\u201d\u2018\u2019")
        written = written.rstrip(".,!?\u2026;:")  # just the sound: "Huh?!" -> "Huh"
        if not written or not any(ch.isalpha() for ch in written) or "{" in written or "}" in written:
            continue
        pat = re.compile(r"(?<![\w{|\-])" + re.escape(written) + r"(?![\w}|\-])", re.IGNORECASE)
        pieces, last = [], 0
        for m in MARK.finditer(text):  # never inside an existing mark
            pieces.append(pat.sub(lambda w: f"{{{s['sound']}|{w.group(0)}}}", text[last:m.start()]))
            pieces.append(m.group(0))
            last = m.end()
        pieces.append(pat.sub(lambda w: f"{{{s['sound']}|{w.group(0)}}}", text[last:]))
        text = "".join(pieces)
    return text


def plain(text: str) -> str:
    """The text with the marks taken out (the sounds as written)."""
    return MARK.sub(lambda m: m.group(2), text)


def kinds(text: str) -> list[str]:
    return [m.group(1) for m in MARK.finditer(text or "")]


_AFTER_TAG = re.compile(r"[\s,.!?\u2026;:\u2014\-]*")


def render(text: str, tags: dict[str, str] | None) -> str:
    """The text for an engine: each mark becomes the engine's tag for that kind (`tags`: kind -> its tag, e.g.
    {"laugh": "[laugh-haha]"}), with the punctuation right after the sound dropped and the next word
    capitalised ("{sigh|Haaah}... really?" -> "[sigh-hah] Really?"); a kind it has no tag for is read as
    written. When the line is nothing but sounds ("{scream|Kyaaah}!"), they are all read as written: a tag
    with no words to say is unreliable."""
    tags = tags or {}
    rest = MARK.sub("", text)
    if not any(ch.isalnum() for ch in rest):
        return plain(text)
    out, last = [], 0
    for m in MARK.finditer(text):
        tag = tags.get(m.group(1))
        out.append(text[last:m.start()])
        last = m.end()
        if not tag:
            out.append(m.group(2))
            continue
        out.append(tag + " ")
        starts = not text[:m.start()].strip() or text[:m.start()].rstrip()[-1] in ".!?\u2026\"\u201c"
        skip = _AFTER_TAG.match(text, last)
        last = skip.end()
        if starts and last < len(text):
            text = text[:last] + text[last].upper() + text[last + 1:]
    out.append(text[last:])
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def prompt_list() -> str:
    return "\n".join(f"  {k}: {v}" for k, v in KINDS.items())
