"""Age a voice description up or down one step, for "Younger" / "Older" in the voice editor.

The voice-design model works from the description, so a younger or older version of a voice is the same
description with its age moved one step along child, teen, young adult, adult, middle-aged, elderly:
"a young woman's warm voice" -> "a teenage girl's warm voice". The age the description is about is its first
clear mention ("a young girl's voice, old-fashioned like a grandmother" is a girl's); other ages it names in
passing are left alone. Numbers ("about ten years old", "in his twenties") follow the new age. A description
without an age gets a "sounds younger/older" sentence instead.
"""
from __future__ import annotations

import re

RUNGS = ("child", "teen", "young adult", "adult", "middle-aged", "elderly")
# How each age is written: (female noun, male noun, neutral noun, adjective).
FORMS = {
    "child": ("little girl", "little boy", "child", "child"),
    "teen": ("teenage girl", "teenage boy", "teenager", "teenage"),
    "young adult": ("young woman", "young man", "young adult", "young adult"),
    "adult": ("adult woman", "adult man", "adult", "adult"),
    "middle-aged": ("middle-aged woman", "middle-aged man", "middle-aged person", "middle-aged"),
    "elderly": ("elderly woman", "elderly man", "elderly person", "elderly"),
}
YEARS = {"child": 9, "teen": 15, "young adult": 24, "adult": 35, "middle-aged": 50, "elderly": 72}
DECADES = {"teen": "teens", "young adult": "twenties", "adult": "thirties", "middle-aged": "fifties",
           "elderly": "seventies"}

_N = r"(?:girl|boy|woman|man|lady|gentleman|person|kid|child)"
# (age, pattern, kind): "noun" phrases name the person, "adj" ones describe the voice; "weak" ones only count when
# nothing clearer is there ("youthful", "mature").
_PATTERNS = [
    ("child", rf"\b(?:little|small|young)\s+(?:girl|boy|kid|child)\b", "noun"),
    ("child", r"\b(?:child|kid)\b(?=\s+of\b)", "noun"),
    ("child", r"\b(?:girl|boy)\b", "bare"),  # a girl can be ten or sixteen: a stated number wins
    ("child", r"\b(?:child|kid)\b(?!\s*hood)", "adj"),
    ("child", r"\bchildlike\b", "weak"),
    ("teen", rf"\bteen(?:age|aged)?\s+{_N}\b", "noun"),
    ("teen", r"\bteenager\b", "noun"),
    ("teen", r"\b(?:teen(?:age|aged)?|adolescent)\b", "adj"),
    ("young adult", rf"\byoung\s+(?:woman|man|lady|gentleman|adult|person)\b", "noun"),
    ("young adult", r"\byoung[- ]adult\b", "adj"),
    ("young adult", r"\byouthful\b", "weak"),
    ("adult", rf"\badult\s+{_N}\b", "noun"),
    ("adult", r"\badult\b", "adj"),
    ("adult", r"\bmature\b", "weak"),
    ("middle-aged", rf"\bmiddle[- ]aged\s+{_N}\b", "noun"),
    ("middle-aged", r"\bmiddle[- ]aged\b", "adj"),
    ("elderly", rf"\b(?:elderly|old|aged)\s+(?:woman|man|lady|gentleman|person)\b", "noun"),
    ("elderly", r"\b(?:elderly|senior)\b", "adj"),
]
_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})
_NUM = r"(\d{1,3}|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")"
_YEARS_OLD = re.compile(_NUM + r"(\s*-?\s*years?\s*-?\s*old)", re.IGNORECASE)
_OF_AGE = re.compile(r"(\bof\s+(?:about|around|roughly|maybe)\s+)" + _NUM + r"\b", re.IGNORECASE)
_DECADE = re.compile(r"\bin\s+(his|her|their)\s+(?:early\s+|mid\s+|late\s+)?"
                     r"(teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b", re.IGNORECASE)
_DECADE_AGE = {"teens": "teen", "twenties": "young adult", "thirties": "adult", "forties": "middle-aged",
               "fifties": "middle-aged", "sixties": "elderly", "seventies": "elderly", "eighties": "elderly",
               "nineties": "elderly"}


def _rung_of_years(n: int) -> str:
    return ("child" if n < 13 else "teen" if n < 18 else "young adult" if n < 30 else "adult" if n < 45
            else "middle-aged" if n < 65 else "elderly")


def _gender(text: str) -> int:
    t = text.lower()
    if re.search(r"\b(?:female|woman|women|girl|lady|she|her|grandmother|mother)\b", t):
        return 0
    if re.search(r"\b(?:male|man|men|boy|gentleman|he|his|grandfather|father)\b", t):
        return 1
    return 2


def _mentions(text: str) -> list[tuple[int, int, str, str]]:
    """Non-overlapping age mentions (start, end, age, kind), clearest first where they overlap."""
    found = []
    for rung, pattern, kind in _PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append((m.start(), m.end(), rung, kind))
    found.sort(key=lambda f: (f[0], -(f[1] - f[0])))
    out: list[tuple[int, int, str, str]] = []
    for f in found:
        if not out or f[0] >= out[-1][1]:
            out.append(f)
    return out


def _stated_years(text: str) -> int | None:
    m = _YEARS_OLD.search(text)
    raw = m.group(1) if m else (_OF_AGE.search(text).group(2) if _OF_AGE.search(text) else None)
    if raw is None:
        return None
    raw = raw.lower()
    return int(raw) if raw.isdigit() else _WORDS[raw]


def current_age(text: str) -> str | None:
    """The age a description is about: a stated number of years, else its first clear mention, else a
    decade ("in his twenties"), else a weak hint ("youthful")."""
    years = _stated_years(text)
    if years is not None:
        return _rung_of_years(years)
    mentions = _mentions(text)
    clear = [m for m in mentions if m[3] != "weak"]
    if clear:
        return clear[0][2]
    m = _DECADE.search(text)
    if m:
        return _DECADE_AGE[m.group(2).lower()]
    return mentions[0][2] if mentions else None


def _cap(new: str, old: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def shift_age(text: str, step: int) -> dict:
    """Moves the description `step` age groups (negative: younger). Returns {"description", "from", "to"};
    "from" is None when the description names no age. Raises ValueError past the youngest or oldest."""
    text = (text or "").strip()
    if not step:
        return {"description": text, "from": current_age(text), "to": current_age(text)}
    now = current_age(text)
    if now is None:
        extra = ("The voice sounds noticeably younger." if step < 0 else "The voice sounds noticeably older.")
        return {"description": f"{text.rstrip('.')}. {extra}" if text else extra, "from": None, "to": None}
    i = RUNGS.index(now) + step
    if i < 0:
        raise ValueError("This voice is already a child's; it can't be made younger")
    if i >= len(RUNGS):
        raise ValueError("This voice is already elderly; it can't be made older")
    target = RUNGS[i]
    gender = _gender(text)
    female, male, neutral, adj = FORMS[target]
    noun = (female, male, neutral)[gender]

    def replace_mention(m_text: str, kind: str, after: str) -> str:
        possessive = after[:2] in ("'s", "\u2019s")
        # A noun phrase stays a noun phrase ("young woman" -> "teenage girl"); "a child's voice" -> "a teenager's".
        if kind in ("noun", "bare"):
            word = noun
        elif possessive:
            word = neutral if gender == 2 or target in ("child", "teen") else noun
            word = {"child": "child", "teen": "teenager"}.get(target, word)
        else:
            word = adj
        return _cap(word, m_text)

    mentions = _mentions(text)
    has_clear = any(k != "weak" for _, _, r, k in mentions if r == now)
    by_number = _stated_years(text) is not None
    out, last = [], 0
    for s, e, rung, kind in mentions:
        # With a stated age, a bare "girl"/"boy" is that age's too ("a 16-year-old girl" -> "a 24-year-old young woman").
        if not (rung == now or (by_number and kind == "bare")) or (kind == "weak" and has_clear):
            continue
        before = text[last:s]
        word = replace_mention(text[s:e], kind, text[e:e + 2])
        # "an adult" -> "a teenage", "a young" -> "an elderly": only the article right before the change.
        before = re.sub(r"\b([Aa]n?)(\s+)$", lambda m: _article(m.group(1), word) + m.group(2), before)
        out += [before, word]
        last = e
    out.append(text[last:])
    result = "".join(out)
    result = _YEARS_OLD.sub(lambda m: f"{YEARS[target]}{m.group(2)}", result)
    result = _OF_AGE.sub(lambda m: f"{m.group(1)}{YEARS[target]}", result)
    if target in DECADES:
        result = _DECADE.sub(lambda m: f"in {m.group(1)} {DECADES[target]}", result)
    else:
        result = re.sub(r"\s*,?\s*" + _DECADE.pattern, "", result, flags=re.IGNORECASE)
    return {"description": result, "from": now, "to": target}


def _article(old: str, word: str) -> str:
    art = "an" if word[:1].lower() in "aeiou" else "a"
    return art.capitalize() if old[:1].isupper() else art
