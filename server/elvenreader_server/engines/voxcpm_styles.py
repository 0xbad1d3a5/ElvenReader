"""Delivery slots for VoxCPM2: the emotion, intensity, manner, volume, pace and non-verbal sound a line can have, with
the words VoxCPM2 is given for each. Candidates came from the style words of the expressive-speech datasets such
models learn from (EARS's 22 emotions and 7 reading styles, Expresso's 26 styles, ParaSpeechCaps's 59 tags) and
from audiobook practice; each was measured (scripts/style_probe.py + style_score.py: 4 voices x 2 neutral
sentences at guidance 1.75, rerun at 3.0 and on emotional sentences; ~1,600 clips).

Status: "works" = recognised as that emotion (emotion2vec+ large); "acoustic" = a clear change in loudness, pitch or
pace, but not recognised as a distinct emotion; "weak" = barely measurable. VoxCPM2 does not render most emotion
words distinctly: they fall into a few acoustic groups (quiet and sad-sounding; higher-pitched; warmer), so one word
per group is kept. It also takes much of a line's feeling from the words (on emotional sentences an instruction
added little). Options that changed nothing are listed in the DROPPED_* tables with what was measured.

The analysis labels lines with the approved tags (delivery.py); RENDER maps each to these words and sounds.
"""
from __future__ import annotations

NEUTRAL = "neutral"

# emotion -> (words for VoxCPM2, status, what was measured against the same voice read "neutral")
EMOTIONS: dict[str, tuple[str, str, str]] = {
    NEUTRAL: ("", "works", "no instruction: the line carries on from the voice's own clip (the clearest voice)"),
    "angry": ("angry", "works", "anger +0.22, pitch x1.19; told apart from the others 56-69% (chance 6%)"),
    "sad": ("sad", "works", "sadness +0.24, -4.7 dB, x0.93 the speed"),
    "crying": ("crying", "acoustic", "the strongest change of any word: -5.9 dB, pitch x1.30, x0.88 the speed"),
    "tender": ("tender", "acoustic", "warmer (anger -0.16, happiness +0.13), -1.5 dB; told apart 38%; calm and "
               "sympathetic sound the same"),
    "happy": ("happy", "acoustic", "brighter: +1.5 dB, pitch x1.16 (happiness itself barely: +0.03)"),
    "confused": ("confused", "acoustic", "pitch x1.20 (questioning); amazed and sarcastic sound the same"),
    "sleepy": ("sleepy", "acoustic", "low energy: -4.0 dB, x0.90 the speed, sadness +0.19; covers tired and bored"),
    "playful": ("playful", "weak", "anger -0.18, happiness +0.10; told apart 19%"),
    "nervous": ("nervous", "weak", "pitch x1.13, -1.7 dB; told apart 6%"),
}
DROPPED_EMOTIONS = {  # no distinct effect: the analysis should use the emotion given instead
    "excited": "only happiness +0.11: use happy", "cheerful": "happiness +0.11: use happy",
    "enthusiastic": "pitch x1.10, nothing else: use happy", "animated": "+1.9 dB, pitch x1.15: use happy, loud",
    "frightened": "sounds sad, not afraid (fear 0.00, sadness +0.13): use nervous or sad",
    "scared": "sounds sad (sadness +0.20, fear 0.00): use nervous or sad",
    "anxious": "pitch x1.12 only: use nervous", "distressed": "sadness +0.20: use sad", "tearful": "use crying or sad",
    "calm": "-2.8 dB, warmer: use tender", "sympathetic": "-3.4 dB, neutral +0.22: use tender",
    "tired": "use sleepy", "bored": "sadness +0.19, slower: use sleepy",
    "surprised": "no surprise (0.00): use the [Surprise-wa] sound", "amazed": "pitch x1.16: use confused",
    "sarcastic": "pitch x1.17: use confused", "disgusted": "disgust 0.05: use angry or [Dissatisfaction-hnn]",
    "disappointed": "sadness +0.12: use sad", "guilty": "-2.6 dB, nothing else", "pained": "sadness +0.12: use sad",
    "embarrassed": "warmer, like tender", "relieved": "warmer, like tender", "proud": "no change",
    "admiring": "pitch x1.15 only", "amused": "sadness +0.11 (!): use [laugh-haha]", "determined": "no change",
    "cold": "sadness +0.13: use the monotonous manner",
}

# intensity -> words placed before the emotion
INTENSITY: dict[str, tuple[str, str, str]] = {
    "normal": ("", "works", ""),
    "strong": ("very", "works", "'very' moved angry 0.09 -> 0.14, sad 0.07 -> 0.13, nervous 0.06 -> 0.12 from neutral"),
}
DROPPED_INTENSITY = {"slight": "'slightly' was no different from the plain word"}

# manner of speaking -> (words, status, measured)
MANNER: dict[str, tuple[str, str, str]] = {
    "normal": ("", "works", ""),
    "breathless": ("breathless", "acoustic", "-4.1 dB, pitch x0.92, x0.93 the speed"),
    "monotonous": ("monotonous", "acoustic", "flat and quieter: -3.8 dB, pitch x0.96 (for cold or deadpan lines)"),
    "enunciated": ("enunciated", "acoustic", "careful: -3.1 dB, x0.93 the speed"),
}
DROPPED_MANNER = {
    "stammering": "words read normally: stutters are spelled out instead (text.spell_stutters)",
    "hesitant": "no change: use the [Uhm] sound", "laughing": "slower, no laugh: use [laugh-haha]",
    "sobbing": "-2.2 dB only: use crying", "groaning": "-2.7 dB only", "gasping": "no change: use [gasp-hah]",
    "screaming": "no louder (-1.2 dB)",
}

# volume -> (words, status, measured)
VOLUME: dict[str, tuple[str, str, str]] = {
    "whisper": ("whispering", "works", "-3.6 dB, voiced share 0.59 (0.67): partly breathy"),
    "soft": ("speaking softly", "works", "-3.2 dB"),
    "normal": ("", "works", ""),
    "loud": ("speaking loudly", "works", "+1.4 dB, pitch x1.25 (+1.0 dB, x1.15 at guidance 3)"),
}
DROPPED_VOLUME = {"shout": "'shouting' and 'screaming' came out no louder: use loud with angry"}

# pace -> (words, status, measured)
PACE: dict[str, tuple[str, str, str]] = {
    "slow": ("slowly", "works", "x0.82 the speed (also -3.3 dB)"),
    "normal": ("", "works", ""),
    "fast": ("speaking quickly", "weak", "x1.07 the speed (x1.12 at guidance 3); 'quickly' alone x1.03"),
}

# non-verbal sound -> (tag written before the line, status, measured over 12 takes: 4 voices x 3 seeds)
# Only tag families VoxCPM2 learned do anything: laugh-, sigh-, Surprise-, Dissatisfaction-, Question-, Uhm, Shh.
SOUNDS: dict[str, tuple[str, str, str]] = {
    "sigh": ("[sigh-hah]", "works", "an audible exhale in 11 of 12 (+1.1 s); [sigh]: 8 of 12"),
    "surprise": ("[Surprise-wa]", "works", "'wah' in 10 of 12; [Surprise-yo] gives 'yo' 12 of 12"),
    "hesitate": ("[Uhm]", "works", "'um' in 8 of 12"),
    "displeasure": ("[Dissatisfaction-hnn]", "works", "'hm/hu' in 8 of 12"),
    "laugh": ("[laugh-haha]", "weak", "a laugh in 5 of 12; [laugh-hehe] 3, [laughing] 2"),
    "gasp": ("[gasp-hah]", "weak", "'ha/ah' in 6 of 12 (+0.3 s)"),
    "hush": ("[Shh]", "weak", "'shh' in 5 of 12"),
}
DROPPED_SOUNDS = {  # silently ignored (no extra time, nothing heard) or capitalised variants of real tags
    "[Laughter]": "nothing", "[cough]": "nothing", "[cough-ahem]": "nothing", "[crying]": "nothing",
    "[cry-huhuhu]": "nothing", "[sob-hic]": "nothing", "[groan]": "nothing", "[groan-ugh]": "nothing",
    "[yawn]": "nothing", "[yawn-haaam]": "nothing", "[scream]": "nothing", "[scream-aaah]": "'ah' in 4 of 12",
    "[gasp]": "nothing", "[breath]": "nothing", "[sniff]": "nothing",
    "[sniff-hmph]": "+0.7 s of something not heard as words (maybe a sniff); untested by ear",
}


def instruction(emotion: str = NEUTRAL, intensity: str = "normal", volume: str = "normal",
                pace: str = "normal", manner: str = "normal") -> str:
    """VoxCPM2's style instruction for a line's slots, e.g. "very angry, speaking loudly"; "" for an all-neutral line
    (no instruction: it carries on from the voice's clip). Unknown values are treated as neutral/normal; an
    intensity without an emotion is dropped."""
    words = EMOTIONS.get(emotion, EMOTIONS[NEUTRAL])[0]
    if words:
        words = " ".join(x for x in (INTENSITY.get(intensity, INTENSITY["normal"])[0], words) if x)
    parts = [words, MANNER.get(manner, MANNER["normal"])[0], VOLUME.get(volume, VOLUME["normal"])[0],
             PACE.get(pace, PACE["normal"])[0]]
    return ", ".join(p for p in parts if p)


def sound_tag(sound: str | None) -> str:
    """The tag written before the line for a non-verbal sound ("" for none or an unknown one)."""
    return SOUNDS[sound][0] if sound in SOUNDS else ""


# Approved delivery tags (delivery.py) -> what VoxCPM2 is given: (instruction words, sound tag before the line).
# Each uses the measured word or sound closest in meaning (the tables above); "" = nothing that measurably works,
# so the words of the line carry it. Unknown tags (typed by the reader) are passed on as they are.
RENDER: dict[str, tuple[str, str]] = {
    "happy": ("happy", ""), "amused": ("playful", ""), "tender": ("tender", ""), "sad": ("sad", ""),
    "crying": ("crying", ""), "angry": ("angry", ""), "cold": ("monotonous", ""), "afraid": ("nervous", ""),
    "surprised": ("amazed", ""),  # 'surprised' did nothing; 'amazed' raises the pitch (x1.16) like surprise
    "confused": ("confused", ""), "embarrassed": ("embarrassed", ""), "firm": ("", ""),
    "sarcastic": ("sarcastic", ""), "tired": ("sleepy", ""), "pained": ("pained", ""),
    "disgusted": ("disgusted", ""),
    "whisper": ("whispering", ""), "soft": ("speaking softly", ""), "loud": ("speaking loudly", ""),
    "shout": ("speaking loudly", ""),  # 'shouting' came out no louder
    "slow": ("slowly", ""), "fast": ("speaking quickly", ""), "breathless": ("breathless", ""),
    "flat": ("monotonous", ""), "clear": ("enunciated", ""), "hesitant": ("", ""), "slurred": ("slurred", ""),
    "laugh": ("", "[laugh-haha]"), "sigh": ("", "[sigh-hah]"), "gasp": ("", "[gasp-hah]"), "um": ("", "[Uhm]"),
    "hmph": ("", "[Dissatisfaction-hnn]"), "whoa": ("", "[Surprise-wa]"), "shh": ("", ""),  # [Shh]: 2 of 6
    "sob": ("", ""), "groan": ("", ""), "cough": ("", ""), "sniff": ("", ""), "yawn": ("", ""),
    "scream": ("", ""),  # no tag VoxCPM2 knows makes these
    "thought": ("speaking softly", ""), "aside": ("speaking softly", ""), "reading": ("", ""),
    "device": ("", ""), "singing": ("singing", ""),
}
RENDER_VERSION = 1  # part of a line's cache key when it has a delivery (app.synthesis_key)

# Sound marks written in the text ("{laugh|Ahaha}, that's fine.", sounds.py) -> the tag VoxCPM2 is given in place
# of the written sound; kinds not listed are read as written. Measured on the library's own lines
# (scripts/sound_probe.py: 15 lines x 3 voices x 2 takes, SenseVoice): laughs were heard 11 of 12 with the tag,
# 8 of 12 as written ("Eheheh" mostly unheard); "Hmph" 5 of 6 (4 as written); a sigh is an exhale with the tag
# (11 of 12 before) where "Haaah" is read as a syllable. Read better as written: "Huh?!" (6 of 6; the
# [Question-*] tags ran into the next word: "ammyal"), "Shhh" (5 of 6; [Shh] 2), "Kyaaah"/"Gyaaah" (5 of 6, no
# runaways; [scream-aaah] 2), "Whoa" (6 of 6; [Surprise-wa] says "wah"), "Ummm" (6 of 6 either way). The tag and
# the written sound together came out doubled and longer every time.
INLINE: dict[str, str] = {"laugh": "[laugh-haha]", "sigh": "[sigh-hah]", "hmph": "[Dissatisfaction-hnn]"}


def render(delivery: str | None) -> tuple[str, str]:
    """(instruction, sound tag) for a delivery: "very angry, shout, laugh" -> ("very angry, speaking loudly",
    "[laugh-haha]"); ("", "") when nothing in it measurably works (the line carries on from the voice's clip)."""
    words, sound = [], ""
    for raw in (delivery or "").split(","):
        tag = " ".join(raw.lower().split())
        if not tag:
            continue
        level, base = tag.split(" ", 1) if tag.split(" ", 1)[0] in ("slightly", "very") and " " in tag else ("", tag)
        if base not in RENDER:
            words.append(tag)  # the reader's own words
            continue
        w, s = RENDER[base]
        if w:
            words.append(f"{level} {w}".strip() if level == "very" else w)  # 'slightly' did nothing
        sound = sound or s
    return ", ".join(dict.fromkeys(words)), sound
