"""The approved delivery tags: how a line may be performed. The analysis labels lines with these (the labelling
schema allows nothing else), the line dialog suggests them, and each speech engine renders what it can
(engines/voxcpm_styles.py for VoxCPM2). A delivery is stored as these tags joined by ", " ("very angry, shout,
laugh"); no tags at all is neutral.

Built from the expressive-speech datasets' vocabularies (EARS, Expresso, ParaSpeechCaps), the caption guides
(Netflix SDH, DCMP Captioning Key, BBC Subtitle Guidelines), screenplay parentheticals and the 1,835 free-text
deliveries the analysis had produced. A line has at most two tags: at most one feeling (optionally "slightly" or
"very"), one volume, one pace/manner, one sound and one situation.
"""
from __future__ import annotations

import functools

# tag -> (group, what it covers: the words that map to it)
APPROVED: dict[str, tuple[str, str]] = {
    # feeling (one per line; "slightly"/"very" may come before it)
    "happy": ("feeling", "cheerful, delighted, excited, enthusiastic, relieved, joyful, glad, eager, elated"),
    "amused": ("feeling", "teasing, playful, mischievous, smug, joking, wry humour, grinning"),
    "tender": ("feeling", "gentle, affectionate, sympathetic, reassuring, soothing, warm, kind, loving, comforting"),
    "sad": ("feeling", "sorrowful, disappointed, resigned, regretful, guilty, melancholy, wistful, apologetic"),
    "crying": ("feeling", "tearful, sobbing, choked up, heartbroken, weeping"),
    "angry": ("feeling", "furious, irritated, frustrated, indignant, defiant, annoyed, outraged, exasperated"),
    "cold": ("feeling", "icy, menacing, contemptuous, dismissive, deadpan, disdainful, threatening"),
    "afraid": ("feeling", "scared, nervous, anxious, panicked, uneasy, worried, terrified, timid"),
    "surprised": ("feeling", "shocked, astonished, incredulous, amazed, startled, stunned, in disbelief"),
    "confused": ("feeling", "puzzled, bewildered, perplexed, baffled, doubtful, at a loss"),
    "embarrassed": ("feeling", "flustered, shy, awkward, sheepish, bashful"),
    "firm": ("feeling", "serious, commanding, determined, stern, confident, urgent, insistent, authoritative, "
                        "resolute, earnest"),
    "sarcastic": ("feeling", "wry, mocking, ironic, sardonic, snide"),
    "tired": ("feeling", "sleepy, weary, exhausted, bored, drowsy, groggy"),
    "pained": ("feeling", "hurt, strained, in pain, agonised, wincing"),
    "disgusted": ("feeling", "repulsed, appalled, revolted, grossed out"),
    # volume
    "whisper": ("volume", "whispering, hushed, sotto voce, under the breath"),
    "soft": ("volume", "quiet, murmuring, muttering, low voice, softly"),
    "loud": ("volume", "calling out, raised voice, projecting, loudly"),
    "shout": ("volume", "shouting, yelling, screaming, roaring, bellowing"),
    # pace and manner
    "slow": ("manner", "deliberate, drawn out, measured, slowly, dragging"),
    "fast": ("manner", "hurried, rushed, rapid, quickly, rattling off"),
    "breathless": ("manner", "panting, out of breath, gasping, winded"),
    "flat": ("manner", "monotone, emotionless, robotic, expressionless"),
    "clear": ("manner", "enunciated, precise, articulate, spelling it out, carefully pronounced"),
    "hesitant": ("manner", "stammering, faltering, stuttering, halting, unsure"),
    "slurred": ("manner", "drunk, mumbled, groggy speech"),
    # sounds (a non-verbal sound with the line)
    "laugh": ("sound", "laughing, chuckling, giggling, snickering, cackling"),
    "sigh": ("sound", "sighing, exhaling, huffing"),
    "gasp": ("sound", "gasping, sharp breath"),
    "um": ("sound", "um, uh, er, hmm, thinking pause"),
    "hmph": ("sound", "hmph, tsk, scoff, snort of displeasure"),
    "whoa": ("sound", "wah, whoa, oh!, exclamation of surprise"),
    "shh": ("sound", "shushing"),
    "sob": ("sound", "sniffling, whimpering"),
    "groan": ("sound", "grunting, moaning, effort"),
    "cough": ("sound", "coughing, clearing the throat"),
    "sniff": ("sound", "sniffing"),
    "yawn": ("sound", "yawning"),
    "scream": ("sound", "a scream, shriek"),
    # situation (how the line is heard)
    "thought": ("situation", "inner voice, thinking to oneself, telepathy"),
    "aside": ("situation", "muttered to oneself, under one's breath to someone near"),
    "reading": ("situation", "reading aloud, reciting, quoting a letter or sign, chanting, incantation"),
    "device": ("situation", "phone, radio, loudspeaker, recording, through a wall"),
    "singing": ("situation", "singing, humming"),
}
INTENSITIES = ("slightly", "very")
GROUPS = ("feeling", "volume", "manner", "sound", "situation")
LIMITS = {"feeling": 1, "volume": 1, "manner": 1, "sound": 1, "situation": 1}
# At most this many tags a line: each one changes how the whole line is voiced, and five at once ("afraid, loud,
# fast, clear, sigh" for "Stay close to me.") made a caricature. Over the limit, the groups are kept in this order.
MAX_TAGS = 2
KEEP_FIRST = ("feeling", "volume", "situation", "sound", "manner")
# A tag is kept only when the labelling LLM gave it with at least this probability (its own token probabilities;
# dramatize.delivery_odds). On 115 sample quotations the doubtful ones scored low ("Great! I feel a lot better." ->
# tired 0.09; "Oh, okay. That's a relief." -> happy 0.01) and plain ones high ("H-how dare you!" -> angry 0.88,
# shout 0.95); the 0.5-0.7 band was about half doubtful. 0.7 left 37 of 88 tags.
MIN_CONF = 0.7
NEUTRAL_WORDS = {"neutral", "none", "normal", "calm", "plain", "matter-of-fact", ""}


def all_tags() -> list[str]:
    """Every value a delivery tag may take, intensities included ("very angry"): the labelling schema's enum."""
    out = []
    for tag, (group, _) in APPROVED.items():
        out.append(tag)
        if group == "feeling":
            out += [f"{i} {tag}" for i in INTENSITIES]
    return out


def split(tag: str) -> tuple[str | None, str]:
    """("very", "angry") for "very angry"; (None, tag) otherwise."""
    words = tag.split(" ", 1)
    if len(words) == 2 and words[0] in INTENSITIES and APPROVED.get(words[1], ("",))[0] == "feeling":
        return words[0], words[1]
    return None, tag


def is_approved(tag: str) -> bool:
    return split(tag.strip().lower())[1] in APPROVED


def normalize(tags) -> str | None:
    """A delivery from approved tags (a list, or a "a, b" string): unknown ones and duplicates dropped, at most
    LIMITS per group (the first wins) and MAX_TAGS in all (by KEEP_FIRST), in the order feeling, volume, manner,
    sound, situation. None = neutral."""
    if isinstance(tags, str):
        tags = tags.split(",")
    picked: dict[str, list[str]] = {g: [] for g in GROUPS}
    seen = set()
    for raw in tags or []:
        t = " ".join(str(raw).lower().split())
        _, base = split(t)
        if base not in APPROVED or base in seen:
            continue
        group = APPROVED[base][0]
        if len(picked[group]) < LIMITS[group]:
            picked[group].append(t)
            seen.add(base)
    kept = [t for g in KEEP_FIRST for t in picked[g]][:MAX_TAGS]
    out = [t for g in GROUPS for t in picked[g] if t in kept]
    return ", ".join(out) or None


def prompt_list() -> str:
    """The tags for the labelling prompt, grouped, with what each covers."""
    lines = []
    for group in GROUPS:
        tags = [t for t, (g, _) in APPROVED.items() if g == group]
        lines.append(f"  {group}: " + "; ".join(f"{t} ({APPROVED[t][1]})" for t in tags))
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def synonyms() -> dict[str, str]:
    """Every covered word -> its tag ("furious" -> "angry"), the tags themselves included (don't change it)."""
    out = {}
    for tag, (_, covers) in APPROVED.items():
        for w in covers.split(", "):
            out.setdefault(w.strip().lower(), tag)
    out.update({t: t for t in APPROVED})
    return out


# Words of degree in free-text deliveries -> the approved intensity ("" = drop the word).
DEGREE = {"very": "very", "extremely": "very", "deeply": "very", "really": "very", "so": "very", "utterly": "very",
          "completely": "very", "intensely": "very", "highly": "very", "slightly": "slightly", "a bit": "slightly",
          "a little": "slightly", "somewhat": "slightly", "mildly": "slightly", "faintly": "slightly",
          "barely": "slightly", "moderately": ""}


def match(word: str, known: dict[str, list[str]] | None = None) -> list[str] | None:
    """The approved tags for one free-text delivery word ("very furious" -> ["very angry"]; "neutral" -> []),
    from the covered words or `known` (word -> tags, e.g. a mapping made by the LLM); None if unknown."""
    w = " ".join(word.lower().split())
    if w in NEUTRAL_WORDS:
        return []
    if known and w in known:
        return list(known[w])
    syn = synonyms()
    if w in syn:
        return [syn[w]]
    for deg in sorted(DEGREE, key=len, reverse=True):
        if w.startswith(deg + " ") and (base := match(w[len(deg) + 1:], known)) is not None:
            level = DEGREE[deg]
            return [f"{level} {t}" if level and APPROVED[t][0] == "feeling" and split(t)[0] is None else t
                    for t in base]
    return None


def from_text(text: str | None, known: dict[str, list[str]] | None = None) -> str | None:
    """A free-text delivery ("shouting, urgent") in approved tags ("firm, shout"); words that match nothing
    are dropped."""
    tags = []
    for part in (text or "").split(","):
        tags += match(part, known) or []
    return normalize(tags)


# Clips are levelled to one loudness (audio.level); these deliveries are meant to stand out from it (dB).
LEVEL_OFFSETS = {"whisper": -6.0, "soft": -3.0, "aside": -3.0, "thought": -2.0, "loud": 2.0, "shout": 3.0}


def level_offset_db(delivery: str | None) -> float:
    """How far a line with this delivery sits from the levelled loudness: the quietest or loudest of its tags."""
    offsets = [LEVEL_OFFSETS[t] for t in (" ".join(p.lower().split()) for p in (delivery or "").split(","))
               if t in LEVEL_OFFSETS]
    return min(offsets) if offsets and min(offsets) < 0 else max(offsets, default=0.0)
