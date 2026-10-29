"""Character profile: what a voice-design model needs to know about a character, nothing more.

Five fields: gender, apparent age (what a listener should hear), physical appearance, speech style
(how they talk, verbal tics) and notes (other voice-relevant traits, kept short). The labelling LLM
fills them in as it reads (`profile_updates` in each passage's answer); the roster step fills what is
still empty from the appearance evidence it gathers; long notes are compacted at the end of the
analysis. The reader's edits (`profile_user`) always win. The profile only shapes voices: it doesn't
change who says what.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("androidreader.drama")

FIELDS = ("gender", "apparent_age", "appearance", "speech_style", "notes")
GENDERS = ("male", "female", "unknown")
AGES = ("child", "teen", "young adult", "adult", "middle-aged", "elderly", "unknown")
NOTES_MAX = 300
TEXT_MAX = 160

UPDATES_SCHEMA = {
    "type": "array", "maxItems": 6,
    "items": {"type": "object", "properties": {
        "character": {"type": "string"},
        "gender": {"type": "string", "enum": list(GENDERS)},
        "apparent_age": {"type": "string", "enum": list(AGES)},
        "appearance": {"type": "string", "maxLength": TEXT_MAX},
        "speech_style": {"type": "string", "maxLength": TEXT_MAX},
        "notes": {"type": "string", "maxLength": 200}},
        "required": ["character"]},
}

PROMPT_RULE = (
    "Character profiles (used to design each character's voice): in \"profile_updates\", note what this passage "
    "states or clearly shows about how its speaking characters look or sound, one entry per character (roster name) "
    "with only the fields the passage supports. Never guess: leave a field out when the passage says nothing about it, "
    "so most entries have one or two fields. gender; apparent_age (how old they look and sound: a centuries-old being "
    "in a little girl's body is \"child\"); appearance (body and form only, not where they are or their job: \"young "
    "girl in a kimono\", \"Java sparrow\", \"android girl\", \"burly man with a scar\"); speech_style (how they talk "
    "and their verbal tics: \"old-fashioned, calls people dearie\", \"terse and formal\"); notes (lasting traits that "
    "shape the voice: true age or nature, temperament, accent; not plot events). Skip what the roster already says, "
    "and give [] if the passage shows nothing about anyone.")

def effective(c: dict) -> dict:
    """The profile as used: the model's, with the reader's edits on top."""
    return {k: v for k, v in {**(c.get("profile") or {}), **(c.get("profile_user") or {})}.items()
            if v and not blank(v)}


# What a model writes when it has nothing to say ("none described", "Not mentioned", "N/A").
_PLACEHOLDER = re.compile(
    r"^\W*(?:none|nothing|n/?a|unknown|unspecified|unclear|not applicable)\W*$"
    r"|^\W*(?:none|nothing|not|never|no\s+\w+)\s+(?:yet\s+|really\s+|clearly\s+)?"
    r"(?:described|mentioned|given|specified|stated|noted|shown|known)\b", re.IGNORECASE)


def blank(text) -> bool:
    return isinstance(text, str) and bool(_PLACEHOLDER.search(text.strip()))


def clean(c: dict) -> bool:
    """Drops placeholder values from the model's profile (the reader's own text is left alone)."""
    p = c.get("profile")
    if not p:
        return False
    before = dict(p)
    for key in ("appearance", "speech_style", "notes"):
        if isinstance(p.get(key), str):
            kept = "; ".join(x for x in (s.strip() for s in p[key].split(";")) if x and not blank(x))
            if kept:
                p[key] = kept
            else:
                p.pop(key)
    return p != before


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", (text or "").lower()))


def _add_note(notes: str, new: str) -> str:
    parts = [p.strip() for p in re.split(r"\s*;\s*", notes or "") if p.strip()]
    for piece in re.split(r"\s*;\s*", new or ""):
        piece = piece.strip().rstrip(".")
        if piece and not any(_norm(piece) in _norm(p) for p in parts):
            parts.append(piece)
    return "; ".join(parts)


def merge(c: dict, upd: dict) -> bool:
    """Folds one update into the model's profile. Fixed fields are only set when empty (or "unknown"), or
    replaced by a more specific version of the same thing; notes and speech style accumulate. Returns
    whether anything changed."""
    p = c.setdefault("profile", {})
    before = dict(p)
    for key in ("gender", "apparent_age"):
        v = (upd.get(key) or "").strip().lower()
        if v and v != "unknown" and v in (GENDERS if key == "gender" else AGES):
            # A majority over the passages, so one passage's guess (a role-play, a disguise) doesn't stick.
            votes = c.setdefault("profile_votes", {}).setdefault(key, {})
            votes[v] = votes.get(v, 0) + 1
            if p.get(key, "unknown") == "unknown" or votes[v] > votes.get(p[key], 0):
                p[key] = v
    for key in ("appearance", "speech_style"):
        v = " ".join(str(upd.get(key) or "").split())[:TEXT_MAX]
        if not v or blank(v):
            continue
        old = p.get(key, "")
        if not old or _norm(old) in _norm(v):
            p[key] = v
        elif key == "speech_style" and _norm(v) not in _norm(old):
            p[key] = _add_note(old, v)
    if upd.get("notes") and not blank(str(upd["notes"])):
        p["notes"] = _add_note(p.get("notes", ""), " ".join(str(upd["notes"]).split()))
    if not p:
        c.pop("profile", None)
    return p != before


def apply_updates(script: dict, updates: list) -> int:
    from .dramatize import find_character

    n = 0
    for upd in updates or []:
        if not isinstance(upd, dict):
            continue
        cid = find_character(script, str(upd.get("character") or ""))
        if cid and merge(script["characters"][cid], upd):
            sync_character(script["characters"][cid])
            n += 1
    return n


def sync_character(c: dict) -> None:
    """Apparent gender and age from the profile drive pool voices and casting."""
    clean(c)
    d = effective(c)
    if d.get("gender") in ("male", "female"):
        c["gender"] = d["gender"]
    if d.get("apparent_age") and d["apparent_age"] != "unknown":
        c["age"] = d["apparent_age"]


def voice_description(c: dict) -> str:
    """A voice-design description from the profile."""
    d = effective(c)
    who = " ".join(x for x in (d.get("apparent_age"), d.get("gender")) if x and x != "unknown") or "adult"
    out = f"{'An' if who[0] in 'aeiou' else 'A'} {who} voice"
    if d.get("appearance"):
        out += f" for a character who appears as {d['appearance']}"
    out += "."
    if d.get("speech_style"):
        out += f" Way of speaking: {d['speech_style']}."
    if d.get("notes"):
        out += f" {d['notes'][0].upper()}{d['notes'][1:]}."
    return out.replace("..", ".")


def from_dossier(d: dict) -> dict:
    """The old dossier shape -> the profile: form becomes appearance, true nature and personality go into
    the notes, speech style is kept."""
    out = {}
    for key in ("gender", "apparent_age", "speech_style"):
        if d.get(key) and d[key] != "unknown":
            out[key] = str(d[key])[:TEXT_MAX * 2]
    if d.get("form"):
        out["appearance"] = str(d["form"])[:TEXT_MAX]
    notes = ""
    for key in ("true_nature", "personality"):
        if d.get(key):
            notes = _add_note(notes, str(d[key]))
    if notes:
        out["notes"] = compact_text(notes)
    return out


def compact_text(notes: str, limit: int = NOTES_MAX) -> str:
    """Without an LLM: drop repeated pieces, then keep whole pieces up to the limit."""
    out = ""
    for piece in re.split(r"\s*;\s*", notes or ""):
        if piece and len(out) + len(piece) + 2 <= limit:
            out = _add_note(out, piece)
    return out


def migrate(script: dict) -> int:
    """Books analysed before profiles: convert dossiers (the reader's edits stay the reader's) and, for
    everyone else, start from what the roster step wrote (appearance, accent, temperament)."""
    n = 0
    for c in script.get("characters", {}).values():
        n += clean(c)  # v2: placeholders such as "none described" are not descriptions
        if "profile" in c or "profile_user" in c:
            continue
        model = from_dossier(c.get("dossier") or {})
        if not model.get("appearance") and c.get("appearance"):
            model["appearance"] = " ".join(c["appearance"].split())[:TEXT_MAX]
        extra = "; ".join(x for x in (c.get("temperament"), c.get("accent") and c["accent"] != "neutral"
                                      and f"{c['accent']} accent") if x)
        if extra:
            model["notes"] = compact_text(_add_note(model.get("notes", ""), extra))
        user = from_dossier(c.get("dossier_user") or {})
        if model:
            c["profile"] = model
        if user:
            c["profile_user"] = user
        n += bool(model or user)
    script["profile_v"] = 2
    return n


def set_user_fields(c: dict, fields: dict) -> None:
    """The reader's profile: a field equal to the model's (or empty) goes back to the model's."""
    model, mine = c.get("profile") or {}, c.setdefault("profile_user", {})
    for key, value in fields.items():
        if key not in FIELDS:
            continue
        value = " ".join(value.split()) if isinstance(value, str) else ""
        if value and value != model.get(key):
            mine[key] = value[:NOTES_MAX if key == "notes" else TEXT_MAX * 2]
        else:
            mine.pop(key, None)
    if not mine:
        c.pop("profile_user", None)
    sync_character(c)


COMPACT_PROMPT = """Shorten each character's profile for a voice-design model. "notes": keep only lasting traits that shape how they should sound (true age or nature, temperament, accent, manner), merge repeats, drop plot details; at most about 40 words. "speech_style": how they talk and their verbal tics, merged into at most about 20 words. Short phrases separated by semicolons."""
SPEECH_MAX = 200
COMPACT_NOTES_AT, COMPACT_SPEECH_AT = 120, 100  # longer than this after reading: tidied by the LLM


async def compact(llm, session, script: dict) -> int:
    """Notes and speech styles that grew while reading (plot details creep in, repeats pile up) are tidied
    by the LLM in one call at the end of the analysis, or cut down without it."""
    long = {cid: c["profile"] for cid, c in script["characters"].items()
            if len((c.get("profile") or {}).get("notes", "")) > COMPACT_NOTES_AT
            or len((c.get("profile") or {}).get("speech_style", "")) > COMPACT_SPEECH_AT}
    if not long:
        return 0
    names = [script["characters"][cid]["name"] for cid in long]
    schema = {"type": "object", "properties": {"characters": {"type": "array", "minItems": len(long), "maxItems": len(long),
              "items": {"type": "object", "properties": {"name": {"type": "string", "enum": names},
                                                         "speech_style": {"type": "string", "maxLength": SPEECH_MAX},
                                                         "notes": {"type": "string", "maxLength": NOTES_MAX}},
                        "required": ["name", "speech_style", "notes"]}}}, "required": ["characters"]}
    user = "\n".join(f"- {script['characters'][cid]['name']}: speech_style: {p.get('speech_style', '')} | notes: {p.get('notes', '')}"
                     for cid, p in long.items())
    try:
        result = await llm.complete(session, COMPACT_PROMPT, user, schema, max_tokens=256 + 160 * len(long))
        short = {a.get("name"): a for a in result.get("characters", [])}
    except Exception as exc:  # noqa: BLE001
        logger.warning("compacting profiles failed (%s); cutting them down", exc)
        short = {}
    for cid, p in long.items():
        a = short.get(script["characters"][cid]["name"]) or {}
        for key, limit in (("notes", NOTES_MAX), ("speech_style", SPEECH_MAX)):
            if not p.get(key):
                continue
            new = " ".join(str(a.get(key) or "").split())
            p[key] = new[:limit] if new else compact_text(p[key], limit)
    return len(long)