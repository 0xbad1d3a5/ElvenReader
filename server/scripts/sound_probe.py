"""Inline sounds written in the books ("Ahaha, that's fine.", "Haaah. Really...", "Kyaaah!"): does VoxCPM2 do
better reading them as written, or given one of its sound tags in their place? Real lines from the library, each
made as written, without the sound, with the tag instead of it, and with the tag and the written sound, on several
voices and takes, through the running server's /api/tts. Score with style_score.py-like measures:

    .venv\\Scripts\\python.exe scripts\\sound_probe.py [--out data\\bench\\sounds]
    ..\\models\\venvs\\voxcpm\\Scripts\\python.exe scripts\\sound_probe.py --score [--out data\\bench\\sounds]
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES = ["ch-c37d5be2-ronie-38930", "ch-5b755a6e-kirito", "ch-c37d5be2-fanatio-37606"]
# (kind, written sound, rest of the line, VoxCPM2 tags to try in its place)
LINES = [
    ("laugh", "Ahaha,", "that's fine. I'll be right back with it.", ["[laugh-haha]"]),
    ("laugh", "Eheheh.", "It's an ornament and a hairpin.", ["[laugh-hehe]", "[laugh-haha]"]),
    ("sigh", "Haaah.", "Really, I just want to read books.", ["[sigh-hah]"]),
    ("sigh", "Haaah...", "You would not be a good merchant, Myne.", ["[sigh-hah]"]),
    ("um", "Ummm,", "I'm going to make pound cake.", ["[Uhm]"]),
    ("um", "Um,", "it sounds like there's a lot going on outside the door...", ["[Uhm]"]),
    ("hmph", "Hmph.", "You can handle a few at a time, but how about many at once?", ["[Dissatisfaction-hnn]"]),
    ("huh", "Huh?!", "Me? Alright. I'll give it a shot.", ["[Question-ah]", "[Question-en]"]),
    ("huh", "Bwuh?", "But I can do it just fine.", ["[Question-ah]", "[Question-en]"]),
    ("shh", "Shhh!", "You absolutely cannot say that out loud!", ["[Shh]"]),
    ("scream", "Kyaaah!", "What's going on?!", ["[scream-aaah]"]),
    ("scream", "Gyaaah!", "So cold! They're so cold!", ["[scream-aaah]"]),
    ("groan", "Ngh...", "I haven't done that much.", []),
    ("groan", "Ngggh... Sniff...", "Good night...", []),
    ("whoa", "Whoa!", "This is so cute!", ["[Surprise-wa]"]),
]
TAKES = 2


def cases():
    for i, (kind, sound, rest, tags) in enumerate(LINES):
        yield {"line": i, "kind": kind, "variant": "written", "text": f"{sound} {rest}", "rest": rest}
        yield {"line": i, "kind": kind, "variant": "none", "text": rest[0].upper() + rest[1:], "rest": rest}
        for t in tags:
            yield {"line": i, "kind": kind, "variant": t, "text": f"{t}{rest[0].upper() + rest[1:]}", "rest": rest}
            yield {"line": i, "kind": kind, "variant": t + " + written", "text": f"{t}{sound} {rest}", "rest": rest}


def make(out: Path, url: str, voices: list[str]):
    token = json.loads((ROOT / "data" / "config.json").read_text(encoding="utf-8-sig"))["token"]
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    manifest, n = [], 0
    for voice in voices:
        for c in cases():
            for take in range(TAKES):
                body = {"text": c["text"], "voice": voice, "language": "English", "sentence_pause_ms": 0,
                        "cache": False}
                if take:
                    body["seed"] = 1000 + 7919 * take
                req = urllib.request.Request(url + "/api/tts", data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=900) as r:
                    pcm, rate = r.read(), int(r.headers["X-Sample-Rate"])
                name = f"{n:04d}.wav"
                with wave.open(str(out / name), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(pcm)
                manifest.append({"file": name, "voice": voice, "take": take, **c})
                n += 1
                if n % 20 == 0:
                    print(n, flush=True)
                    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("made", n)


def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def score(out: Path):
    import numpy as np
    import soundfile as sf
    from funasr import AutoModel
    from scipy.signal import resample_poly

    os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT.parent / "models" / "modelscope"))
    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda", disable_update=True, disable_pbar=True,
                    log_level="ERROR")
    items = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for it in items:
        x, sr = sf.read(out / it["file"], dtype="float32")
        raw = asr.generate(input=resample_poly(x, 16000, sr).astype(np.float32), language="en", use_itn=False,
                           disable_pbar=True)[0]["text"]
        heard = re.sub(r"<\|[^|]*\|>", "", raw).strip()
        a, b = letters(it["rest"]), letters(heard)
        blocks = [m for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks() if m.size >= 3]
        got = sum(m.size for m in blocks)
        it.update(heard=heard, dur=len(x) / sr, missed=(len(a) - got) / max(1, len(a)),
                  before=b[: blocks[0].b] if blocks and blocks[0].a < 4 else "?",
                  events=[t for t in re.findall(r"<\|([^|]*)\|>", raw)
                          if t not in ("en", "woitn", "withitn", "Speech", "EMO_UNKNOWN", "NEUTRAL")])
    (out / "scored.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    rows = collections.defaultdict(list)
    for it in items:
        rows[(it["line"], it["variant"])].append(it)
    base = {k[0]: np.mean([i["dur"] for i in v]) for k, v in rows.items() if k[1] == "none"}
    lines = ["| line | variant | extra s vs none | heard before the words | events | missed rest | max dur |",
             "|---|---|---|---|---|---|---|"]
    for (ln, var), its in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] != "written", kv[0][1])):
        ev = collections.Counter(e for i in its for e in i["events"])
        pre = collections.Counter(i["before"] or "(nothing)" for i in its)
        lines.append(f"| {LINES[ln][1]} {LINES[ln][2][:25]} | {var} | {np.mean([i['dur'] for i in its]) - base[ln]:+.2f} | "
                     f"{', '.join(f'{k} x{v}' for k, v in pre.most_common(4))} | "
                     f"{', '.join(f'{k} x{v}' for k, v in ev.items()) or '-'} | "
                     f"{np.mean([i['missed'] for i in its]):.0%} | {max(i['dur'] for i in its):.1f} |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "bench" / "sounds"))
    p.add_argument("--url", default="http://127.0.0.1:3950")
    p.add_argument("--voices", default=",".join(VOICES))
    p.add_argument("--score", action="store_true", help="score the clips (in the VoxCPM2 venv)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.score:
        score(out)
    else:
        make(out, a.url, a.voices.split(","))


if __name__ == "__main__":
    sys.exit(main())
