"""Which delivery words VoxCPM2 actually acts out: generates every candidate style (emotion x intensity, volume,
pace, non-verbal sound tags) on several voices through the running server's /api/tts, for style_score.py to
measure. Neutral sentences, so the words carry no feeling of their own.

    .venv\\Scripts\\python.exe scripts\\style_probe.py [--voices ID,ID,...] [--out data\\bench\\styles]

Writes <out>\\<n>.wav and <out>\\manifest.json ({file, slot, style, instruction, text, voice}).
"""
import argparse
import json
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ["The train leaves at seven, and I will be on it.",
         "I put the letters on the table by the door this morning."]
VOICES = ["ch-c37d5be2-ronie-38930", "ch-5b755a6e-kirito", "ch-c37d5be2-fanatio-37606",
          "ch-5b755a6e-deusolbert-synthesis-seven"]
EMOTIONS = ["happy", "excited", "sad", "tearful", "angry", "frightened", "nervous", "surprised", "disgusted",
            "tender", "playful", "embarrassed", "determined", "cold", "sarcastic", "tired"]
INTENSITY = {"slight": "slightly {}", "moderate": "{}", "strong": "very {}"}
VOLUME = ["whispering", "speaking softly", "speaking loudly", "shouting"]
PACE = ["slowly", "speaking slowly", "quickly", "speaking quickly"]
SOUNDS = ["[laughing]", "[laugh-haha]", "[sigh]", "[sigh-hah]", "[Uhm]", "[Shh]", "[Surprise-wa]", "[Surprise-yo]",
          "[Dissatisfaction-hnn]"]


def cases():
    for text in TEXTS:
        yield {"slot": "baseline", "style": "none", "instruction": "", "text": text}  # carries on from the clip
        yield {"slot": "baseline", "style": "neutral", "instruction": "neutral", "text": text}
        for e in EMOTIONS:
            for level, fmt in INTENSITY.items():
                yield {"slot": "emotion", "style": f"{e}/{level}", "instruction": fmt.format(e), "text": text}
        for v in VOLUME:
            yield {"slot": "volume", "style": v, "instruction": v, "text": text}
        for p in PACE:
            yield {"slot": "pace", "style": p, "instruction": p, "text": text}
    for s in SOUNDS + ([""] if SOUNDS else []):
        for i in range(3):  # sounds vary most from take to take: three takes, the voice's seed and two others
            yield {"slot": "sound", "style": s or "no tag", "instruction": "", "text": f"{s} {TEXTS[0]}".strip(),
                   "seed_offset": i}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voices", default=",".join(VOICES))
    p.add_argument("--out", default=str(ROOT / "data" / "bench" / "styles"))
    p.add_argument("--url", default="http://127.0.0.1:3950")
    p.add_argument("--guidance", type=float, help="every clip at this guidance (default: the site's)")
    p.add_argument("--only", help="comma-separated styles to make (e.g. neutral,angry/moderate,shouting)")
    p.add_argument("--text", action="append", help="sentences to read instead of the neutral ones (repeatable)")
    p.add_argument("--emotions", help="comma-separated emotion words to try instead of the built-in list")
    p.add_argument("--levels", help="comma-separated intensities: slight,moderate,strong (default all)")
    p.add_argument("--sounds", help="comma-separated tags to try instead of the built-in list ('none' for no sounds)")
    p.add_argument("--no-volume-pace", action="store_true", help="skip the volume and pace styles")
    a = p.parse_args()
    if a.text:
        TEXTS[:] = a.text
    if a.emotions:
        EMOTIONS[:] = a.emotions.split(",")
    if a.levels:
        for k in [k for k in INTENSITY if k not in a.levels.split(",")]:
            del INTENSITY[k]
    if a.sounds:
        SOUNDS[:] = [] if a.sounds == "none" else a.sounds.split(",")
    if a.no_volume_pace:
        VOLUME[:], PACE[:] = [], []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    token = json.loads((ROOT / "data" / "config.json").read_text(encoding="utf-8-sig"))["token"]
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    manifest, n = [], 0
    all_cases = list(cases())
    if a.only:
        wanted = set(a.only.split(","))
        all_cases = [c for c in all_cases if c["style"] in wanted]
    for voice in a.voices.split(","):
        for c in all_cases:
            body = {"text": c["text"], "voice": voice, "language": "English", "sentence_pause_ms": 0, "cache": False}
            if a.guidance is not None:
                body["guidance"] = a.guidance
            if c["instruction"]:
                body["delivery"] = c["instruction"]
            if c.get("seed_offset"):
                body["seed"] = 1000 + 7919 * c["seed_offset"]
            req = urllib.request.Request(a.url + "/api/tts", data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=900) as r:
                pcm, rate = r.read(), int(r.headers["X-Sample-Rate"])
            name = f"{n:04d}.wav"
            with wave.open(str(out / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            manifest.append({"file": name, "voice": voice, **{k: v for k, v in c.items() if k != "seed_offset"},
                             "take": c.get("seed_offset", 0)})
            n += 1
            if n % 25 == 0:
                print(f"{n} of {len(all_cases) * len(a.voices.split(','))}", flush=True)
                (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"done: {n} clips in {out}")


if __name__ == "__main__":
    sys.exit(main())
