"""Transcribe the Breeze and VoxCPM2 test clips with Whisper and compare them with the text they were given."""
import json
import re
import sys
from pathlib import Path

import whisper

TEXTS = {
    "narration": "Ronie watched her dragon nervously. Tsukigake blinked a few times, thinking it over, then stretched "
                 "her neck toward her master and dropped the trout onto the grass.",
    "shout": "Stop right there! Don't you dare take another step!",
    "whisper": "Quiet. They're right outside the door, so don't make a sound.",
    "long": "It was still far too early to go to school, of course. School started at eight o'clock sharp, and it took "
            "them thirty minutes to get to school, including walking. They just needed to leave at seven thirty. Making "
            "breakfast, eating it, and cleaning up would still leave them with more than an hour of free time.",
}


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower().replace("\u2019", "'"))


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


model = whisper.load_model(sys.argv[1])
for folder in sys.argv[2:]:
    for name, text in TEXTS.items():
        f = Path(folder) / f"{name}.wav"
        if not f.exists():
            continue
        heard = model.transcribe(str(f), language="en", fp16=True)["text"].strip()
        print(json.dumps({"set": Path(folder).name, "case": name, "wer": round(wer(text, heard), 3), "heard": heard}))
