"""Stored speech of short lines (under MISSED_LETTERS letters: "Yes.", "Huh?!", one-word quotations) heard back with
the adapter's current check, and the takes that fail it forgotten, so they are made again when played. Takes made
before the check covered short lines (or before "No." stopped being read "number") were kept as they came.
Run in the VoxCPM2 venv while the server runs (it asks the server for the clips and to forget them):

    ..\\models\\venvs\\voxcpm\\Scripts\\python.exe scripts\\recheck_short_lines.py [--dry-run]
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT.parent / "models" / "modelscope"))
from elvenreader_server.engines.voxcpm_engine import MISSED_LETTERS, missed  # noqa: E402

URL = "http://127.0.0.1:3950"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    import numpy as np
    from funasr import AutoModel
    from scipy.signal import resample_poly

    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda", disable_update=True, disable_pbar=True,
                    log_level="ERROR")
    tok = json.loads((ROOT / "data" / "config.json").read_text(encoding="utf-8-sig"))["token"]
    headers = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
    total_bad = 0
    for f in sorted((ROOT / "data" / "books").glob("*/book.elvenbook")):
        bid = f.parent.name
        with zipfile.ZipFile(f) as z:
            index = json.loads(z.read("audio/index.json")) if "audio/index.json" in z.namelist() else {}
            segs = [json.loads(l) for l in z.read("analysis/segments.jsonl").decode("utf-8").splitlines()]
            bad = []
            for s in segs:
                words = re.sub(r"\{[a-z]+\|([^{}|]*)\}", r"\1", s["text"])
                letters = len(re.sub(r"[^a-z]", "", words.lower()))
                if s["id"] not in index or not 0 < letters < MISSED_LETTERS:
                    continue
                data = z.read(index[s["id"]]["entry"])
                pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000",
                                      "pipe:1"], input=data, capture_output=True).stdout
                x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
                heard = "" if x.size < 400 else re.sub(r"<\|[^|]*\|>", "", asr.generate(
                    input=x, language="en", use_itn=False, disable_pbar=True)[0]["text"])
                miss, total = missed(words, heard)
                longer = len(re.sub(r"[^a-z]", "", heard.lower())) - total
                if not heard or 2 * miss >= total or longer >= max(3, total // 2):
                    bad.append(s["id"])
                    print(f"  {bid} {s['id']}: {words!r} heard {heard!r}", flush=True)
        total_bad += len(bad)
        if bad and not a.dry_run:
            req = urllib.request.Request(f"{URL}/api/books/{bid}/file/forget", headers=headers,
                                         data=json.dumps({"segments": bad}).encode())
            with urllib.request.urlopen(req, timeout=120) as r:
                print(f"{bid}: forgot {len(json.load(r)['forgotten'])} of {len(bad)}", flush=True)
    print(f"{total_bad} short takes failed the check" + (" (dry run)" if a.dry_run else " and were forgotten"))


if __name__ == "__main__":
    main()
