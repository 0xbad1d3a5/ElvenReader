"""Scores style_probe.py's clips: which delivery words VoxCPM2 acts out clearly and distinctly. Runs in the
VoxCPM2 venv (FunASR):

    models\\venvs\\voxcpm\\Scripts\\python.exe server\\scripts\\style_score.py [data\\bench\\styles]

Per clip: emotion2vec+ (large) emotion probabilities and its utterance embedding, SenseVoice-Small transcript,
loudness, voicing, pitch and speaking rate. Each style is compared with the same voice and sentence read
"neutral". Emotions: how far the embedding moves from neutral, the recogniser's probability of the intended basic
emotion, whether slight < moderate < very, and whether an emotion can be told from the others (nearest centroid,
centroids from the other voices). Writes scores.json and report.md next to the clips.
"""
import collections
import difflib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BASIC = {"happy": "happy", "excited": "happy", "sad": "sad", "tearful": "sad", "angry": "angry",
         "frightened": "fearful", "nervous": "fearful", "surprised": "surprised", "disgusted": "disgusted"}
E2V = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]


def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def missed_share(text, heard):
    text = re.sub(r"\[[^\]]*\]", "", text)
    a, b = letters(text), letters(heard)
    got = sum(m.size for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks() if m.size >= 3)
    return (len(a) - got) / max(1, len(a))


def acoustics(x, sr):
    win = int(sr * 0.03)
    n = len(x) // win
    fr = x[: n * win].reshape(n, win)
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    speech = db > db.max() - 35
    idx = np.where(speech)[0]
    span = (idx[-1] - idx[0] + 1) * win / sr if len(idx) else len(x) / sr
    voiced, pitches = 0, []
    lo, hi = int(sr / 500), int(sr / 70)
    for f in fr[speech]:
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.45 * ac[0]:
            voiced += 1
            pitches.append(sr / lag)
    return {"loud_db": float(20 * np.log10(np.sqrt((fr[speech] ** 2).mean()) + 1e-9)),
            "voiced": voiced / max(1, speech.sum()), "pitch": float(np.median(pitches)) if pitches else 0.0,
            "span_s": float(span)}


def score(folder):
    from funasr import AutoModel

    kw = dict(device="cuda", disable_update=True, disable_pbar=True, log_level="ERROR")
    from huggingface_hub import snapshot_download

    # From Hugging Face: ModelScope's copy stalled at 0 bytes.
    e2v = AutoModel(model=snapshot_download("emotion2vec/emotion2vec_plus_large"), **kw)
    asr = AutoModel(model="iic/SenseVoiceSmall", **kw)
    items = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for i, it in enumerate(items):
        x, sr = sf.read(folder / it["file"], dtype="float32")
        y = resample_poly(x, 16000, sr).astype(np.float32)
        r = e2v.generate(input=y, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]
        probs = {lab.split("/")[-1]: float(s) for lab, s in zip(r["labels"], r["scores"])}
        raw = asr.generate(input=y, language="en", use_itn=False, disable_pbar=True)[0]["text"]
        heard = re.sub(r"<\|[^|]*\|>", "", raw).strip()
        it.update(probs=probs, emb=np.asarray(r["feats"], dtype=np.float32).tolist(), heard=heard,
                  events=[t for t in re.findall(r"<\|([^|]*)\|>", raw) if t not in ("en", "woitn", "withitn")],
                  missed=missed_share(it["text"], heard), dur=len(x) / sr, **acoustics(x, sr))
        if (i + 1) % 50 == 0:
            print(f"scored {i + 1} of {len(items)}", flush=True)
    return items


def cos(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def report(items):
    out = []
    by = {(it["voice"], it["text"], it["style"]): it for it in items if it["slot"] != "sound"}
    neutral = {(v, t): it for (v, t, s), it in by.items() if s == "neutral"}
    voices = sorted({it["voice"] for it in items})

    # ---- emotions
    rows = collections.defaultdict(list)
    for it in items:
        if it["slot"] != "emotion":
            continue
        n = neutral[(it["voice"], it["text"])]
        emo, level = it["style"].split("/")
        rows[(emo, level)].append({
            "shift": cos(np.array(it["emb"]), np.array(n["emb"])),
            "target": it["probs"].get(BASIC.get(emo, ""), None),
            "target_neutral": n["probs"].get(BASIC.get(emo, ""), None),
            "missed": it["missed"], "voice": it["voice"], "text": it["text"],
            "delta": np.array(it["emb"]) - np.array(n["emb"]), "top": max(it["probs"], key=it["probs"].get)})
    # Telling emotions apart: moderate and strong takes, centroids from the other voices.
    emos = sorted({e for e, _ in rows})
    correct, confused = collections.Counter(), collections.defaultdict(collections.Counter)
    for held in voices:
        cents = {}
        for e in emos:
            ds = [r["delta"] for lv in ("moderate", "strong") for r in rows[(e, lv)] if r["voice"] != held]
            cents[e] = np.mean(ds, axis=0)
        for e in emos:
            for lv in ("moderate", "strong"):
                for r in rows[(e, lv)]:
                    if r["voice"] != held:
                        continue
                    guess = min(emos, key=lambda k: cos(r["delta"], cents[k]))
                    correct[e] += guess == e
                    confused[e][guess] += 1
    out.append("## Emotions\n")
    out.append("Shift: how far from the same voice read 'neutral' (emotion2vec cosine distance, mean). Target: the "
               "recogniser's probability of the intended basic emotion (neutral read in brackets). Told apart: "
               "share of moderate/very takes whose change from neutral is nearest that emotion's (held-out voice). "
               "Missed: share of letters not heard.\n")
    out.append("| emotion | shift slight / moderate / very | rising | target (neutral) | told apart | mistaken for | missed |")
    out.append("|---|---|---|---|---|---|---|")
    summary = {}
    for e in emos:
        sh = [np.mean([r["shift"] for r in rows[(e, lv)]]) for lv in ("slight", "moderate", "strong")]
        pairs = list(zip(rows[(e, "slight")], rows[(e, "moderate")], rows[(e, "strong")]))
        rising = np.mean([a["shift"] <= b["shift"] <= c["shift"] or a["shift"] <= c["shift"] for a, b, c in pairs])
        tg = [r["target"] for lv in ("moderate", "strong") for r in rows[(e, lv)] if r["target"] is not None]
        tn = [r["target_neutral"] for r in rows[(e, "moderate")] if r["target_neutral"] is not None]
        total = sum(confused[e].values())
        acc = correct[e] / max(1, total)
        wrong = ", ".join(f"{k} {v}" for k, v in confused[e].most_common() if k != e)[:40]
        miss = np.mean([r["missed"] for lv in ("slight", "moderate", "strong") for r in rows[(e, lv)]])
        summary[e] = {"shift": [round(float(s), 3) for s in sh], "rising": round(float(rising), 2),
                      "target": round(float(np.mean(tg)), 2) if tg else None,
                      "target_neutral": round(float(np.mean(tn)), 2) if tn else None,
                      "told_apart": round(acc, 2), "mistaken_for": dict(confused[e]), "missed": round(float(miss), 3)}
        out.append(f"| {e} | {sh[0]:.3f} / {sh[1]:.3f} / {sh[2]:.3f} | {rising:.0%} | "
                   + (f"{np.mean(tg):.2f} ({np.mean(tn):.2f})" if tg else "–")
                   + f" | {acc:.0%} | {wrong} | {miss:.1%} |")

    # ---- volume and pace, against neutral
    def against(slot, keys):
        res = {}
        out.append(f"\n## {slot.title()}\n")
        out.append("| style | loudness vs neutral (dB) | voiced share (neutral) | pitch vs neutral | rate vs neutral | missed |")
        out.append("|---|---|---|---|---|---|")
        for s in keys:
            its = [it for it in items if it["slot"] == slot and it["style"] == s]
            ns = [neutral[(it["voice"], it["text"])] for it in its]
            dl = np.mean([a["loud_db"] - b["loud_db"] for a, b in zip(its, ns)])
            vo, vn = np.mean([a["voiced"] for a in its]), np.mean([b["voiced"] for b in ns])
            pr = np.mean([a["pitch"] / b["pitch"] for a, b in zip(its, ns) if a["pitch"] and b["pitch"]])
            rr = np.mean([b["span_s"] / a["span_s"] for a, b in zip(its, ns)])
            mi = np.mean([a["missed"] for a in its])
            res[s] = {"loud_db": round(float(dl), 1), "voiced": round(float(vo), 2), "voiced_neutral": round(float(vn), 2),
                      "pitch_ratio": round(float(pr), 2), "rate_ratio": round(float(rr), 2), "missed": round(float(mi), 3)}
            out.append(f"| {s} | {dl:+.1f} | {vo:.2f} ({vn:.2f}) | ×{pr:.2f} | ×{rr:.2f} | {mi:.1%} |")
        return res

    vol = against("volume", sorted({it["style"] for it in items if it["slot"] == "volume"}))
    pace = against("pace", sorted({it["style"] for it in items if it["slot"] == "pace"}))

    # ---- sounds: extra time before the words, what was heard before them, events
    out.append("\n## Sounds (tag before the sentence; 3 takes x voices)\n")
    out.append("| tag | extra time vs no tag | heard before the words | events | missed |")
    out.append("|---|---|---|---|---|")
    snd = {}
    base = {(it["voice"], it["take"]): it for it in items if it["slot"] == "sound" and it["style"] == "no tag"}
    for s in sorted({it["style"] for it in items if it["slot"] == "sound"} - {"no tag"}):
        its = [it for it in items if it["slot"] == "sound" and it["style"] == s]
        extra = np.mean([it["span_s"] - base[(it["voice"], it["take"])]["span_s"] for it in its])
        pre = collections.Counter()
        for it in its:
            h = it["heard"].lower()
            k = h.find("the train")
            pre[h[:k].strip() if k > 0 else ("(nothing)" if k == 0 else "?")] += 1
        ev = collections.Counter(e for it in its for e in it["events"] if e not in ("Speech", "EMO_UNKNOWN"))
        mi = np.mean([it["missed"] for it in its])
        snd[s] = {"extra_s": round(float(extra), 2), "before": dict(pre), "events": dict(ev), "missed": round(float(mi), 3)}
        out.append(f"| `{s}` | {extra:+.2f}s | {', '.join(f'{k} ×{v}' for k, v in pre.most_common(4))} | "
                   f"{', '.join(f'{k} ×{v}' for k, v in ev.items()) or '–'} | {mi:.1%} |")
    return "\n".join(out), {"emotions": summary, "volume": vol, "pace": pace, "sounds": snd}


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "bench" / "styles")
    os.environ.setdefault("MODELSCOPE_CACHE", str(Path(__file__).resolve().parents[2] / "models" / "modelscope"))
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "models" / "hf"))
    cache = folder / "scored.json"
    if cache.exists():
        items = json.loads(cache.read_text(encoding="utf-8"))
    else:
        items = score(folder)
        cache.write_text(json.dumps(items), encoding="utf-8")
    text, summary = report(items)
    (folder / "report.md").write_text(text, encoding="utf-8")
    (folder / "scores.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
