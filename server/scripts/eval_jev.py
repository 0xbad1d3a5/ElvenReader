"""Can a Jev-like classifier flag the LLM's attribution mistakes? (OpenJev-Flash-9B, jeff/GLiFormer)

Each gold quotation is asked as one classification: the state is the local scene (about 250 words
before, the paragraph with the quotation marked, a little after), the options are the characters the
default LLM pass found speaking nearby, plus "someone else" and "nobody". Scored like eval_flagger.py:
standalone accuracy, and as a flagger (disagreement with the LLM), with precision and recall of the
LLM's errors at several confidence thresholds.

    # OpenJev: llama.cpp serves the GGUF (this script starts it)
    python scripts\\eval_jev.py --backend openjev --llm data\\eval\\llm-classics.json tests\\gold\\pride-ch1.json ...
    # jeff (GLiFormer-large): start its server first, in the jeff venv:
    #   $env:JEFF_MODEL="D:\\Models\\ElvenReader\\gliformer-large-v1"; $env:JEFF_API_KEYS="devkey"; jeff
    python scripts\\eval_jev.py --backend jeff --book 2f145f7d38c07f00 tests\\gold\\private\\sasaki10-ch8.json ...
    # routing: re-check the confident disagreements with the default LLM (reasoning, wider scene)
    python scripts\\eval_jev.py --recheck rows.json --min-conf 0.5 --before-words 600 --after-words 150 ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))

import aiohttp  # noqa: E402

from elvenreader_server import dramatize as dz  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402

OTHER, NOBODY = "someone else (not listed)", "nobody (a sign, a title or quoted words)"
QUESTION = "Who says the quotation marked with \u27e6 \u27e7?"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def norm(x: str) -> str:
    return dz.norm_name(x or "")


# ----------------------------------------------------------------- items


def items_for(gold: dict, chapters: list[dict], llm_names: dict[str, str], narrator_at, window: int = 25,
              before_words: int = 250, after_words: int = 80) -> list[dict]:
    """One classification item per gold quotation."""
    out = []
    flat = [(ci, q) for ci, ch in enumerate(chapters) for q in ch.get("quotes", [])]
    index = {q[0]: i for i, (_, q) in enumerate(flat)}
    for g in gold["quotes"]:
        i = index[g["id"]]
        ci, (qid, p, s, e) = flat[i]
        paras = chapters[ci]["paragraphs"]
        before, words, k = [], 0, p - 1
        while k >= 0 and words < before_words:
            before.insert(0, paras[k])
            words += len(paras[k].split())
            k -= 1
        after, words, k = [], 0, p + 1
        while k < len(paras) and words < after_words:
            after.append(paras[k])
            words += len(paras[k].split())
            k += 1
        t = paras[p]
        marked = t[:s] + "\u27e6" + t[s:e] + "\u27e7" + t[e:]
        state = "\n\n".join(before + [marked] + after)
        who = narrator_at(ci, p)
        if who:
            state = f"(This part of the story is told in the first person by {who}: \"I\" is {who}.)\n\n" + state
        names = []
        for _, q in flat[max(0, i - window): i + window + 1]:
            n = llm_names.get(q[0])
            if n and n != "NARRATOR" and norm(n) not in {norm(x) for x in names}:
                names.append(n)
        if who and norm(who) not in {norm(x) for x in names}:
            names.append(who)
        out.append({"id": qid, "gold": g["speaker"], "llm": llm_names.get(qid, "NARRATOR"), "state": state,
                    "options": names + [OTHER, NOBODY]})
    return out


# ----------------------------------------------------------------- backends


async def ask_openjev(session, url: str, item: dict) -> dict[str, float]:
    opts = "\n".join(f"[{LETTERS[i]}] {o}" for i, o in enumerate(item["options"]))
    prompt = (f"State:\n{item['state']}\n\nQuestion: {QUESTION}\nOptions:\n{opts}\n\n"
              "Answer with the letter of the best option only.")
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
    async with session.post(f"{url}/v1/chat/completions", json=body) as r:
        data = await r.json()
    probs = {o: 0.0 for o in item["options"]}
    for t in data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        letter = t["token"].strip().strip("[]()")
        if len(letter) == 1 and letter in LETTERS[:len(item["options"])]:
            probs[item["options"][LETTERS.index(letter)]] += math.exp(t["logprob"])
    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}


async def ask_jeff(session, url: str, item: dict) -> dict[str, float]:
    body = {"state": item["state"], "model": "jev-latest",
            "questions": {"speaker": {"type": "choice", "instructions": QUESTION,
                                      "criteria": {o: None for o in item["options"]}}}}
    key = os.environ.get("JEFF_API_KEY", "devkey")
    async with session.post(f"{url}/v1/systemone", json=body, headers={"Authorization": "Bearer " + key}) as r:
        data = await r.json()
    if "answers" not in data:
        raise RuntimeError(f"jeff: {data}")
    return data["answers"]["speaker"]["probabilities"]


RECHECK_SYSTEM = ("You attribute dialogue in novels. Read the scene and work out who says the marked quotation: "
                  "follow speech tags, turn-taking, who is addressed, and who is present. In first-person passages "
                  "\"I\" is the narrator named in the note. Then pick one option.")


async def ask_llm(llm, session, item: dict) -> dict[str, float] | None:
    """Routing re-check: the default LLM, reasoning first, on a wider scene. None if it fails."""
    opts = "\n".join(f"- {o}" for o in item["options"])
    user = f"Scene:\n{item['state']}\n\nQuestion: {QUESTION}\nOptions:\n{opts}"
    schema = {"type": "object", "properties": {"speaker": {"type": "string", "enum": item["options"]}},
              "required": ["speaker"]}
    try:
        r = await llm.complete(session, RECHECK_SYSTEM, user, schema, max_tokens=12000, thinking=True,
                               effort="medium")
    except dz.LlmError as exc:
        print("   re-check failed:", item["id"], exc)
        return None
    return {o: float(o == r["speaker"]) for o in item["options"]}


# ----------------------------------------------------------------- scoring


def evaluate(items: list[dict], gold: dict) -> list[dict]:
    alias = {}
    for canon, al in gold.get("characters", {}).items():
        for a in [canon, *al]:
            alias[norm(a)] = canon
    canon = lambda x: alias.get(norm(x), norm(x))  # noqa: E731
    rows = []
    for it in items:
        if it["gold"] == "*":
            continue
        truth = {canon(t) for t in it["gold"].split("|")}
        listed = {canon(o) for o in it["options"][:-2]}
        pick = max(it["probs"], key=it["probs"].get)
        conf = it["probs"][pick]

        def right(label: str) -> bool:
            if label == OTHER:
                return not truth & listed and "NARRATOR" not in it["gold"]
            if label == NOBODY:
                return "NARRATOR" in {t.upper() for t in it["gold"].split("|")}
            return canon(label) in truth

        llm = it["llm"]
        llm_right = ("NARRATOR" in {t.upper() for t in it["gold"].split("|")}) if llm == "NARRATOR" else canon(llm) in truth
        if pick == OTHER:
            agree = canon(llm) not in listed and llm != "NARRATOR"
        elif pick == NOBODY:
            agree = llm == "NARRATOR"
        else:
            agree = canon(pick) == canon(llm)
        rows.append({"id": it["id"], "jev_right": right(pick), "llm_right": llm_right, "agree": agree, "conf": conf,
                     "pick": pick, "llm": llm, "gold": it["gold"]})
    return rows


def report(name: str, rows: list[dict], secs: float) -> None:
    n = len(rows)
    errs = sum(not r["llm_right"] for r in rows)
    print(f"\n{name}: {n} quotations, Jev-like right {sum(r['jev_right'] for r in rows)}/{n} "
          f"({sum(r['jev_right'] for r in rows) / n:.1%}), LLM right {n - errs}/{n} ({(n - errs) / n:.1%}); "
          f"{secs:.0f}s -> {secs / n * 2600 / 60:.1f} min per 2,600-quotation book")
    print(f"  {'flag if disagree and conf >=':30} {'flagged':>8} {'caught':>8} {'precision':>9}")
    for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        f = [r for r in rows if not r["agree"] and r["conf"] >= t]
        caught = sum(not r["llm_right"] for r in f)
        print(f"  {t:30.2f} {len(f) / n:8.1%} {caught:3}/{errs:<4} {caught / len(f) if f else 0:9.1%}")


def load_items(args, cfg, path: str) -> tuple[dict, list[dict]]:
    gold = json.loads(Path(path).read_text(encoding="utf-8"))
    if args.book:
        store = dz.ScriptStore(cfg.data_path / "books")
        book, s = store.book(args.book), store.script(args.book)
        lines = dz.ScriptStore.public(s)["lines"]
        names = {q: (s["characters"].get(v["speaker"], {}).get("name") if v.get("speaker") else "NARRATOR")
                 for q, v in lines.items()}
        chapters, narration = book["chapters"], s.get("narration", [])

        def narrator_at(ci, pi):
            who = None
            for c, pp, w in narration:
                if (c, pp) <= (ci, pi):
                    who = w
            return s["characters"].get(who, {}).get("name") if who else None
    else:
        names = json.loads(Path(args.llm).read_text(encoding="utf-8"))[Path(path).name]
        chapters = [{"paragraphs": gold["paragraphs"],
                     "quotes": [[q["id"], q["p"], q["s"], q["e"]] for q in gold["quotes"]]}]
        narrator_at = lambda ci, pi: None  # noqa: E731
    return gold, items_for(gold, chapters, names, narrator_at, before_words=args.before_words,
                           after_words=args.after_words)


async def recheck(args, cfg) -> None:
    """Route the confident disagreements to the default LLM (thinking, wider scene); keep its new
    answer for those lines only, and score end to end."""
    flags = json.loads(Path(args.recheck).read_text(encoding="utf-8"))
    cfg.llm_mode = "gpu"
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    tot = [0, 0, 0, 0.0]
    try:
        async with aiohttp.ClientSession() as session:
            for path in args.gold:
                name = Path(path).name
                rows = [r for r in flags if r["set"] == name]
                flagged = {r["id"] for r in rows if not r["agree"] and r["conf"] >= args.min_conf}
                gold, items = load_items(args, cfg, path)
                todo = [it for it in items if it["id"] in flagged]
                t0 = time.perf_counter()
                for it in todo:
                    it["probs"] = await ask_llm(llm, session, it) or {it["llm"] if it["llm"] in it["options"]
                                                                       else NOBODY: 1.0}
                secs = time.perf_counter() - t0
                fixed = {r["id"]: r for r in evaluate(todo, gold)}
                before = sum(r["llm_right"] for r in rows)
                after = sum(fixed[r["id"]]["jev_right"] if r["id"] in fixed else r["llm_right"] for r in rows)
                n = len(rows)
                print(f"{name}: flagged {len(todo)}/{n} at conf >= {args.min_conf}; LLM {before}/{n} ({before / n:.1%})"
                      f" -> routed {after}/{n} ({after / n:.1%}); re-check {secs:.0f}s = {secs / max(1, len(todo)):.0f}s"
                      f" per line -> +{secs / n * 2600 / 60:.0f} min per 2,600-quotation book")
                for r in fixed.values():
                    print(f"   {r['id']}: gold {r['gold']!r}, first {r['llm']!r}, re-check {r['pick']!r}")
                tot = [tot[0] + n, tot[1] + before, tot[2] + after, tot[3] + secs]
    finally:
        await llm.stop()
    n = tot[0]
    print(f"all: LLM {tot[1]}/{n} ({tot[1] / n:.1%}) -> routed {tot[2]}/{n} ({tot[2] / n:.1%}); "
          f"+{tot[3] / n * 2600 / 60:.0f} min per 2,600-quotation book")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gold", nargs="+")
    p.add_argument("--backend", choices=["openjev", "jeff"], default="openjev")
    p.add_argument("--recheck", help="routing: re-check the lines flagged in these --out rows with the default LLM")
    p.add_argument("--min-conf", type=float, default=0.5, help="flag threshold for --recheck")
    p.add_argument("--book", help="book id on this PC (gold excerpts of it; LLM labels from its script)")
    p.add_argument("--llm", help="eval_attribution.py --dump output (gold chapters)")
    p.add_argument("--url", default=None)
    p.add_argument("--model", default="OpenJev-Flash-9B-Q8_0.gguf")
    p.add_argument("--out", help="write per-quotation rows here")
    p.add_argument("--before-words", type=int, default=250, help="scene words before the quotation")
    p.add_argument("--after-words", type=int, default=80)
    args = p.parse_args()
    cfg = load_config()
    if args.recheck:
        await recheck(args, cfg)
        return
    proc = None
    url = args.url or ("http://127.0.0.1:3998" if args.backend == "openjev" else "http://127.0.0.1:8000")
    if args.backend == "openjev" and not args.url:
        exe = cfg.models_path / "llama.cpp" / "cuda" / "llama-server.exe"
        proc = subprocess.Popen([str(exe), "-m", str(cfg.models_path / "llm" / args.model), "--port", "3998",
                                 "--host", "127.0.0.1", "-ngl", "99", "-c", "8192", "-np", "1"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000)
    ask = ask_openjev if args.backend == "openjev" else ask_jeff
    all_rows, total_secs = [], 0.0
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            for _ in range(300):
                try:
                    async with session.get(url + ("/health" if args.backend == "openjev" else "/healthz")) as r:
                        if r.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1)
            for path in args.gold:
                gold, items = load_items(args, cfg, path)
                t0 = time.perf_counter()
                for it in items:
                    it["probs"] = await ask(session, url, it)
                rows = evaluate(items, gold)
                secs = time.perf_counter() - t0
                total_secs += secs
                report(Path(path).name, rows, secs)
                all_rows += [{**r, "set": Path(path).name} for r in rows]
    finally:
        if proc:
            proc.terminate()
    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    report("all", all_rows, total_secs)


if __name__ == "__main__":
    asyncio.run(main())
