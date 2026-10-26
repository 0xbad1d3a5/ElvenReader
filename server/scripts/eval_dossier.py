"""Evaluate the two-pass (dossier) pipeline on a processed book.

    .venv\\Scripts\\python.exe scripts\\eval_dossier.py --dossier-model Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf ^
        --roster tests\\gold\\sasaki10-roster.json tests\\gold\\private\\sasaki10-ch8.json tests\\gold\\private\\sasaki10-ch12.json

Pass 1 reads the whole book (as the server does) with --dossier-model and is cached in
data/eval/<model>/; the dossier is checked against the roster gold (apparent gender, age,
form). Pass 2 then labels the chunks that contain the gold quotations with --label-model
(add --label-thinking to let it reason) and is scored. --score-script scores an existing
script.json (e.g. the old single-pass pipeline) on the same gold files instead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from androidreader_server import dossier as ds  # noqa: E402
from androidreader_server import dramatize as dz  # noqa: E402
from androidreader_server.config import load_config  # noqa: E402


def _norm(x: str) -> str:
    return re.sub(r"[^a-z]", "", (x or "").lower().replace("the ", ""))


def score(gold: dict, script: dict) -> dict:
    alias = {_norm(a): name for name, al in gold["characters"].items() for a in [name, *al]}
    chars = script["characters"]
    lines = dz.ScriptStore.public(script)["lines"] if "corrections" in script else script["lines"]
    ok = n = 0
    wrong = []
    for q in gold["quotes"]:
        if q["speaker"] == "*":
            continue
        cid = (lines.get(q["id"]) or {}).get("speaker")
        if cid is None:
            got = "NARRATOR"
        else:
            names = [chars[cid]["name"], *chars[cid].get("aliases", [])] if cid in chars else [str(cid)]
            got = next((alias[_norm(x)] for x in names if _norm(x) in alias), names[0])
        n += 1
        if got == q["speaker"]:
            ok += 1
        else:
            wrong.append((q["id"], q["speaker"], got))
    return {"correct": ok, "quotes": n, "wrong": wrong}


def check_roster(roster: dict, script: dict) -> tuple[int, int, list[str]]:
    ok = total = 0
    report = []
    for name, want in roster["characters"].items():
        cid = dz.find_character(script, name) or dz.find_character(script, name.replace("Ms. ", "").replace("Miss ", ""))
        c = script["characters"].get(cid or "", {})
        d = ds.effective(c)
        checks = {"gender": d.get("gender", c.get("gender")) == want["gender"],
                  "age": d.get("apparent_age", c.get("age")) in want["age"]}
        if "form" in want:
            checks["form"] = bool(re.search(want["form"], f"{d.get('form', '')} {d.get('true_nature', '')}", re.I))
        if "nature" in want:
            checks["nature"] = bool(re.search(want["nature"], d.get("true_nature", ""), re.I))
        if "introduced" in want:
            checks["introduced"] = bool(d.get("introduced_as"))
        ok += sum(checks.values())
        total += len(checks)
        bad = [k for k, v in checks.items() if not v]
        report.append(f"{'ok ' if not bad else 'BAD'} {name:18} {d.get('gender', c.get('gender'))}/"
                      f"{d.get('apparent_age', c.get('age'))} form={d.get('form', '')!r} true={d.get('true_nature', '')!r}"
                      f" first={d.get('introduced_as', '')!r}" + (f"  wrong: {', '.join(bad)}" if bad else ""))
    return ok, total, report


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gold", nargs="*")
    p.add_argument("--roster")
    p.add_argument("--dossier-model")
    p.add_argument("--label-model")
    p.add_argument("--label-thinking", action="store_true")
    p.add_argument("--cpu", action="store_true", help="run the LLM on the CPU (default GPU)")
    p.add_argument("--effort", help="reasoning effort for pass 1 (low/medium/high)")
    p.add_argument("--words", type=int, help="words per pass-1 passage")
    p.add_argument("--cpu-moe", type=int, help="layers whose experts stay in RAM")
    p.add_argument("--tag", default="", help="suffix for the cache folder")
    p.add_argument("--seed-cast", help="start from this script.json's cast (names, aliases, gender, age, voices; "
                                       "no line labels), as series memory or a re-run would")
    p.add_argument("--passages", type=int, help="read only the first N passages in pass 1 (speed probe)")
    p.add_argument("--no-dossier", action="store_true", help="pass 2 without the dossier (old pipeline)")
    p.add_argument("--score-script", help="score this script.json instead of running anything")
    args = p.parse_args()
    golds = [json.loads(Path(g).read_text(encoding="utf-8")) for g in args.gold]
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8")) if args.roster else None
    if args.score_script:
        script = json.loads(Path(args.score_script).read_text(encoding="utf-8"))
        for g in golds:
            r = score(g, script)
            print(f"{g['source'][:60]}: {r['correct']}/{r['quotes']} = {r['correct'] / r['quotes']:.1%}")
        if roster:
            ok, total, rep = check_roster(roster, script)
            print("\n".join(rep), f"\nroster checks {ok}/{total}")
        return

    cfg = load_config()
    cfg.llm_mode = "cpu" if args.cpu else "gpu"
    if args.effort:
        cfg.llm_dossier_effort = args.effort
    if args.words:
        cfg.llm_dossier_words = args.words
    cpu_moe_default = cfg.llm_cpu_moe
    if args.cpu_moe is not None:
        cfg.llm_cpu_moe = args.cpu_moe
    book_id = (roster or {}).get("book_id") or "2f145f7d38c07f00"
    src = dz.ScriptStore(cfg.data_path / "books")
    book = src.book(book_id)
    model = args.dossier_model or cfg.llm_dossier_model
    work = cfg.data_path / "eval" / (Path(model).stem + args.tag)
    store = dz.ScriptStore(work / "books")
    store.dir(book_id).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src.dir(book_id) / "book.json", store.dir(book_id) / "book.json")
    if args.seed_cast and not (store.dir(book_id) / "script.json").exists():
        seed = json.loads(Path(args.seed_cast).read_text(encoding="utf-8"))
        s = store.script(book_id)
        for cid, c in seed["characters"].items():
            s["characters"][cid] = {k: c.get(k) for k in ("name", "aliases", "gender", "age", "voice",
                                                          "dossier_user", "user") if c.get(k) is not None}
            s["characters"][cid].setdefault("aliases", [])
        store.save_script(book_id, s)
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = not args.cpu
    if args.passages:
        full = ds.dossier_chunks
        ds.dossier_chunks = lambda *a, **k: full(*a, **k)[:args.passages]

    async def gpu_step(make):
        return await make()

    async def progress(s, frac, detail):
        store.save_script(book_id, s)
        print(f"\r  pass 1: {frac:5.1%} {detail[:60]:60}", end="", flush=True)

    async with aiohttp.ClientSession() as session:
        try:
            t0 = time.perf_counter()
            await ds.build(llm, session, book, store, book_id, model, gpu_step, progress)
            s = store.script(book_id)
            st = s["dossier_state"]
            print(f"\npass 1 ({model}, effort {cfg.llm_dossier_effort}, {cfg.llm_dossier_words} words/passage): "
                  f"{len(ds.dossier_chunks(book, cfg.llm_dossier_words))} passages, {st.get('tokens', 0)} tokens, "
                  f"{st.get('seconds', 0) / 60:.1f} min (this run {(time.perf_counter() - t0) / 60:.1f} min)")
            dz.Analyzer.narrators_from_dossier(s)
            print("narrators:", ds.narrators(s), "by tense:", s.get("narrator_by_tense"))
            if roster:
                ok, total, rep = check_roster(roster, s)
                print("\n".join(rep), f"\nroster checks {ok}/{total}")
            label_model = args.label_model or cfg.llm_model
            if args.no_dossier:  # the old single-pass pipeline on the same chunks
                s = {"characters": {}, "lines": {}, "corrections": {}, "chunks_done": []}
            cfg.llm_cpu_moe = cpu_moe_default  # --cpu-moe is for the pass-1 model
            chunks = dz.make_chunks(book, runs=dz.mixed_tense_runs(book))
            s["lines"], s["narrator_by_tense"], s["narrator"], s["last_tense"] = {}, {}, "", None
            for g in golds:
                ids = {q["id"] for q in g["quotes"]}
                todo = [c for c in chunks if any(q[0] in ids for q in c.quotes)]
                t1 = time.perf_counter()
                n_quotes = 0
                for chunk in todo:
                    by_tense = s.setdefault("narrator_by_tense", {})
                    result, local = await dz.label_chunk(llm, session, book, s, chunk, thinking=args.label_thinking,
                                                         model=label_model)
                    if chunk.tense and by_tense.get(chunk.tense):
                        result["narrator"] = by_tense[chunk.tense]
                    dz.apply_chunk_result(s, result, local)
                    if chunk.tense:  # same narrator learning as the server's Analyzer
                        s["last_tense"] = chunk.tense
                        if chunk.tense not in by_tense and dz.find_character(s, s.get("narrator") or ""):
                            by_tense[chunk.tense] = s["narrator"]
                    n_quotes += len(chunk.quotes)
                secs = time.perf_counter() - t1
                r = score(g, s)
                print(f"\npass 2 ({Path(label_model).stem}{', thinking' if args.label_thinking else ''}) on "
                      f"{g['source'][:70]}: {r['correct']}/{r['quotes']} = {r['correct'] / r['quotes']:.1%}; "
                      f"{secs:.0f}s for {n_quotes} quotes -> {secs / max(1, n_quotes) * 2599 / 60:.0f} min per 2,600-quote book")
                for w in r["wrong"]:
                    print("   wrong:", w)
            store.save_script(book_id, s)
        finally:
            await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())
