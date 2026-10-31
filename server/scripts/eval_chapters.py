"""Speaker attribution on whole hand-labelled chapters of library books, per eval_reasoning.py variant.

    .venv\\Scripts\\python.exe scripts\\eval_chapters.py --runs 3 context-small context-after

Each gold file (tests/gold/private/*.json, written by hand) names a stored book, a chapter, the narrator a
whole-book run would already know, the characters with the names they may be given, and every quotation's
speaker ("A|B": either is right; "*": someone unnamed, not one of the named characters or the narrator;
"NARRATOR"). The chapter is labelled from its start with an empty cast, chunk by chunk as the server does.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "scripts"))

import aiohttp  # noqa: E402

from elvenreader_server import dramatize as dz  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402
from eval_reasoning import VARIANTS, empty_script, run  # noqa: E402

GOLD = sorted((SERVER / "tests/gold/private").glob("*-ch*.json"))
GOLD = [g for g in GOLD if "book" in json.loads(g.read_text(encoding="utf-8"))]


def score(gold: dict, s: dict) -> tuple[int, list]:
    alias = {name: {dz.norm_name(a) for a in [name, *al]} for name, al in gold["characters"].items()}
    named = set().union(*alias.values())
    narrator = dz.norm_name("NARRATOR")
    ok, wrong = 0, []
    for q in gold["quotes"]:
        sp = (s["lines"].get(q["id"]) or {}).get("speaker", "MISSING")
        if sp == "MISSING":
            got, shown = set(), "MISSING"
        elif sp is None:
            got, shown = {narrator}, "NARRATOR"
        else:
            c = s["characters"][sp]
            got, shown = {dz.norm_name(c["name"]), *(dz.norm_name(a) for a in c.get("aliases", []))}, c["name"]
        right = False
        for want in q["speaker"].split("|"):
            if want == "*":
                right |= bool(got) and not got & (named | {narrator})
            elif want == "NARRATOR":
                right |= narrator in got
            else:
                right |= bool(got & alias.get(want, {dz.norm_name(want)}))
        ok += right
        if not right:
            wrong.append((q["n"], q["speaker"], shown, q["text"][:50]))
    return ok, wrong


async def evaluate(name: str, runs: int, out: dict) -> None:
    v = VARIANTS[name]
    cfg = load_config()
    cfg.llm_mode = "gpu"
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    store = dz.ScriptStore(cfg.data_path / "books")
    async with aiohttp.ClientSession() as session:
        try:
            await llm.ensure(session, v.model)
            for r in range(runs):
                for path in GOLD:
                    gold = json.loads(path.read_text(encoding="utf-8"))
                    book = store.book(gold["book"])
                    chunks = [c for c in dz.make_chunks(book, max_words=v.max_words, runs=dz.mixed_tense_runs(book))
                              if c.chapter == gold["chapter"]]
                    s = empty_script(gold.get("narrator", ""))
                    t0 = time.perf_counter()
                    await run(llm, session, book, s, chunks, v)
                    secs = time.perf_counter() - t0
                    ok, wrong = score(gold, s)
                    n = len(gold["quotes"])
                    out.setdefault(name, {}).setdefault(path.stem, []).append(
                        {"correct": ok, "quotes": n, "seconds": round(secs), "wrong": wrong})
                    print(f"{name:14} run {r + 1} {path.stem:15} {ok}/{n} = {ok / n:.1%} in {secs:.0f}s", flush=True)
        finally:
            await llm.stop()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="+")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default=str(SERVER / "data/eval/chapters.json"))
    args = p.parse_args()
    out: dict = {}
    # Alternate the variants run by run, so drift (heat, other GPU use) affects them alike.
    for r in range(args.runs):
        for name in args.variants:
            await evaluate(name, 1, out)
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for name in args.variants:
        for stem, rs in out[name].items():
            c = [x["correct"] for x in rs]
            print(f"{name:14} {stem:15} mean {sum(c) / len(c):.1f}/{rs[0]['quotes']} ({', '.join(map(str, c))}); "
                  f"{sum(x['seconds'] for x in rs) / len(rs):.0f}s per run")


if __name__ == "__main__":
    asyncio.run(main())
