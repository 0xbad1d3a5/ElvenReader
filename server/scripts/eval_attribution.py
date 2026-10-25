"""Measure speaker-attribution accuracy of the dramatization pipeline on hand-labelled chapters.

    .venv\\Scripts\\python.exe scripts\\eval_attribution.py [--model FILE.gguf] [--gpu-layers N] tests\\gold\\*.json

Runs the same chunking, prompts and JSON schema as the server, against a local
llama.cpp server it starts itself, then compares speakers with the gold labels.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from androidreader_server import dramatize as dz  # noqa: E402
from androidreader_server.config import load_config  # noqa: E402


def gold_book(gold: dict) -> dict:
    quotes = [[q["id"], q["p"], q["s"], q["e"]] for q in gold["quotes"]]
    return {"title": gold.get("title", gold["source"]), "author": gold.get("author", ""),
            "chapters": [{"title": gold["paragraphs"][0][:60], "paragraphs": gold["paragraphs"], "quotes": quotes}]}


def names_for(gold: dict, label: str) -> set[str]:
    out = set()
    for alt in label.split("|"):
        out.add(dz.norm_name(alt))
        for a in gold.get("characters", {}).get(alt, []):
            out.add(dz.norm_name(a))
    return out


async def evaluate(path: Path, llm: dz.LocalLlm, session) -> dict:
    gold = json.loads(path.read_text(encoding="utf-8"))
    book = gold_book(gold)
    script = {"characters": {}, "lines": {}}
    t0 = time.perf_counter()
    prompt_tps, gen_tps, tokens = [], [], 0
    for chunk in dz.make_chunks(book):
        result, local = await dz.label_chunk(llm, session, book, script, chunk)
        dz.apply_chunk_result(script, result, local)
        t = result.get("_timings", {})
        if t.get("prompt_per_second"):
            prompt_tps.append(t["prompt_per_second"])
        if t.get("predicted_per_second"):
            gen_tps.append(t["predicted_per_second"])
        tokens += int(result.get("_usage", {}).get("total_tokens", 0))
    elapsed = time.perf_counter() - t0
    correct, wrong = 0, []
    for q in gold["quotes"]:
        line = script["lines"].get(q["id"])
        sp = line.get("speaker") if line else "MISSING"
        if sp and sp != "MISSING":
            c = script["characters"][sp]
            predicted = {dz.norm_name(c["name"]), *(dz.norm_name(a) for a in c.get("aliases", []))}
        else:
            predicted = {dz.norm_name("NARRATOR")} if sp is None else set()
        if predicted & names_for(gold, q["speaker"]):
            correct += 1
        else:
            wrong.append((q["id"], q["text"][:50], q["speaker"], next(iter(predicted), "?")))
    deliveries = [(q["text"][:45], script["lines"].get(q["id"], {}).get("delivery")) for q in gold["quotes"]]
    return {
        "file": path.name, "quotes": len(gold["quotes"]), "correct": correct,
        "accuracy": round(correct / len(gold["quotes"]), 3), "seconds": round(elapsed, 1),
        "prompt_tps": round(sum(prompt_tps) / len(prompt_tps), 1) if prompt_tps else None,
        "gen_tps": round(sum(gen_tps) / len(gen_tps), 1) if gen_tps else None,
        "tokens": tokens, "wrong": wrong,
        "deliveries": [d for d in deliveries if d[1]][:20],
        "words": sum(len(p.split()) for p in gold["paragraphs"]),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", nargs="+")
    parser.add_argument("--model", help="GGUF file under <models_dir>/llm")
    parser.add_argument("--gpu", action="store_true", help="offload to the GPU (default: CPU only)")
    parser.add_argument("--cpu-moe", type=int, default=None, help="layers whose experts stay in RAM with --gpu")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.model:
        cfg.llm_model = args.model
    cfg.llm_mode = "gpu" if args.gpu else "cpu"
    if args.cpu_moe is not None:
        cfg.llm_cpu_moe = args.cpu_moe
    if args.threads:
        cfg.llm_threads = args.threads
    llm = dz.LocalLlm(cfg, None)
    results = []
    async with aiohttp.ClientSession() as session:
        try:
            for g in args.gold:
                r = await evaluate(Path(g), llm, session)
                results.append(r)
                print(json.dumps({k: v for k, v in r.items() if k not in ("wrong", "deliveries")}), flush=True)
                for w in r["wrong"]:
                    print("   wrong:", w)
                for d in r["deliveries"]:
                    print("   delivery:", d)
        finally:
            await llm.stop()
    total = sum(r["quotes"] for r in results)
    ok = sum(r["correct"] for r in results)
    words = sum(r["words"] for r in results)
    secs = sum(r["seconds"] for r in results)
    print(f"\nmodel={Path(cfg.llm_model).name} mode={cfg.llm_mode} cpu_moe={cfg.llm_cpu_moe} accuracy={ok}/{total}={ok / total:.1%} "
          f"time={secs:.0f}s for {words} words -> {secs / words * 100000 / 60:.0f} min per 100k-word book")


if __name__ == "__main__":
    asyncio.run(main())
