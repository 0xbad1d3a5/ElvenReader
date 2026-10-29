"""What does extra LLM time buy for single-pass attribution? (reasoning, review pass, context, self-consistency)

Each variant labels the chunks that contain the gold quotations, starting from an empty cast, the
same way the server's analysis does (Sasaki and Peeps v10 excerpts: chunks of the processed book;
classics: the gold chapters alone), and reports accuracy and minutes per 100k words.

    .venv\\Scripts\\python.exe scripts\\eval_reasoning.py fast review think-1024 gptoss-medium
    .venv\\Scripts\\python.exe scripts\\eval_reasoning.py --list
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "scripts"))

import aiohttp  # noqa: E402

from androidreader_server import dramatize as dz  # noqa: E402
from androidreader_server.config import load_config  # noqa: E402
from eval_attribution import gold_book, names_for  # noqa: E402
from eval_dossier import score as score_book  # noqa: E402

BOOK = "2f145f7d38c07f00"
SASAKI = [SERVER / "tests/gold/private/sasaki10-ch8.json", SERVER / "tests/gold/private/sasaki10-ch12.json"]
CLASSICS = [SERVER / "tests/gold/pride-ch1.json", SERVER / "tests/gold/alice-ch7.json"]
GPTOSS = "gpt-oss-20b-MXFP4.gguf"
QWEN3_THINK = "Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf"


@dataclass(frozen=True)
class Variant:
    about: str
    model: str | None = None  # None = the configured llm_model
    thinking: bool = False
    effort: str = "high"
    budget: int = -1  # reasoning tokens per request
    cpu_moe: int | None = None  # None = llm_cpu_moe; 0 = the whole model on the GPU
    max_words: int = 600
    context_words: int = 0
    recent: int = 6
    review: bool = False  # second call per chunk that checks the first pass's labels
    review_thinking: bool = False
    consistency: bool = False  # a second run on shifted chunks; disagreements get a review call
    revise: int = 0  # earlier lines each call may revise (label_chunk's revise window)
    revise_unnamed: bool = False
    profiles: bool = False  # the labelling pass also fills voice profiles (profile_updates)


V = Variant
VARIANTS = {
    "fast": V("current default: Qwen3.5, thinking off"),
    "think-512": V("Qwen3.5 thinking, 512-token budget", thinking=True, budget=512),
    "think-1024": V("Qwen3.5 thinking, 1,024-token budget", thinking=True, budget=1024),
    "think-2048": V("Qwen3.5 thinking, 2,048-token budget", thinking=True, budget=2048),
    "qwen3think-1024": V("Qwen3-30B-A3B-Thinking, 1,024-token budget", model=QWEN3_THINK, thinking=True, budget=1024),
    "gptoss-low": V("gpt-oss-20b, low reasoning", model=GPTOSS, thinking=True, effort="low", cpu_moe=0),
    "gptoss-medium": V("gpt-oss-20b, medium reasoning", model=GPTOSS, thinking=True, effort="medium", cpu_moe=0),
    "gptoss-high": V("gpt-oss-20b, high reasoning", model=GPTOSS, thinking=True, effort="high", cpu_moe=0),
    "review": V("fast pass + review pass per chunk (thinking off)", review=True),
    "review-think": V("fast pass + review pass with thinking (1,024 budget)", review=True, review_thinking=True,
                      budget=1024),
    "context": V("1,200-word chunks, 600 words of preceding text, 12 recent lines", max_words=1200,
                 context_words=600, recent=12),
    "context-small": V("600-word chunks + 600 words of preceding text, 12 recent lines", context_words=600, recent=12),
    "context-review": V("context-small + review pass per chunk", context_words=600, recent=12, review=True),
    "context-big": V("600-word chunks + 1,200 words of preceding text, 20 recent lines", context_words=1200, recent=20),
    "consistency": V("two runs (600/420-word chunks); disagreeing chunks reviewed", consistency=True),
    "context-profiles": V("context-small + voice profiles noted while labelling", context_words=600, recent=12,
                          profiles=True),
    "rev12": V("context-small + may revise the last 12 earlier lines", context_words=600, recent=12, revise=12),
    "rev30": V("context-small + may revise the last 30 earlier lines", context_words=600, recent=12, revise=30),
    "rev60": V("context-small + may revise the last 60 earlier lines (about the previous chunk or two)",
               context_words=600, recent=12, revise=60),
    "rev-unnamed": V("context-small + may revise earlier lines given to unnamed speakers (last 60)",
                     context_words=600, recent=12, revise=60, revise_unnamed=True),
    "rev30-think": V("rev30 with Qwen3.5 thinking, 1,024-token budget", context_words=600, recent=12, revise=30,
                     thinking=True, budget=1024),
}


def speaker_name(script: dict, qid: str) -> str:
    sp = (script["lines"].get(qid) or {}).get("speaker")
    return script["characters"].get(sp, {}).get("name", "NARRATOR") if sp else "NARRATOR"


def score_classic(gold: dict, script: dict) -> dict:
    """eval_attribution.py's scoring ("a|b" alternatives, "*" = any unnamed speaker)."""
    ok, wrong = 0, []
    for q in gold["quotes"]:
        line = script["lines"].get(q["id"])
        sp = line.get("speaker") if line else "MISSING"
        if sp and sp != "MISSING":
            c = script["characters"][sp]
            predicted = {dz.norm_name(c["name"]), *(dz.norm_name(a) for a in c.get("aliases", []))}
        else:
            predicted = {dz.norm_name("NARRATOR")} if sp is None else set()
        if q["speaker"] == "*":
            named = set().union(*(names_for(gold, n) for n in gold.get("characters", {})))
            right = bool(predicted) and not predicted & (named | {dz.norm_name("NARRATOR")})
        else:
            right = bool(predicted & names_for(gold, q["speaker"]))
        ok += right
        if not right:
            wrong.append((q["id"], q["speaker"], speaker_name(script, q["id"])))
    return {"correct": ok, "quotes": len(gold["quotes"]), "wrong": wrong}


def sasaki_judge(gold: dict):
    """quote id, character id -> right? (None: not a scored gold quote)."""
    from eval_dossier import _norm

    alias = {_norm(a): name for name, al in gold["characters"].items() for a in [name, *al]}
    want = {q["id"]: q["speaker"] for q in gold["quotes"] if q["speaker"] != "*"}

    def judge(s: dict, qid: str, cid):
        if qid not in want:
            return None
        if cid is None:
            got = "NARRATOR"
        else:
            c = s["characters"].get(cid, {"name": str(cid)})
            names = [c["name"], *c.get("aliases", [])]
            got = next((alias[_norm(x)] for x in names if _norm(x) in alias), names[0])
        return got == want[qid]
    return judge


def classic_judge(gold: dict):
    quotes = {q["id"]: q for q in gold["quotes"]}

    def judge(s: dict, qid: str, cid):
        q = quotes.get(qid)
        if q is None:
            return None
        one = {"characters": s["characters"], "lines": {qid: {"speaker": cid}}}
        return score_classic({**gold, "quotes": [q]}, one)["correct"] == 1
    return judge


def tally_revisions(s: dict, start: int, judges: list, out: dict, key: str = "revisions") -> None:
    """Counts the revisions logged since `start`: fixed an error, broke a right label, or neither."""
    for ev in s.get("revisions", [])[start:]:
        for judge in judges:
            before, after = judge(s, ev["id"], ev["from"]), judge(s, ev["id"], ev["to"])
            if before is None:
                continue
            kind = "fixed" if after and not before else "broke" if before and not after else "neutral"
            out.setdefault(key, []).append({**ev, "kind": kind})
            break
        else:
            out.setdefault(key, []).append({**ev, "kind": "ungraded"})


def empty_script(narrator: str = "") -> dict:
    return {"characters": {}, "lines": {}, "corrections": {}, "chunks_done": [], "narrator": narrator,
            "narrator_by_tense": {}, "last_tense": None}


async def label(llm, session, book: dict, s: dict, chunk, v: Variant, proposed=None, thinking=None) -> None:
    """One label_chunk call applied like the server's Analyzer (narrator learnt per tense)."""
    by_tense = s.setdefault("narrator_by_tense", {})
    result, local = await dz.label_chunk(
        llm, session, book, s, chunk, thinking=v.thinking if thinking is None else thinking, model=v.model,
        effort=v.effort, context_words=v.context_words, recent=v.recent, proposed=proposed,
        revise=0 if proposed else v.revise, revise_unnamed=v.revise_unnamed, profiles=v.profiles)
    if chunk.tense and by_tense.get(chunk.tense):
        result["narrator"] = by_tense[chunk.tense]
    dz.apply_chunk_result(s, result, local)
    if chunk.tense:
        s["last_tense"] = chunk.tense
        if chunk.tense not in by_tense and dz.find_character(s, s.get("narrator") or ""):
            by_tense[chunk.tense] = s["narrator"]


async def run(llm, session, book: dict, s: dict, chunks: list, v: Variant) -> None:
    for chunk in chunks:
        await label(llm, session, book, s, chunk, v)
        if v.review:
            _, local = dz.render_chunk(book, chunk)
            proposed = {n: speaker_name(s, ids[0]) for n, ids in local.items()}
            await label(llm, session, book, s, chunk, v, proposed=proposed, thinking=v.review_thinking)


async def run_consistency(llm, session, book: dict, s: dict, chunks: list, alt_chunks: list, v: Variant,
                          alt: dict) -> int:
    """Run A on `chunks` into `s`, run B on `alt_chunks` into `alt`; then review the A chunks where the
    two disagree, showing both readings for the disputed lines. Returns the number of reviewed chunks."""
    await run(llm, session, book, s, chunks, v)
    await run(llm, session, book, alt, alt_chunks, v)
    reviewed = 0
    for chunk in chunks:
        _, local = dz.render_chunk(book, chunk)
        proposed, disputed = {}, False
        for n, ids in local.items():
            a, b = speaker_name(s, ids[0]), speaker_name(alt, ids[0])
            if ids[0] in alt["lines"] and dz.norm_name(a) != dz.norm_name(b):
                proposed[n], disputed = f"{a} or {b}? (two readings disagree)", True
            else:
                proposed[n] = a
        if disputed:
            reviewed += 1
            await label(llm, session, book, s, chunk, v, proposed=proposed)
    return reviewed


def words_of(book: dict, chunks: list) -> int:
    return sum(len(book["chapters"][c.chapter]["paragraphs"][p].split()) for c in chunks for p in c.paragraphs)


async def evaluate(name: str, v: Variant, sets: list[str]) -> dict:
    cfg = load_config()
    cfg.llm_mode = "gpu"
    cfg.llm_reasoning_budget = v.budget
    if v.cpu_moe is not None:
        cfg.llm_cpu_moe = v.cpu_moe
    if v.thinking or v.review_thinking:  # one context size, so llama.cpp isn't restarted between calls
        cfg.llm_context = cfg.llm_dossier_context
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    out = {"variant": name, "about": v.about}
    async with aiohttp.ClientSession() as session:
        try:
            await llm.ensure(session, v.model)
            if "sasaki" in sets:
                book = dz.ScriptStore(cfg.data_path / "books").book(BOOK)
                runs = dz.mixed_tense_runs(book)
                chunks = dz.make_chunks(book, max_words=v.max_words, runs=runs)
                alt_all = dz.make_chunks(book, max_words=420, runs=runs) if v.consistency else []
                s, alt = empty_script(), empty_script()
                ok = n = words = 0
                secs = 0.0
                wrong = []
                for path in SASAKI:
                    gold = json.loads(path.read_text(encoding="utf-8"))
                    ids = {q["id"] for q in gold["quotes"]}
                    todo = [c for c in chunks if any(q[0] in ids for q in c.quotes)]
                    t0 = time.perf_counter()
                    if v.consistency:
                        alt_todo = [c for c in alt_all if any(q[0] in ids for q in c.quotes)]
                        out.setdefault("reviewed", 0)
                        out["reviewed"] += await run_consistency(llm, session, book, s, todo, alt_todo, v, alt)
                    else:
                        await run(llm, session, book, s, todo, v)
                    secs += time.perf_counter() - t0
                    words += words_of(book, todo)
                    r = score_book(gold, s)
                    out[path.stem] = f"{r['correct']}/{r['quotes']}"
                    if path.stem == "sasaki10-ch8":  # the chapter-start chunk where Nick is still unnamed
                        judge = sasaki_judge(gold)
                        nick = [q["id"] for q in gold["quotes"] if q["speaker"] == "Nick" and q["p"] < 36]
                        right = sum(bool(judge(s, q, (s["lines"].get(q) or {}).get("speaker"))) for q in nick)
                        out["nick_start"] = f"{right}/{len(nick)}"
                    ok, n = ok + r["correct"], n + r["quotes"]
                    wrong += r["wrong"]
                out["sasaki"] = [ok, n, round(secs / words * 100000 / 60, 1)]
                if v.profiles:
                    out["profiles"] = {c["name"]: c["profile"] for c in s["characters"].values() if c.get("profile")}
                    print(f"{name:16} profiles for {len(out['profiles'])} characters", flush=True)
                out["sasaki_wrong"] = wrong
                tally_revisions(s, 0, [sasaki_judge(json.loads(p_.read_text(encoding="utf-8"))) for p_ in SASAKI], out)
                kinds = [r_["kind"] for r_ in out.get("revisions", [])]
                print(f"{name:16} Sasaki   {ok}/{n} = {ok / n:.1%} ({out['sasaki10-ch8']}, {out['sasaki10-ch12']}); "
                      f"{secs:.0f}s for {words} words -> {out['sasaki'][2]:.0f} min per 100k words; Nick at chapter "
                      f"start {out.get('nick_start')}; revisions {len(kinds)}: fixed {kinds.count('fixed')}, broke "
                      f"{kinds.count('broke')}, neutral {kinds.count('neutral')}, ungraded {kinds.count('ungraded')}",
                      flush=True)
            if "classics" in sets:
                ok = n = words = 0
                secs = 0.0
                wrong = []
                for path in CLASSICS:
                    gold = json.loads(path.read_text(encoding="utf-8"))
                    book = gold_book(gold)
                    chunks = dz.make_chunks(book, max_words=v.max_words)
                    s = empty_script(gold.get("previous_narrator", ""))
                    t0 = time.perf_counter()
                    if v.consistency:
                        await run_consistency(llm, session, book, s, chunks, dz.make_chunks(book, max_words=420), v,
                                              empty_script(gold.get("previous_narrator", "")))
                    else:
                        await run(llm, session, book, s, chunks, v)
                    secs += time.perf_counter() - t0
                    words += words_of(book, chunks)
                    r = score_classic(gold, s)
                    tally_revisions(s, 0, [classic_judge(gold)], out, "classics_revisions")
                    ok, n = ok + r["correct"], n + r["quotes"]
                    wrong += r["wrong"]
                out["classics"] = [ok, n, round(secs / words * 100000 / 60, 1)]
                out["classics_wrong"] = wrong
                kinds = [r_["kind"] for r_ in out.get("classics_revisions", [])]
                print(f"{name:16} classics {ok}/{n} = {ok / n:.1%}; {secs:.0f}s for {words} words -> "
                      f"{out['classics'][2]:.0f} min per 100k words; revisions {len(kinds)}: fixed "
                      f"{kinds.count('fixed')}, broke {kinds.count('broke')}", flush=True)
        finally:
            await llm.stop()
    return out


async def nick_probe(names: list[str], runs: int) -> None:
    """The whole-book failure the gold-chunk harness misses: in Sasaki and Peeps v10 the prison chapter opens
    with an unnamed man who names himself (Nick) a page later. Re-label that chapter's first chunks the way a
    whole-book run does (full cast, earlier chapters labelled) and count the 12 gold lines he speaks before."""
    from eval_dossier import _norm  # noqa: F401  (sasaki_judge)

    cfg = load_config()
    cfg.llm_mode = "gpu"
    store = dz.ScriptStore(cfg.data_path / "books")
    book, base = store.book(BOOK), store.script(BOOK)
    gold = json.loads(SASAKI[0].read_text(encoding="utf-8"))
    judge = sasaki_judge(gold)
    nick = [q["id"] for q in gold["quotes"] if q["speaker"] == "Nick" and q["p"] < 36]
    chunks = [c for c in dz.make_chunks(book, runs=dz.mixed_tense_runs(book)) if c.chapter == 4][:3]
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    async with aiohttp.ClientSession() as session:
        try:
            for name in names:
                v = VARIANTS[name]
                for run in range(runs):
                    s = json.loads(json.dumps(base))
                    s["lines"] = {q: line for q, line in s["lines"].items() if not q.startswith("c4p")}
                    s["revisions"], s["narrator"], s["last_tense"] = [], "Sasaki", None
                    t0 = time.perf_counter()
                    for chunk in chunks:
                        await label(llm, session, book, s, chunk, v)
                    got = sum(bool(judge(s, q, (s["lines"].get(q) or {}).get("speaker"))) for q in nick)
                    who = sorted({speaker_name(s, q) for q in nick})
                    revs = [(e["id"], s["characters"].get(e["from"], {}).get("name"), s["characters"].get(e["to"], {}).get("name"))
                            for e in s["revisions"]]
                    print(f"{name:12} run {run + 1}: Nick before naming {got}/{len(nick)} (labelled {who}); "
                          f"{time.perf_counter() - t0:.0f}s; revisions {revs}", flush=True)
        finally:
            await llm.stop()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--sets", default="sasaki,classics")
    p.add_argument("--out", default=str(SERVER / "data/eval/reasoning.json"))
    p.add_argument("--nick-probe", type=int, metavar="RUNS", help="run nick_probe for the variants instead")
    args = p.parse_args()
    if args.nick_probe:
        await nick_probe(args.variants, args.nick_probe)
        return
    if args.list or not args.variants:
        for k, v in VARIANTS.items():
            print(f"{k:16} {v.about}")
        return
    path = Path(args.out)
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in args.variants:
        base, _, budget = name.split("#")[0].partition("@")  # review-think@512 overrides the budget; fast#2 repeats
        v = VARIANTS[base]
        if budget:
            v = replace(v, budget=int(budget))
        try:
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            results[name] = await evaluate(name, v, args.sets.split(","))
            # Speech requested meanwhile (someone listening) shares the GPU and spoils the timing.
            log = SERVER / "data" / "server.log"
            tts = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
                   if ln[:19] >= started and " tts " in ln] if log.exists() else []
            if tts:
                results[name]["tts_during_run"] = len(tts)
                print(f"{name}: {len(tts)} speech requests during this run; its timing is not comparable", flush=True)
        except dz.LlmError as exc:
            print(f"{name}: failed: {exc}", flush=True)
            results[name] = {"variant": name, "error": str(exc)}
        path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
