"""Can a fast quotation-attribution model flag the LLM's mistakes? (ModernBookNLP as a second opinion)

Two steps, because ModernBookNLP needs its own environment
(D:\\Models\\AndroidReader\\venvs\\booknlp, see the README):

    # 1. ModernBookNLP's speaker for every gold quotation (run with the booknlp venv's python,
    #    from a folder that has an "examples" subfolder; PYTHONUTF8=1)
    python scripts\\eval_flagger.py booknlp --out data\\eval\\booknlp.json tests\\gold\\pride-ch1.json ...
    #    a gold excerpt of a book on this PC is run over the whole book: --book <book id>
    # 2. score it against the gold labels and the LLM's labels (server venv)
    python scripts\\eval_flagger.py score --booknlp data\\eval\\booknlp.json --llm data\\eval\\llm.json ...

The LLM labels come from eval_attribution.py --dump (gold chapters) or a book's script.json
(--script, for gold excerpts of a book processed on this PC).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent


def norm(name: str) -> str:
    s = (name or "").lower().replace(".", "").strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return " ".join(s.split())


# ----------------------------------------------------------------- step 1: run ModernBookNLP


def book_text(chapters: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """Paragraphs joined by blank lines; [(start offset, chapter, paragraph)] for each paragraph."""
    parts, starts, pos = [], [], 0
    for ci, ch in enumerate(chapters):
        for pi, p in enumerate(ch["paragraphs"]):
            starts.append((pos, ci, pi))
            parts.append(p)
            pos += len(p) + 2
    return "\n\n".join(parts), starts


def run_booknlp(text: str, work: Path, name: str):
    from booknlp.english.english_booknlp import EnglishBookNLP

    work.mkdir(parents=True, exist_ok=True)
    (work / f"{name}.txt").write_text(text, encoding="utf-8")
    nlp = EnglishBookNLP({"pipeline": "entity,quote,coref", "model": "big", "modern_qa": True, "direct_qa": False})
    t0 = time.perf_counter()
    nlp.process(str(work / f"{name}.txt"), str(work / name), name)
    return time.perf_counter() - t0


def read_booknlp(work: Path, name: str):
    out = work / name
    tokens = {}
    lines = (out / f"{name}.tokens").read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    tid, onset = head.index("token_ID_within_document"), head.index("byte_onset")
    for line in lines[1:]:
        f = line.split("\t")
        tokens[int(f[tid])] = int(f[onset])  # token id -> character offset
    book = json.loads((out / f"{name}.book").read_text(encoding="utf-8"))
    names = {}
    for c in book.get("characters", []):
        m = c.get("mentions", {})
        best = (m.get("proper") or m.get("common") or m.get("pronoun") or [{"n": f"#{c['id']}"}])[0]["n"]
        names[c["id"]] = best
    quotes = []
    for line in (out / f"{name}.quotes").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split("\t")
        quotes.append((tokens[int(f[0])], int(f[5]), f[4]))  # (offset, character id, mention phrase)
    return quotes, names


FIRST_PERSON = {"i", "me", "my", "myself"}


def cmd_booknlp(args) -> None:
    work = Path(args.work)
    preds, timing = {}, {}
    for path in args.gold:
        gold = json.loads(Path(path).read_text(encoding="utf-8"))
        if args.book:
            book = json.loads((SERVER / "data" / "books" / args.book / "book.json").read_text(encoding="utf-8"))
            script = json.loads((SERVER / "data" / "books" / args.book / "script.json").read_text(encoding="utf-8"))
            chapters, narration, name = book["chapters"], script.get("narration", []), f"book-{args.book}"
            chars = script.get("characters", {})
        else:
            chapters = [{"paragraphs": gold["paragraphs"], "quotes": [[q["id"], q["p"], q["s"], q["e"]] for q in gold["quotes"]]}]
            narration, name, chars = [], Path(path).stem, {}
        text, starts = book_text(chapters)
        if not (work / name / f"{name}.quotes").exists():
            timing[name] = run_booknlp(text, work, name)
        bq, names = read_booknlp(work, name)

        def where(offset):
            lo, hi = 0, len(starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if starts[mid][0] <= offset:
                    lo = mid
                else:
                    hi = mid - 1
            return starts[lo]

        def narrator(ci, pi):
            who = None
            for c, p, w in narration:
                if (c, p) <= (ci, pi):
                    who = w
            return chars.get(who, {}).get("name") if who else None

        found = {}
        for offset, cid, phrase in bq:
            start, ci, pi = where(offset)
            found[(ci, pi, offset - start)] = names.get(cid, phrase) if cid >= 0 else phrase
        ids = {q["id"] for q in gold["quotes"]}
        out = {}
        for ci, ch in enumerate(chapters):
            for qid, p, s, e in ch.get("quotes", []):
                if qid not in ids:
                    continue
                hit = next((v for (c, pp, o), v in found.items() if c == ci and pp == p and s <= o < e), None)
                if hit is not None and norm(hit) in FIRST_PERSON:
                    hit = narrator(ci, p) or hit
                out[qid] = hit
        preds[Path(path).name] = out
    Path(args.out).write_text(json.dumps({"predictions": preds, "seconds": timing}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("seconds:", {k: round(v, 1) for k, v in timing.items()})


# ----------------------------------------------------------------- step 2: score


def cmd_score(args) -> None:
    sys.path.insert(0, str(SERVER))
    bn = json.loads(Path(args.booknlp).read_text(encoding="utf-8"))["predictions"]
    llm =  json.loads(Path(args.llm).read_text(encoding="utf-8")) if args.llm else {}
    if args.script:
        s = json.loads(Path(args.script).read_text(encoding="utf-8"))
        lines = dict(s["lines"])
        lines.update({k: {**lines.get(k, {}), **v} for k, v in s.get("corrections", {}).items()})
        by_script = {q: (s["characters"].get(v.get("speaker"), {}).get("name") if v.get("speaker") else "NARRATOR")
                     for q, v in lines.items()}
    rows = []
    for path in args.gold:
        gold = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(path).name
        alias = {}
        for canon, al in gold.get("characters", {}).items():
            for a in [canon, *al]:
                alias[norm(a)] = canon
        canon = lambda x: alias.get(norm(x or ""), norm(x or "") or "?")  # noqa: E731
        if name in llm:
            lp = llm[name]
        elif args.script:
            lp = {q["id"]: by_script.get(q["id"]) for q in gold["quotes"]}
        else:
            sys.exit(f"no LLM labels for {name}")
        n = b_ok = l_ok = flags = err = err_flagged = 0
        for q in gold["quotes"]:
            if q["speaker"] == "*":
                continue
            truth = {canon(t) for t in q["speaker"].split("|")}
            b, ll = bn[name].get(q["id"]), lp.get(q["id"])
            b_right, l_right = canon(b) in truth if b else False, canon(ll) in truth
            disagree = b is None or canon(b) != canon(ll)
            n += 1
            b_ok += b_right
            l_ok += l_right
            flags += disagree
            err += not l_right
            err_flagged += (not l_right) and disagree
        rows.append((name, n, b_ok, l_ok, flags, err, err_flagged))
    print(f"{'gold set':28} {'quotes':>6} {'BookNLP':>9} {'LLM':>9} {'flagged':>9} {'LLM errors':>10} {'caught':>7} {'precision':>9}")
    tot = [0] * 6
    for name, n, b_ok, l_ok, flags, err, ef in rows:
        print(f"{name:28} {n:6} {b_ok / n:9.1%} {l_ok / n:9.1%} {flags / n:9.1%} {err:10} {ef:3}/{err:<3} {ef / flags if flags else 0:9.1%}")
        tot = [a + b for a, b in zip(tot, (n, b_ok, l_ok, flags, err, ef))]
    n, b_ok, l_ok, flags, err, ef = tot
    print(f"{'all':28} {n:6} {b_ok / n:9.1%} {l_ok / n:9.1%} {flags / n:9.1%} {err:10} {ef:3}/{err:<3} {ef / flags if flags else 0:9.1%}")


def cmd_route(args) -> None:
    """Re-check the lines ModernBookNLP disagrees on: re-label their passages with the LLM reasoning
    first (thinking on), keep the new answer only for the flagged lines, and score before/after."""
    import asyncio
    import copy

    sys.path.insert(0, str(SERVER))
    sys.path.insert(0, str(SERVER / "scripts"))
    import aiohttp

    from elvenreader_server import dramatize as dz
    from elvenreader_server.config import load_config
    from eval_dossier import score

    cfg = load_config()
    cfg.llm_mode = "gpu"
    store = dz.ScriptStore(cfg.data_path / "books")
    book, base = store.book(args.book), store.script(args.book)
    bn = json.loads(Path(args.booknlp).read_text(encoding="utf-8"))["predictions"] if args.booknlp else None
    jev = json.loads(Path(args.jev).read_text(encoding="utf-8")) if args.jev else []
    golds = [(Path(p).name, json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.gold]
    lines = dz.ScriptStore.public(base)["lines"]

    async def run():
        llm = dz.LocalLlm(cfg, None)
        llm.gpu = True
        chunks = dz.make_chunks(book, runs=dz.mixed_tense_runs(book))
        async with aiohttp.ClientSession() as session:
            try:
                for name, gold in golds:
                    flagged = {r["id"] for r in jev if r["set"] == name and not r["agree"] and r["conf"] >= args.min_conf}
                    for q in gold["quotes"] if bn else []:
                        b = bn[name].get(q["id"])
                        b_cid = dz.find_character(base, b) if b else None
                        if b is None or b_cid != (lines.get(q["id"]) or {}).get("speaker"):
                            flagged.add(q["id"])
                    after = copy.deepcopy(base)
                    t0 = time.perf_counter()
                    todo = [c for c in chunks if any(q[0] in flagged for q in c.quotes)]
                    for chunk in todo:
                        work = copy.deepcopy(after)
                        result, local = await dz.label_chunk(llm, session, book, work, chunk, thinking=True)
                        by = work.get("narrator_by_tense") or {}
                        if chunk.tense and by.get(chunk.tense):
                            result["narrator"] = by[chunk.tense]
                        dz.apply_chunk_result(work, result, local)
                        for qid in flagged & set(local_ids for ids in local.values() for local_ids in ids):
                            after["lines"][qid] = work["lines"][qid]
                            for cid in {work["lines"][qid].get("speaker")} - set(after["characters"]) - {None}:
                                after["characters"][cid] = work["characters"][cid]
                    secs = time.perf_counter() - t0
                    r0, r1 = score(gold, base), score(gold, after)
                    n = r0["quotes"]
                    print(f"{name}: flagged {len(flagged)}/{len(gold['quotes'])}, re-checked {len(todo)} passages in "
                          f"{secs:.0f}s; accuracy {r0['correct']}/{n} = {r0['correct'] / n:.1%} -> "
                          f"{r1['correct']}/{n} = {r1['correct'] / n:.1%}")
                    for w in r1["wrong"]:
                        print("   still wrong:", w)
            finally:
                await llm.stop()

    asyncio.run(run())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("route")
    r.add_argument("gold", nargs="+")
    r.add_argument("--book", required=True)
    r.add_argument("--booknlp", help="flag where ModernBookNLP disagrees")
    r.add_argument("--jev", help="or flag from eval_jev.py --out rows (disagrees with confidence >= --min-conf)")
    r.add_argument("--min-conf", type=float, default=0.5)
    a = sub.add_parser("booknlp")
    a.add_argument("gold", nargs="+")
    a.add_argument("--out", required=True)
    a.add_argument("--book", help="run over this whole book (id on this PC) instead of the gold excerpt")
    a.add_argument("--work", default=r"D:\Models\AndroidReader\work\booknlp")
    b = sub.add_parser("score")
    b.add_argument("gold", nargs="+")
    b.add_argument("--booknlp", required=True)
    b.add_argument("--llm", help="eval_attribution.py --dump output")
    b.add_argument("--script", help="a book's script.json (for gold excerpts of that book)")
    args = p.parse_args()
    {"booknlp": cmd_booknlp, "score": cmd_score, "route": cmd_route}[args.cmd](args)


if __name__ == "__main__":
    main()
