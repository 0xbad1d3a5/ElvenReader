"""How sure the labelling LLM is of each delivery tag it gives: labels sample passages the way the analysis does and
prints every tag with its probability (dramatize.delivery_odds), for choosing delivery_min_conf. Stop the server
first (the LLM needs the GPU):

    .venv\\Scripts\\python.exe scripts\\delivery_conf_probe.py BOOK_ID [BOOK_ID ...] [--passages 8]
"""
import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server import dramatize as dz  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("books", nargs="+")
    p.add_argument("--passages", type=int, default=8)
    p.add_argument("--out", default=str(ROOT / "data" / "eval" / "delivery-conf.json"))
    p.add_argument("--delivery-only", action="store_true",
                   help="score with score_deliveries (the backfill's prompt, speakers given) instead of labelling")
    a = p.parse_args()
    cfg = load_config()
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    import aiohttp

    rows = []
    try:
        async with aiohttp.ClientSession() as session:
            for bid in a.books:
                d = ROOT / "data" / "books" / bid
                book = json.loads((d / "book.json").read_text(encoding="utf-8"))
                script = json.loads((d / "script.json").read_text(encoding="utf-8"))
                chunks = [c for c in dz.make_chunks(book, runs=dz.mixed_tense_runs(book)) if len(c.quotes) >= 6]
                step = max(1, len(chunks) // a.passages)
                for chunk in chunks[::step][:a.passages]:
                    paras = book["chapters"][chunk.chapter]["paragraphs"]
                    spans = {q[0]: q for q in chunk.quotes}
                    if a.delivery_only:
                        scored = await dz.score_deliveries(llm, session, book, script, chunk,
                                                           context_words=cfg.llm_label_context_words)
                        for qid, (_, odds) in scored.items():
                            s = spans[qid]
                            rows.append({"book": bid, "qid": qid, "text": paras[s[1]][s[2]:s[3]][:110],
                                         "odds": odds or []})
                        print(f"{bid}: passage done ({len(rows)} quotations)", flush=True)
                        continue
                    result, local = await dz.label_chunk(
                        llm, session, book, script, chunk, context_words=cfg.llm_label_context_words,
                        after_words=cfg.llm_label_after_words, recent=cfg.llm_label_recent_lines)
                    for q in result.get("quotes", []):
                        ids = local.get(str(q.get("id"))) or []
                        if not ids:
                            continue
                        s = spans[ids[0]]
                        rows.append({"book": bid, "qid": ids[0], "text": paras[s[1]][s[2]:s[3]][:110],
                                     "speaker": q.get("speaker"), "odds": q.get("_dodds") or []})
                    print(f"{bid}: passage done ({len(rows)} quotations)", flush=True)
    finally:
        await llm.stop()
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    probs = [p for r in rows for _, p in r["odds"]]
    hist = collections.Counter(min(9, int(p * 10)) for p in probs)
    print(f"\n{len(rows)} quotations, {sum(1 for r in rows if r['odds'])} with tags, {len(probs)} tags")
    print("tag probability:", " ".join(f"{k / 10:.1f}+:{hist[k]}" for k in range(10)))
    for r in rows:
        if r["odds"]:
            print(f"  {', '.join(f'{t} {p:.2f}' for t, p in r['odds']):34} {r['text']}")


if __name__ == "__main__":
    asyncio.run(main())
