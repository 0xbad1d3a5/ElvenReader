"""One-off: quotations that were labelled as one interrupted quotation although they are two turns got one speaker
for both. First a sentence ending in the narration between them ("…now?” Asuna whispered. Right behind her ear,
Kirito said, “Sounds good.”), now narration bringing in another speaker by name ("Sh! sh!” and the Dormouse
sulkily remarked, “If you can’t be civil…”). Each passage holding such a pair is labelled again
(dramatize.interrupts now numbers them apart), and only those quotations take the new labels; corrections are
untouched. Stop the server first (the LLM needs the GPU and the server would overwrite the scripts):

    .venv\\Scripts\\python.exe scripts\\fix_merged_quotes.py [--dry-run]
"""
import argparse
import asyncio
import copy
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server import dramatize as dz  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402

_CONTINUES = re.compile(r"[,;:\u2014\u2013-]\s*$")


def affected(book: dict) -> set[str]:
    """Quotations the old rule joined to the next one although they are two turns (both of each pair)."""
    out = set()
    for ch in book["chapters"]:
        by: dict[int, list] = {}
        for q in ch.get("quotes", []):
            by.setdefault(q[1], []).append(q)
        for pi, qs in by.items():
            qs.sort(key=lambda q: q[2])
            p = ch["paragraphs"][pi]
            for (q1, _, _, e1), (q2, _, s2, _) in zip(qs, qs[1:]):
                between = p[e1:s2]
                # the previous rule (dramatize.interrupts before it knew a named new speaker starts a new turn)
                old = len(between) <= 80 and bool(_CONTINUES.search(between)) and not dz._SENTENCE_BREAK.search(between)
                if old and not dz.interrupts(between):
                    out |= {q1, q2}
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    import aiohttp

    books = sorted((ROOT / "data" / "books").glob("*/book.json"))
    try:
        async with aiohttp.ClientSession() as session:
            for f in books:
                book = json.loads(f.read_text(encoding="utf-8"))
                sp = f.with_name("script.json")
                script = json.loads(sp.read_text(encoding="utf-8"))
                todo = {q for q in affected(book) if q in script.get("lines", {})
                        and q not in (script.get("corrections") or {})}
                if not todo:
                    continue
                chunks = [c for c in dz.make_chunks(book, runs=dz.mixed_tense_runs(book))
                          if any(q[0] in todo for q in c.quotes)]
                print(f"{f.parent.name}: {len(todo)} quotations in {len(chunks)} passages", flush=True)
                if a.dry_run:
                    continue
                changed = 0
                for chunk in chunks:
                    result, local = await dz.label_chunk(
                        llm, session, book, script, chunk, thinking=cfg.llm_label_thinking,
                        context_words=cfg.llm_label_context_words, after_words=cfg.llm_label_after_words,
                        recent=cfg.llm_label_recent_lines)
                    trial = copy.deepcopy(script)
                    dz.apply_chunk_result(trial, result, local)
                    for qid in (q[0] for q in chunk.quotes if q[0] in todo):
                        new = trial["lines"].get(qid)
                        if not new:
                            continue
                        who = new.get("speaker")
                        if who and who not in script["characters"] and who in trial["characters"]:
                            script["characters"][who] = trial["characters"][who]
                        before = script["lines"].get(qid, {})
                        if (before.get("speaker"), before.get("delivery")) != (who, new.get("delivery")):
                            changed += 1
                            print(f"   {qid}: {before.get('speaker')} -> {who} ({new.get('delivery')})", flush=True)
                        script["lines"][qid] = new
                tmp = sp.with_suffix(".tmp")
                tmp.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, sp)
                print(f"   {changed} changed", flush=True)
    finally:
        await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())
