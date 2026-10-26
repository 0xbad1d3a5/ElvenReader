"""Check the roster pass: do the main characters get the gender and apparent age a listener expects?

    .venv\\Scripts\\python.exe scripts\\eval_roster.py tests\\gold\\sasaki10-roster.json [--gpu]

Uses the analysed book on this PC (data dir), runs the same roster prompt as the
server against a local llama.cpp server it starts itself, and compares gender and
age group with the gold labels. Nothing on the server is changed.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from androidreader_server import dramatize as dz  # noqa: E402
from androidreader_server.config import load_config  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("--gpu", action="store_true", help="offload to the GPU (default: CPU only)")
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args()
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    cfg = load_config()
    cfg.llm_mode = "gpu" if args.gpu else "cpu"
    store = dz.ScriptStore(cfg.data_path / "books")
    book = store.book(gold["book_id"])
    if book is None:
        sys.exit(f"book {gold['book_id']} is not on this PC; process it first")
    script = copy.deepcopy(store.script(gold["book_id"]))
    for c in script["characters"].values():
        c.pop("voice", None)  # describe every main character afresh
    if args.show_evidence:
        for name in gold["characters"]:
            cid = dz.find_character(script, name)
            c = script["characters"].get(cid, {})
            print(f"== {name}")
            for t in dz.appearance_evidence(book, [c.get("name", name), *c.get("aliases", [])]):
                print("   -", t)
    llm = dz.LocalLlm(cfg, None)
    async with aiohttp.ClientSession() as session:
        try:
            await llm.ensure(session)
            await dz.describe_cast(llm, session, book, script)
        finally:
            await llm.stop()
    ok_g = ok_a = 0
    for name, want in gold["characters"].items():
        c = script["characters"].get(dz.find_character(script, name) or "", {})
        g, a = c.get("gender"), c.get("age")
        good_g, good_a = g == want["gender"], a in want["age"]
        ok_g += good_g
        ok_a += good_a
        mark = "ok " if good_g and good_a else "BAD"
        print(f"{mark} {name:20} {g}/{a:12} want {want['gender']}/{'|'.join(want['age'])}")
        print(f"      appearance: {c.get('appearance', '')}")
        print(f"      voice: {c.get('description', '')}")
    n = len(gold["characters"])
    print(f"\ngender {ok_g}/{n}, apparent age {ok_a}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
