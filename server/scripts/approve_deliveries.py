"""One-off: every delivery in the library (analysis lines and readers' corrections) rewritten in the approved tags
(elvenreader_server/delivery.py). Words the approved list covers map directly ("furious" -> angry, "very
nervous" -> very afraid); the rest ("casual", "concerned", "pleading") are mapped once each by the local LLM,
constrained to the approved tags, and the mapping is kept in data/eval/delivery-mapping.json (rerunning reuses
it). Each script.json is backed up to script.before-tags.json first. Stop the server before running (the LLM
needs the GPU, and the server would overwrite the scripts):

    .venv\\Scripts\\python.exe scripts\\approve_deliveries.py [--dry-run]
"""
import argparse
import asyncio
import collections
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server import delivery as dl  # noqa: E402
from elvenreader_server import dramatize as dz  # noqa: E402
from elvenreader_server.config import load_config  # noqa: E402

MAPPING = ROOT / "data" / "eval" / "delivery-mapping.json"
BATCH = 50
PROMPT = """You convert the free-text delivery notes of an audiobook script (how an actor performs a line) into a fixed list of approved delivery tags. Each tag changes how the speech engine voices the whole line, so a wrong tag does harm and a missing one does little: when in doubt, give [].

Give [] for words that describe what the line says or does, or an ordinary way of talking; an actor reads those lines plainly: informative, explaining, analytical, academic, logical, practical, factual, agreeing, asking, answering, reporting, polite, casual, formal, friendly, brief, direct, abrupt, conversational, thoughtful, reflective, curious, questioning, decisive, confirming, helpful, professional, respectful, honest, sincere, accepting, persuasive, strategic.

Give tags only for words that clearly name a feeling the voice carries, a loudness, a speed or manner of speaking, a non-verbal sound, or a situation, usually one tag, at most three; a feeling may start with "slightly" or "very" when the word says so. Examples: furious -> ["very angry"]; irritated -> ["angry"]; pleading -> ["sad"]; terrified -> ["very afraid"]; booming -> ["loud"]; barely audible -> ["whisper"]; mumbling -> ["soft"]; gasping -> ["breathless"]; giggling -> ["amused", "laugh"]; grumbling -> ["angry", "soft"]; smug -> ["amused"]; concerned -> ["tender"]; flustered -> ["embarrassed"]; stern -> ["firm"]. "clear" is only for careful pronunciation (enunciating, spelling out), "fast"/"slow" only for words about speed, "flat" only for a voice without expression.

Approved tags, with the words each covers:
{tags}"""


def scripts():
    return sorted((ROOT / "data" / "books").glob("*/script.json"))


def collect() -> collections.Counter:
    words = collections.Counter()
    for f in scripts():
        backup = f.with_name("script.before-tags.json")
        s = json.loads((backup if backup.exists() else f).read_text(encoding="utf-8"))
        for src in ("lines", "corrections"):
            for line in (s.get(src) or {}).values():
                for part in (line.get("delivery") or "").split(","):
                    w = " ".join(part.lower().split())
                    if w:
                        words[w] += 1
    return words


async def ask_llm(words: list[str], known: dict) -> None:
    cfg = load_config()
    llm = dz.LocalLlm(cfg, None)
    llm.gpu = True
    system = PROMPT.replace("{tags}", dl.prompt_list())
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            for start in range(0, len(words), BATCH):
                batch = words[start:start + BATCH]
                ids = [str(i) for i in range(1, len(batch) + 1)]
                schema = {"type": "object", "properties": {"map": {
                    "type": "array", "minItems": len(batch), "maxItems": len(batch),
                    "items": {"type": "object", "properties": {
                        "id": {"type": "string", "enum": ids},
                        "tags": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": dl.all_tags()}}},
                        "required": ["id", "tags"]}}}, "required": ["map"]}
                user = "\n".join(f"[{i}] {w}" for i, w in zip(ids, batch))
                result = await llm.complete(session, system, user, schema, max_tokens=4000)
                got = 0
                for item in result.get("map") or []:
                    try:
                        w = batch[int(item["id"]) - 1]
                    except (KeyError, ValueError, IndexError):
                        continue
                    known[w] = [t for t in item.get("tags") or [] if dl.is_approved(t)]
                    got += 1
                MAPPING.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")
                print(f"{min(start + BATCH, len(words))} of {len(words)} ({got} answered)", flush=True)
    finally:
        await llm.stop()


def rewrite(known: dict, dry: bool) -> None:
    before_after = collections.Counter()
    tags = collections.Counter()
    for f in scripts():
        backup = f.with_name("script.before-tags.json")
        # Always from the deliveries as they were (a rerun with a corrected mapping starts over), with the rest
        # of the script as it is now.
        s = json.loads(f.read_text(encoding="utf-8"))
        if backup.exists():
            before = json.loads(backup.read_text(encoding="utf-8"))
            for src in ("lines", "corrections"):
                for qid, line in (s.get(src) or {}).items():
                    old = (before.get(src) or {}).get(qid)
                    if old is not None and "delivery" in old:
                        line["delivery"] = old["delivery"]
        changed = 0
        for src in ("lines", "corrections"):
            for line in (s.get(src) or {}).values():
                if "delivery" not in line:
                    continue
                old = line.get("delivery")
                new = dl.from_text(old, known)
                for t in (new or "").split(", "):
                    tags[t or "(none)"] += 1
                if new != old:
                    before_after[(old, new)] += 1
                    line["delivery"] = new
                    changed += 1
        print(f"{f.parent.name}: {changed} deliveries rewritten")
        if not dry and changed:
            if not backup.exists():
                shutil.copy2(f, backup)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            tmp.replace(f)
    print("\ntags now:", ", ".join(f"{t} {n}" for t, n in tags.most_common()))
    print("\ncommonest rewrites:")
    for (old, new), n in before_after.most_common(40):
        print(f"  {n:5d}  {old!r} -> {new!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    a = p.parse_args()
    words = collect()
    known = json.loads(MAPPING.read_text(encoding="utf-8")) if MAPPING.exists() else {}
    # The prompt's own examples, as given there.
    plain = PROMPT.split("Give [] for words", 1)[1].split(": ", 1)[1].split(".\n", 1)[0]
    for w in plain.split(", "):
        known.setdefault(w.strip(), [])
    for w, tags in re.findall(r"([a-z][a-z ]*) -> (\[[^\]]*\])", PROMPT):
        known.setdefault(w.strip(), json.loads(tags))
    unknown = [w for w, _ in words.most_common() if dl.match(w, known) is None]
    print(f"{len(words)} distinct words, {len(unknown)} for the LLM")
    if unknown and not a.dry_run:
        MAPPING.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(ask_llm(unknown, known))
    rewrite(known, a.dry_run)


if __name__ == "__main__":
    main()
