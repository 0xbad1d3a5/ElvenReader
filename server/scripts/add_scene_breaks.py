"""One-off: record the scene breaks of books already in the library (book.json "breaks", see epub.py) by parsing
their EPUBs again. Only chapters whose paragraphs come out exactly as stored are touched, so quotation IDs,
corrections and stored speech stay as they are. With --narration it then works out who narrates each section again
(script.json "narration"; sections now end at scene breaks too). Stop the server first for --narration (it saves
scripts too).

    .venv\\Scripts\\python.exe scripts\\add_scene_breaks.py [--narration] [--dry-run]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from elvenreader_server import epub  # noqa: E402


def main() -> None:
    dry = "--dry-run" in sys.argv
    total = 0
    silent_total = [0]
    for f in sorted((ROOT / "data" / "books").glob("*/book.json")):
        src = f.with_name("book.epub")
        if not src.exists():
            continue
        book = json.loads(f.read_text(encoding="utf-8"))
        try:
            fresh = epub.parse(src)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.parent.name}: could not parse ({exc})")
            continue
        if len(fresh["chapters"]) != len(book["chapters"]):
            print(f"{f.parent.name}: chapters differ; skipped")
            continue
        changed = added = skipped = 0
        for old, new in zip(book["chapters"], fresh["chapters"]):
            if old["paragraphs"] != new["paragraphs"]:
                skipped += 1
                continue
            touched = False
            for key in ("breaks", "silent"):
                value = new.get(key)
                if old.get(key) != value:
                    touched = True
                    if value:
                        old[key] = value
                    else:
                        old.pop(key, None)
            if touched:
                changed += 1
                added += len(new.get("breaks") or [])
                silent_total[0] += sum(len(v) for v in (new.get("silent") or {}).values())
        total += added
        print(f"{f.parent.name} {book.get('title', '')[:50]}: {added} breaks in {changed} chapters"
              + (f", {skipped} chapters skipped (text differs)" if skipped else ""))
        if changed and not dry:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, f)
    print(f"{total} scene breaks, {silent_total[0]} wordless lines" + (" (dry run)" if dry else ""))


def refresh_narration(root: Path, dry: bool) -> None:
    """Who narrates each section, again with the scene breaks. A narration correction made before them covered up to
    the next point-of-view heading or chapter: it is carried across the breaks in between, so it keeps that span."""
    from elvenreader_server import dramatize as dz

    for f in sorted((root / "data" / "books").glob("*/script.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("status") != "done":
            continue
        book = json.loads(f.with_name("book.json").read_text(encoding="utf-8"))
        fixed = dict(s.get("narration_fixed") or {})
        for key, who in list(fixed.items()):
            ci, pi = map(int, key.split(":"))
            ch = book["chapters"][ci]
            for p in sorted(ch.get("breaks") or []):
                if p <= pi:
                    continue
                if any(dz.is_pov_marker(ch["paragraphs"][q]) for q in range(pi + 1, p + 1)):
                    break
                fixed.setdefault(f"{ci}:{p}", who)
        new = dz.compute_narration(book, {**s, "narration_fixed": fixed})
        if new == s.get("narration") and fixed == (s.get("narration_fixed") or {}):
            continue
        print(f"{f.parent.name}: narration {len(s.get('narration') or [])} -> {len(new)} sections")
        if not dry:
            s["narration"], s["narration_fixed"] = new, fixed
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, f)


if __name__ == "__main__":
    main()
    if "--narration" in sys.argv:
        refresh_narration(ROOT, "--dry-run" in sys.argv)
