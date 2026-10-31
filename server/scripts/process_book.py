"""Add a book from the command line: the same as "+ Add book" in the library page.

    .venv\\Scripts\\python.exe scripts\\process_book.py "E:\\Books\\book.epub" [--no-cast] [--reanalyze] [--engine qwen3tts]

The running server (run-server.ps1) parses the EPUB, stores it in the library, analyses it (who says what,
the cast's tiers and voice profiles) and then creates the cast's voices. This script starts that and prints
progress until it is done.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from elvenreader_server.config import load_config  # noqa: E402


class Server:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip("/")
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 600):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}


def wait(server: Server, book_id: str, label: str) -> dict:
    t0 = time.time()
    last = ""
    while True:
        try:
            status, s = server.call("GET", f"/api/books/{book_id}/script")
        except (urllib.error.URLError, OSError):
            time.sleep(10)  # server restarting
            continue
        if status != 200:
            sys.exit(f"server error {status}: {s}")
        line = f"{label}: {s['status']} {s.get('progress', 0) * 100:.0f}% - {s.get('detail', '')}"
        if line != last:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {line}", flush=True)
            last = line
        if s["status"] not in ("queued", "running", "casting"):
            return s
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("epub", type=Path)
    parser.add_argument("--no-cast", action="store_true", help="don't create character voices")
    parser.add_argument("--engine", help="voice-design engine for the cast (default from config)")
    parser.add_argument("--reanalyze", action="store_true", help="analyse again even if already done")
    args = parser.parse_args()
    if not args.epub.is_file():
        sys.exit(f"no such file: {args.epub}")

    cfg = load_config()
    server = Server(f"http://127.0.0.1:{cfg.port}", cfg.token)
    try:
        status, _ = server.call("GET", "/api/health", timeout=10)
    except (urllib.error.URLError, OSError):
        status = 0
    if status != 200:
        sys.exit("The server is not running; start it with run-server.ps1")

    t_start = time.time()
    body = {"path": str(args.epub.resolve()), "cast": not args.no_cast, "reanalyze": args.reanalyze}
    if args.engine:
        body["engine"] = args.engine
    status, book = server.call("POST", "/api/library/import", body)
    if status != 200:
        sys.exit(f"could not add the book ({status}): {book.get('error', book)}")
    book_id = book["id"]
    print(f"'{book['title']}' by {book.get('author', '')}: {book.get('words', 0)} words, {book.get('chapters', 0)} "
          f"chapters, {book.get('quotes', 0)} quotations, id {book_id}", flush=True)
    s = wait(server, book_id, "progress")
    if s["status"] != "done":
        sys.exit(f"ended with {s['status']}: {s.get('detail')}")
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")
    print(f"{len(s['lines'])} lines, {len(s['characters'])} characters:")
    for tier in ("main", "supporting", "extra", None):
        members = [c for c in s["characters"] if c.get("role") == tier]
        if members:
            print(f"  {(tier or 'unsorted').title()}: " + ", ".join(f"{c['name']} ({c['lines']})" for c in members))
    print(f"\nOpen it: http://127.0.0.1:{cfg.port}/books/{book_id}")


if __name__ == "__main__":
    main()