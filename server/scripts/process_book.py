"""Process a book on the PC: dramatize it and (optionally) put it on the phone.

    .venv\\Scripts\\python.exe scripts\\process_book.py "E:\\Books\\book.epub" [--phone] [--no-cast]

1. Runs the app's own EPUB parser on the JVM (android ExportBookTool) so the book
   ID and quotation IDs are identical to what the phone produces.
2. Uploads the text to the running server and starts the analysis (LLM speaker
   and delivery labels), waiting for it to finish.
3. Creates character voices (cast).
4. With --phone: copies the EPUB to the phone over adb and imports it into the
   app's library with dramatized reading switched on.

The server must be running (run-server.ps1). Progress is printed as it goes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from androidreader_server.config import load_config  # noqa: E402

ANDROID = ROOT.parent / "android"
PACKAGE = "com.androidreader"


class Server:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip("/")
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 120):
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


    def put_bytes(self, path: str, data: bytes, timeout: float = 120) -> int:
        req = urllib.request.Request(self.base + path, data=data, method="PUT")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/octet-stream")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code


def upload_pictures(server: Server, book_id: str, epub: Path) -> None:
    """The book's illustrations, for the viewer (the server only takes pictures the text refers to)."""
    status, have = server.call("GET", f"/api/books/{book_id}/pictures")
    need = have.get("need", []) if status == 200 else []
    if not need:
        return
    with zipfile.ZipFile(epub) as z:
        names = set(z.namelist())
        sent = sum(server.put_bytes(f"/api/books/{book_id}/picture?path={urllib.parse.quote(p)}", z.read(p)) == 200
                   for p in need if p in names)
    print(f"Uploaded {sent} of {len(need)} pictures", flush=True)


def java_home() -> str | None:
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    tool = Path(r"D:\Dev\.toolchains")
    jdks = sorted(tool.glob("jdk-17*")) if tool.exists() else []
    return str(jdks[-1]) if jdks else None


def export(epub: Path) -> dict:
    out = Path(tempfile.gettempdir()) / f"areader-export-{os.getpid()}.json"
    env = dict(os.environ)
    if java_home():
        env["JAVA_HOME"] = java_home()
    env.setdefault("ANDROID_HOME", os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk"))
    gradlew = ANDROID / ("gradlew.bat" if os.name == "nt" else "gradlew")
    cmd = [str(gradlew), "-q", "--console=plain", "testDebugUnitTest", "--tests",
           "com.androidreader.tools.ExportBookTool", f"-PexportEpub={epub}", f"-PexportOut={out}"]
    print("Parsing the EPUB with the app's parser...", flush=True)
    r = subprocess.run(cmd, cwd=ANDROID, env=env, capture_output=True, text=True, shell=os.name == "nt")
    if r.returncode != 0 or not out.exists():
        sys.exit(f"export failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    payload = json.loads(out.read_text(encoding="utf-8"))
    out.unlink()
    return payload


def wait(server: Server, book_id: str, busy: set[str], label: str) -> dict:
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
        if s["status"] not in busy:
            return s
        time.sleep(10)


def adb(*args: str, check: bool = True) -> str:
    r = subprocess.run(["adb", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"adb {' '.join(args)} failed: {r.stderr or r.stdout}")
    return r.stdout


def to_phone(epub: Path, book_id: str) -> None:
    """Imports the EPUB into the app (dramatized) through its VIEW intent."""
    print("Copying the book to the phone...", flush=True)
    tmp = f"/data/local/tmp/{book_id}.epub"
    adb("push", str(epub), tmp)
    dest = f"/data/user/0/{PACKAGE}/cache/import-{book_id}.epub"
    adb("shell", f"run-as {PACKAGE} sh -c 'cat {tmp} > cache/import-{book_id}.epub'")
    adb("shell", "rm", tmp)
    adb("shell", "am", "start", "-n", f"{PACKAGE}/.ui.MainActivity", "-a", "android.intent.action.VIEW",
        "-d", f"file://{dest}", "-t", "application/epub+zip", "--ez", "dramatized", "true")
    for _ in range(120):
        time.sleep(2)
        out = adb("shell", f"run-as {PACKAGE} cat files/books/{book_id}/meta.json", check=False)
        if out.strip().startswith("{"):
            meta = json.loads(out)
            print(f"On the phone: '{meta['title']}', dramatized={meta.get('dramatized')}", flush=True)
            adb("shell", f"run-as {PACKAGE} rm -f cache/import-{book_id}.epub", check=False)
            return
    sys.exit("The phone did not import the book (is the app installed and the phone unlocked?)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("epub", type=Path)
    parser.add_argument("--phone", action="store_true", help="also import the book on the phone over adb")
    parser.add_argument("--no-cast", action="store_true", help="skip creating character voices")
    parser.add_argument("--engine", help="voice-design engine for the cast (default from config)")
    parser.add_argument("--reanalyze", action="store_true", help="analyse again even if already done")
    args = parser.parse_args()
    if not args.epub.exists():
        sys.exit(f"no such file: {args.epub}")
    if args.phone and not shutil.which("adb"):
        sys.exit("adb is not on PATH")

    cfg = load_config()
    server = Server(f"http://127.0.0.1:{cfg.port}", cfg.token)
    status, _ = server.call("GET", "/api/health", timeout=10)
    if status != 200:
        sys.exit("The server is not running; start it with run-server.ps1")

    t_start = time.time()
    payload = export(args.epub)
    book_id = payload.pop("id")
    words = payload.pop("words")
    quotes = sum(len(c["quotes"]) for c in payload["chapters"])
    print(f"'{payload['title']}' by {payload['author']}: {words} words, {len(payload['chapters'])} "
          f"sections, {quotes} quotations, id {book_id}", flush=True)

    status, current = server.call("GET", f"/api/books/{book_id}/script")
    already = status == 200 and current.get("status") == "done" and not args.reanalyze
    # Always send the text: a newer parser may split the book differently, and the server then moves
    # the existing analysis to the new chapter numbering.
    payload["reanalyze"] = args.reanalyze
    status, s = server.call("POST", f"/api/books/{book_id}", {**payload, "analyze": not already}, timeout=600)
    if status != 200:
        sys.exit(f"upload failed ({status}): {s}")
    upload_pictures(server, book_id, args.epub)
    t_analysis = time.time()
    s = wait(server, book_id, {"queued", "running"}, "analysis")
    if s["status"] != "done":
        sys.exit(f"analysis ended with {s['status']}: {s.get('detail')}")
    analysis_min = (time.time() - t_analysis) / 60

    if not args.no_cast:
        body = {"engine": args.engine} if args.engine else {}
        status, s = server.call("POST", f"/api/books/{book_id}/cast", body)
        if status not in (200, 202):
            sys.exit(f"cast failed ({status}): {s}")
        s = wait(server, book_id, {"casting"}, "cast")

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min (analysis {analysis_min:.1f} min).")
    print(f"{len(s['lines'])} lines, {len(s['characters'])} characters. Main cast:")
    for c in s["characters"][:12]:
        print(f"  {c['name']:<28} {c['lines']:>4} lines  voice={c.get('voice')}  {c.get('description', '')[:70]}")
    print(f"\nView it on this PC: http://127.0.0.1:{cfg.port}/books/{book_id}")
    if args.phone:
        to_phone(args.epub, book_id)


if __name__ == "__main__":
    main()
