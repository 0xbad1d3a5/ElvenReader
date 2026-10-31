"""Dump one chapter of an EPUB as a gold-labelling file for eval_attribution.py.

    .venv\\Scripts\\python.exe scripts\\make_gold.py book.epub "It is a truth universally" out.json

Finds the spine document containing the given phrase, extracts its paragraphs
and quotations (same quote rules as the server) and writes them with empty
"speaker"/"delivery" fields to fill in by hand.
"""
from __future__ import annotations

import html
import json
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elvenreader_server.dramatize import quote_regions  # noqa: E402


class Paras(HTMLParser):
    BLOCK = {"p", "h1", "h2", "h3", "h4", "div", "li", "blockquote"}

    def __init__(self):
        super().__init__()
        self.out, self.buf, self.skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in self.BLOCK:
            self.flush()
        if tag == "br":
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if tag in self.BLOCK:
            self.flush()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)

    def flush(self):
        text = " ".join("".join(self.buf).replace("\u00ad", "").split())
        if any(c.isalnum() for c in text):
            self.out.append(html.unescape(text))
        self.buf = []


def main() -> None:
    epub, phrase, out = sys.argv[1], sys.argv[2], sys.argv[3]
    with zipfile.ZipFile(epub) as z:
        for name in z.namelist():
            if not re.search(r"\.x?html?$", name):
                continue
            doc = z.read(name).decode("utf-8", "replace")
            if phrase not in html.unescape(doc):
                continue
            p = Paras()
            p.feed(doc)
            p.flush()
            paras = p.out
            start = next(i for i, t in enumerate(paras) if phrase in t)
            # Keep from the chapter heading (if just before) to the next "CHAPTER" heading.
            first = max(0, start - 1)
            end = next((i for i in range(start + 1, len(paras)) if re.match(r"^(CHAPTER|Chapter) ", paras[i])), len(paras))
            paras = paras[first:end]
            quotes = []
            for pi, t in enumerate(paras):
                for s, e in quote_regions(t):
                    quotes.append({"id": f"c0p{pi}q{s}", "p": pi, "s": s, "e": e, "text": t[s:e],
                                   "speaker": "", "delivery": ""})
            Path(out).write_text(json.dumps({"source": Path(epub).name, "paragraphs": paras, "quotes": quotes},
                                            indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{len(paras)} paragraphs, {len(quotes)} quotes -> {out}")
            return
    raise SystemExit("phrase not found")


if __name__ == "__main__":
    main()
