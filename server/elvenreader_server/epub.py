"""EPUB parsing: chapters, paragraphs, quotations and pictures, with the IDs the reader has always used.

A port of the former Android app's EpubParser (jsoup) and BookPayload, so books keep their IDs and
analyses: book ID = the first 16 hex digits of the file's SHA-1; quotation ID = "c{chapter}p{paragraph}q{start}".
Chapters follow the book's table of contents (spine files the contents don't point to continue the chapter
before them); pictures are attached to the paragraph they precede, so they never shift the text.
"""
from __future__ import annotations

import hashlib
import posixpath
import re
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from .dramatize import quote_regions

PARSER_VERSION = 3  # the app's EpubParser.VERSION this matches

BLOCKS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dt", "dd", "section", "article",
          "header", "footer", "aside", "figure", "figcaption", "table", "tr", "td", "th", "ul", "ol", "dl", "hr", "body",
          "main", "center", "address"}
SKIP = {"script", "style", "nav", "head", "svg", "math", "rt", "rp", "template", "noscript"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
        "basefont", "bgsound", "frame", "keygen", "image"}
_WHITESPACE = re.compile("[ \t\n\x0b\f\r\u00a0\u2007\u202f]+")
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))


class EpubError(Exception):
    pass


# ----------------------------------------------------------------- a small jsoup-like HTML tree


@dataclass
class Node:
    tag: str = ""  # "" = text
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    text: str = ""

    def elements(self):
        return [c for c in self.children if c.tag]

    def select(self, tag: str):
        for c in self.elements():
            if c.tag == tag:
                yield c
            yield from c.select(tag)

    def all_text(self) -> str:
        return self.text if not self.tag else "".join(c.all_text() for c in self.children)


class _Builder(HTMLParser):
    """Builds a tree the way jsoup does for (X)HTML: tag names lower-cased, self-closing tags are empty
    elements, an end tag closes up to its open element (stray ones are ignored), <image> outside <svg> is <img>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def _in_svg(self) -> bool:
        return any(n.tag == "svg" for n in self.stack)

    def handle_starttag(self, tag, attrs, empty=False):
        if tag == "image" and not self._in_svg():
            tag = "img"
        # HTML5 (and jsoup) put rows of a <table> inside an implicit <tbody> (and cells inside a <tr>).
        if tag in ("tr", "td", "th") and self.stack[-1].tag == "table":
            body = Node("tbody")
            self.stack[-1].children.append(body)
            self.stack.append(body)
        if tag in ("td", "th") and self.stack[-1].tag in ("tbody", "thead", "tfoot"):
            row = Node("tr")
            self.stack[-1].children.append(row)
            self.stack.append(row)
        node = Node(tag, {k: (v if v is not None else "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if not empty and tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs, empty=True)

    def handle_endtag(self, tag):
        if tag == "image" and not self._in_svg():
            tag = "img"
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(Node("", text=data))


def parse_html(html: str) -> Node:
    b = _Builder()
    b.feed(html)
    b.close()
    return b.root


def _body(root: Node) -> Node | None:
    return next(root.select("body"), None)


# ----------------------------------------------------------------- text extraction (EpubParser.Walker)


def clean(raw: str) -> str:
    """Collapses whitespace and removes soft hyphens and zero-width characters."""
    return _WHITESPACE.sub(" ", raw.translate(_INVISIBLE)).strip()


def _speakable(text: str) -> bool:
    return any(ch.isalpha() or unicodedata.category(ch) == "Nd" for ch in text)


# A scene break: <hr>, an ornament picture, or the class publishers put on the paragraph after one
# (<p class="space-break">). Kept on that paragraph ("break"), never as a paragraph of its own, so numbering stays.
_BREAK_CLASS = re.compile(r"(?:^|[\s_-])(?:space|scene|section)[-_]?break[\w-]*(?:$|\s)", re.IGNORECASE)
# A paragraph of nothing but separator signs ("* * *", "◆ ◆ ◆", "＊＊＊", "⁂"): a scene break written as text.
_SEPARATOR = re.compile(r"(?:[*\u2042\u25c6\u25c7\u25cf\u25cb\u25a0\u25a1\u2666\u2726\u2727\u2756\u203b#~\uff0a\u2605\u2606]\s*){1,}")
_QUOTE_MARK = re.compile("[\"\u201c\u201d\u300c\u300d\u300e\u300f\u00ab\u00bb]")


def _is_break(el: Node) -> bool:
    if el.tag == "hr":
        return True
    cls = el.attrs.get("class", "")
    return bool(_BREAK_CLASS.search(cls)) or (el.tag == "img" and "ornament" in cls.lower().split())


def _skip(el: Node) -> bool:
    if el.tag in SKIP:
        return True
    kind = el.attrs.get("epub:type", "")
    if "noteref" in kind or "pagebreak" in kind:
        return True
    if el.attrs.get("role") in ("doc-noteref", "doc-pagebreak"):
        return True
    if el.tag == "sup":  # footnote markers such as <sup><a href="#n1">1</a></sup>
        t = el.all_text().strip()
        if len(t) <= 3 and not any(ch.isalpha() for ch in t):
            return True
    return False


class _Walker:
    def __init__(self, base: str):
        self.base = base
        self.out: list[dict] = []
        self.pending: list[str] = []
        self.brk = False  # a scene break before the next paragraph
        self.silent: list[str] = []  # wordless lines before the next paragraph
        self.alone: set[str] = set()  # pictures standing in a block of their own

    def picture(self, el: Node) -> None:
        if _is_break(el):
            self.brk = True
        if el.tag == "img":
            refs = [el.attrs.get("src", "")]
        elif el.tag == "svg":
            refs = [i.attrs.get("xlink:href") or i.attrs.get("href", "") for i in el.select("image")]
        else:
            return
        for ref in refs:
            if ref.strip() and not ref.startswith("data:"):
                self.pending.append(resolve(self.base, ref) if self.base else ref)

    def walk(self, el: Node) -> None:
        self.picture(el)
        if _skip(el):
            return
        if not any(c.tag in BLOCKS and not _skip(c) for c in el.elements()):
            before = len(self.pending)
            text = self.inline_text(el)
            if not clean(text) and len(self.pending) > before:
                self.alone.update(self.pending[before:])  # pictures in a block of their own (no text at all)
            self.emit(text, el.tag in HEADINGS)
            return
        run = []
        for node in el.children:
            if node.tag in BLOCKS:
                self.emit("".join(run), False)
                run = []
                self.walk(node)
            else:
                run.append(self.node_text(node))
        self.emit("".join(run), False)

    def inline_text(self, el: Node) -> str:
        return "".join(self.node_text(c) for c in el.children)

    def node_text(self, node: Node) -> str:
        if not node.tag:
            return node.text
        self.picture(node)
        if _skip(node):
            return ""
        if node.tag == "br":
            return " "
        if node.tag == "img":
            return ""
        return self.inline_text(node)

    def emit(self, raw: str, heading: bool) -> None:
        text = clean(raw)
        if _speakable(text):
            self.out.append({"text": text, "heading": heading, "images": list(self.pending), "break": self.brk,
                             "silent": list(self.silent), "alone": [p for p in self.pending if p in self.alone]})
            self.pending.clear()
            self.silent.clear()
            self.brk = False
        elif _SEPARATOR.fullmatch(text):
            self.brk = True  # "* * *", "◆◆◆": a scene break written as text
        elif text and _QUOTE_MARK.search(text):
            # A wordless line ("“…?”", "“……!”"): nothing to say aloud, but someone reacts. Kept with the next
            # paragraph, like a picture, so the paragraphs (and quotation IDs) are numbered as before.
            self.silent.append(text)


def extract_document(html: str, base: str = "") -> tuple[list[dict], list[str]]:
    """Paragraphs ({text, heading, images}) of one XHTML file, and pictures after its last paragraph."""
    body = _body(parse_html(html))
    if body is None:
        return [], []
    w = _Walker(base)
    w.walk(body)
    return w.out, w.pending


def extract_paragraphs(html: str) -> list[dict]:
    return extract_document(html)[0]


# ----------------------------------------------------------------- package and table of contents


def resolve(base: str, href: str) -> str:
    """Resolves `href` relative to the file at `base` (both zip paths)."""
    clean_href = href.split("#", 1)[0]
    if not clean_href:
        return base
    try:
        decoded = urllib.parse.unquote(clean_href.replace("+", "%2B"))
    except Exception:  # noqa: BLE001
        decoded = clean_href
    parts = [] if decoded.startswith("/") else [p for p in posixpath.dirname(base).split("/") if p]
    for seg in decoded.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _xml(text: str) -> ET.Element:
    try:
        return ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    except ET.ParseError:
        # Not well-formed: fall back to the lenient HTML tree, wrapped to look like ElementTree.
        return _as_etree(parse_html(text))


def _as_etree(node: Node) -> ET.Element:
    el = ET.Element(node.tag or "#text", node.attrs)
    el.text = "".join(c.text for c in node.children if not c.tag)
    for c in node.elements():
        el.append(_as_etree(c))
    return el


def _iter(el: ET.Element, name: str):
    for e in el.iter():
        if _local(e.tag) == name:
            yield e


def _attr(el: ET.Element, name: str) -> str:
    for k, v in el.attrib.items():
        if _local(k) == name or k == name:
            return v
    return ""


def _text(el: ET.Element) -> str:
    return " ".join("".join(el.itertext()).split())


def _meta(opf: ET.Element, name: str) -> str | None:
    metadata = next(_iter(opf, "metadata"), opf)
    for e in metadata.iter():
        tag = e.tag.lower()
        local = _local(e.tag)
        if tag in (f"dc:{name}", name) or local == name:
            t = _text(e)
            if t:
                return t
    return None


def _nav_titles(nav_path: str, html: str) -> dict[str, str]:
    root = parse_html(html)
    navs = list(root.select("nav"))
    nav = next((n for n in navs if "toc" in n.attrs.get("epub:type", "")), navs[0] if navs else None)
    out: dict[str, str] = {}
    if nav is None:
        return out
    for a in nav.select("a"):
        if "href" not in a.attrs:
            continue
        text = " ".join(a.all_text().split())
        if text:
            out.setdefault(resolve(nav_path, a.attrs["href"]), text)
    return out


def _ncx_titles(ncx_path: str, xml: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for point in _iter(_xml(xml), "navpoint"):
        content = next(_iter(point, "content"), None)
        if content is None:
            continue
        label = next(_iter(point, "navlabel"), None)
        text = _text(label) if label is not None else ""
        if text:
            out.setdefault(resolve(ncx_path, _attr(content, "src")), text)
    return out


def book_id(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()[:16]


def parse(path: Path) -> dict:
    """The book as the server stores it (title, author, language, chapters with paragraphs, quotations and
    pictures), plus the cover's bytes under "cover" and the zip paths of every picture."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())

        def read(p: str) -> bytes | None:
            return z.read(p) if p in names else None

        def read_text(p: str) -> str | None:
            b = read(p)
            return b.decode("utf-8", errors="replace") if b is not None else None

        container = read_text("META-INF/container.xml")
        if container is None:
            raise EpubError("Not an EPUB: no container.xml")
        rootfile = next(_iter(_xml(container), "rootfile"), None)
        opf_path = _attr(rootfile, "full-path") if rootfile is not None else ""
        if not opf_path:
            raise EpubError("EPUB has no package document")
        opf_text = read_text(opf_path)
        if opf_text is None:
            raise EpubError(f"Missing {opf_path}")
        opf = _xml(opf_text)
        title = _meta(opf, "title") or path.stem
        author = _meta(opf, "creator") or ""
        language = _meta(opf, "language")

        manifest: dict[str, tuple[str, str, str]] = {}
        for item in _iter(opf, "item"):
            manifest[_attr(item, "id")] = (resolve(opf_path, _attr(item, "href")), _attr(item, "media-type"),
                                           _attr(item, "properties"))
        cover_path = _find_cover(opf, manifest)

        toc: dict[str, str] = {}
        nav = next((v for v in manifest.values() if "nav" in v[2].split(" ")), None)
        if nav and read_text(nav[0]) is not None:
            toc.update(_nav_titles(nav[0], read_text(nav[0])))
        if not toc:
            spine = next(_iter(opf, "spine"), None)
            ncx = manifest.get(_attr(spine, "toc") if spine is not None else "") or next(
                (v for v in manifest.values() if v[1] == "application/x-dtbncx+xml"), None)
            if ncx and read_text(ncx[0]) is not None:
                toc.update(_ncx_titles(ncx[0], read_text(ncx[0])))

        chapters: list[dict] = []
        toc_starts = set(toc)
        group = len(toc_starts) >= 2
        carry: list[tuple[str, bool]] = []  # pictures waiting for the next paragraph; True = they end the chapter before

        def with_images(paras, images):
            if images:
                paras[0]["images"] = images + paras[0]["images"]
            return paras

        for ref in _iter(opf, "itemref"):
            if _attr(ref, "linear") == "no":
                continue
            item = manifest.get(_attr(ref, "idref"))
            if not item or "html" not in item[1]:
                continue
            html = read_text(item[0])
            if html is None:
                continue
            paras, trailing = extract_document(html, item[0])
            continues = group and item[0] not in toc_starts and bool(chapters)
            if not paras:
                carry += [(p, continues) for p in trailing]
                continue
            if continues:
                chapters[-1]["paragraphs"] += with_images(paras, [p for p, _ in carry])
            else:
                before = [p for p, end in carry if end]
                if before:
                    chapters[-1]["end_images"] += before
                heading = next((p["text"] for p in paras if p["heading"]), None)
                chapters.append({"title": toc.get(item[0]) or (heading[:80] if heading else None)
                                 or f"Section {len(chapters) + 1}",
                                 "paragraphs": with_images(paras, [p for p, end in carry if not end]),
                                 "end_images": []})
            carry = [(p, True) for p in trailing]
        if not chapters:
            raise EpubError("No readable text in this EPUB")
        if carry:
            chapters[-1]["end_images"] += [p for p, _ in carry]
        _ornament_breaks(chapters, read)
        cover = read(cover_path) if cover_path else None
    return _payload(title.strip(), author.strip(), language.strip() if language else None, chapters, cover)


# A picture this small, standing alone between paragraphs, is an ornament marking a scene break (Way of Kings: a 14x12
# glyph in <p class="centerimage">, with no class saying so). Illustrations are hundreds of pixels; small pictures
# inside a sentence (Yen On draws some em dashes and foreign letters as pictures) are not breaks.
ORNAMENT_MAX_PX = (240, 48)  # (width, height)


def _ornament_breaks(chapters: list[dict], read) -> None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a server dependency
        return
    import io

    sizes: dict[str, tuple[int, int] | None] = {}

    def size(path: str):
        if path not in sizes:
            try:
                data = read(path)
                sizes[path] = Image.open(io.BytesIO(data)).size if data else None
            except Exception:  # noqa: BLE001 - an unreadable picture is not an ornament
                sizes[path] = None
        return sizes[path]

    for ch in chapters:
        for pi, p in enumerate(ch["paragraphs"]):
            alone = p.get("alone") or []
            if pi and alone and not p.get("break"):
                dims = [size(i) for i in alone]
                if all(d and d[0] <= ORNAMENT_MAX_PX[0] and d[1] <= ORNAMENT_MAX_PX[1] for d in dims):
                    p["break"] = True


def _find_cover(opf: ET.Element, manifest: dict[str, tuple[str, str, str]]) -> str | None:
    for href, _, props in manifest.values():
        if "cover-image" in props.split(" "):
            return href
    meta = next((m for m in _iter(opf, "meta") if _attr(m, "name") == "cover"), None)
    item = manifest.get(_attr(meta, "content")) if meta is not None else None
    if item and item[1].startswith("image/"):
        return item[0]
    for iid, (href, mtype, _) in manifest.items():
        if mtype.startswith("image/") and ("cover" in iid.lower() or "cover" in href.lower()):
            return href
    return None


def _payload(title: str, author: str, language: str | None, chapters: list[dict], cover: bytes | None) -> dict:
    """BookPayload.build: paragraphs, quotations with their IDs, pictures."""
    out = []
    for ci, ch in enumerate(chapters):
        paragraphs, quotes, images, breaks, silent = [], [], {}, [], {}
        for pi, p in enumerate(ch["paragraphs"]):
            paragraphs.append(p["text"])
            for s, e in quote_regions(p["text"]):
                quotes.append([f"c{ci}p{pi}q{s}", pi, s, e])
            if p["images"]:
                images[str(pi)] = p["images"]
            if p.get("break") and pi > 0:
                breaks.append(pi)
            if p.get("silent"):
                silent[str(pi)] = p["silent"]
        entry = {"title": ch["title"], "paragraphs": paragraphs, "quotes": quotes}
        if images:
            entry["images"] = images
        if breaks:
            entry["breaks"] = breaks  # paragraphs that start a new scene
        if silent:
            entry["silent"] = silent  # wordless lines ("…?") shown before these paragraphs
        if ch["end_images"]:
            entry["end_images"] = ch["end_images"]
        out.append(entry)
    return {"title": title, "author": author, "language": language, "chapters": out, "cover": cover}


def words(book: dict) -> int:
    return sum(len(p.split()) for ch in book["chapters"] for p in ch["paragraphs"])
