"""The EPUB parser (a port of the former Android app's EpubParser/BookPayload; same chapters, paragraphs,
quotation IDs and pictures, so existing analyses stay valid)."""
import io
import zipfile
from pathlib import Path

from elvenreader_server import epub
from elvenreader_server.dramatize import quote_regions

FIXTURES = Path(__file__).parent / "fixtures"


def make_epub(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def test_parses_metadata_toc_cover_and_paragraphs(tmp_path):
    book = epub.parse(make_epub(tmp_path / "test.epub", {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": """<?xml version="1.0"?><container><rootfiles>
            <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>""",
        "OEBPS/content.opf": """<?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
            <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
              <dc:title>The Test Book</dc:title><dc:creator>A. Writer</dc:creator><dc:language>en-GB</dc:language>
            </metadata>
            <manifest>
              <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
              <item id="cover" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
              <item id="c1" href="text/ch%201.xhtml" media-type="application/xhtml+xml"/>
              <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
              <item id="empty" href="text/blank.xhtml" media-type="application/xhtml+xml"/>
            </manifest>
            <spine><itemref idref="empty"/><itemref idref="c1"/><itemref idref="c2"/></spine></package>""",
        "OEBPS/nav.xhtml": """<html><body><nav epub:type="toc"><ol>
            <li><a href="text/ch%201.xhtml#start">The Beginning</a></li>
            <li><a href="text/ch2.xhtml">The End</a></li></ol></nav></body></html>""",
        "OEBPS/images/cover.jpg": "JPEGDATA",
        "OEBPS/text/blank.xhtml": "<html><body><div><img src='x.png'/></div></body></html>",
        "OEBPS/text/ch 1.xhtml": """<html><head><style>p{}</style></head><body>
            <h1>Chapter 1</h1>
            <p>It was a <i>dark</i> and stormy&nbsp;night.<sup><a href="#n1">1</a></sup></p>
            <p>Soft\u00adhyphen here.<br/>New line.</p>
            <div>Loose text <p>inner para</p> tail</div>
            </body></html>""",
        "OEBPS/text/ch2.xhtml": "<html><body><p>\u201cStop!\u201d she shouted.</p><p>   </p></body></html>",
    }))
    assert (book["title"], book["author"], book["language"]) == ("The Test Book", "A. Writer", "en-GB")
    assert book["cover"] == b"JPEGDATA"
    assert [c["title"] for c in book["chapters"]] == ["The Beginning", "The End"]
    assert book["chapters"][0]["paragraphs"] == ["Chapter 1", "It was a dark and stormy night.", "Softhyphen here. New line.",
                                                 "Loose text", "inner para", "tail"]
    assert book["chapters"][1]["paragraphs"] == ["\u201cStop!\u201d she shouted."]
    assert book["chapters"][1]["quotes"] == [["c1p0q0", 0, 0, 7]]
    # the picture-only page before chapter 1 attaches to its first paragraph
    assert book["chapters"][0]["images"] == {"0": ["OEBPS/text/x.png"]}
    heading = epub.extract_document("<html><body><h2>Title</h2><p>Text.</p></body></html>")[0]
    assert [p["heading"] for p in heading] == [True, False]


def test_resolves_relative_hrefs():
    assert epub.resolve("OEBPS/nav.xhtml", "text/ch%201.xhtml#x") == "OEBPS/text/ch 1.xhtml"
    assert epub.resolve("OEBPS/text/c.xhtml", "../images/a.jpg") == "OEBPS/images/a.jpg"
    assert epub.resolve("content.opf", "c.xhtml") == "c.xhtml"


def test_files_missing_from_the_toc_continue_the_previous_chapter(tmp_path):
    book = epub.parse(make_epub(tmp_path / "split.epub", {
        "META-INF/container.xml": """<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>""",
        # not well-formed XML (undeclared dc: prefix): read leniently, like jsoup did
        "content.opf": """<package><metadata><dc:title>Split</dc:title></metadata><manifest>
            <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
            <item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>
            <item id="a1" href="a-01.xhtml" media-type="application/xhtml+xml"/>
            <item id="b" href="b.xhtml" media-type="application/xhtml+xml"/></manifest>
            <spine><itemref idref="a"/><itemref idref="a1"/><itemref idref="b"/></spine></package>""",
        "nav.xhtml": """<html><body><nav epub:type="toc"><ol><li><a href="a.xhtml">Part One</a></li>
            <li><a href="b.xhtml">Part Two</a></li></ol></nav></body></html>""",
        "a.xhtml": "<html><body><p>One.</p></body></html>",
        "a-01.xhtml": "<html><body><p>Still one.</p><h3>(The Neighbor's POV)</h3><p>Hers.</p></body></html>",
        "b.xhtml": "<html><body><p>Two.</p></body></html>",
    }))
    assert book["title"] == "Split"
    assert [c["title"] for c in book["chapters"]] == ["Part One", "Part Two"]
    assert book["chapters"][0]["paragraphs"] == ["One.", "Still one.", "(The Neighbor's POV)", "Hers."]


def test_pictures_attach_to_the_next_paragraph_without_changing_the_text(tmp_path):
    book = epub.parse(make_epub(tmp_path / "pics.epub", {
        "META-INF/container.xml": """<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>""",
        "OEBPS/content.opf": """<package><metadata><dc:title>Pics</dc:title></metadata><manifest>
            <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
            <item id="i0" href="text/insert0.xhtml" media-type="application/xhtml+xml"/>
            <item id="a" href="text/a.xhtml" media-type="application/xhtml+xml"/>
            <item id="i1" href="text/insert1.xhtml" media-type="application/xhtml+xml"/>
            <item id="b" href="text/b.xhtml" media-type="application/xhtml+xml"/></manifest>
            <spine><itemref idref="i0"/><itemref idref="a"/><itemref idref="i1"/><itemref idref="b"/></spine></package>""",
        "OEBPS/nav.xhtml": """<html><body><nav epub:type="toc"><ol><li><a href="text/a.xhtml">One</a></li>
            <li><a href="text/b.xhtml">Two</a></li></ol></nav></body></html>""",
        "OEBPS/text/insert0.xhtml": """<html><body><div><img src="../images/color.jpg"/></div></body></html>""",
        "OEBPS/text/a.xhtml": """<html><body><p>First.</p><p class="pic"><img src="../images/mid.png" alt=""/></p>
            <p>Second.</p></body></html>""",
        "OEBPS/text/insert1.xhtml": """<html><body><svg xmlns:xlink="http://www.w3.org/1999/xlink">
            <image xlink:href="../images/end.jpg"/></svg></body></html>""",
        "OEBPS/text/b.xhtml": "<html><body><p>Third.</p></body></html>",
    }))
    a, b = book["chapters"]
    assert (a["title"], b["title"]) == ("One", "Two")
    assert a["paragraphs"] == ["First.", "Second."]
    assert a["images"] == {"0": ["OEBPS/images/color.jpg"], "1": ["OEBPS/images/mid.png"]}
    assert a["end_images"] == ["OEBPS/images/end.jpg"]
    assert "images" not in b


def test_parsing_quirks_match_jsoup():
    doc = lambda body: [p["text"] for p in epub.extract_paragraphs(f"<html><body>{body}</body></html>")]  # noqa: E731
    # rows straight inside <table> get an implicit <tbody> (not a block): the table is one paragraph
    assert doc("<table>\n<tr><td>A</td></tr>\n<tr><td>B</td></tr></table>") == ["A B"]
    assert doc("<table><tbody><tr><td>A</td></tr><tr><td>B</td></tr></tbody></table>") == ["AB"]
    # self-closing tags are empty elements, not left open
    assert doc('<div><a id="x"/>Before<div/>After</div>') == ["Before", "After"]
    # stray end tags are ignored; footnote markers, page breaks and scripts are skipped
    assert doc('<p>One</span> two<sup>12</sup><span epub:type="pagebreak">7</span><script>x()</script></p>') == ["One two"]
    assert doc("<p>\u00a0 </p><p>\u2014 \u2014</p>") == []  # nothing speakable


def test_quotation_regions():
    text = "\u201cStop!\u201d she said. He said \"no\" twice. \u2018Don\u2019t,\u2019 she begged. It\u2019s 6\" long."
    assert [text[s:e] for s, e in quote_regions(text)] == ["\u201cStop!\u201d", "\"no\"", "\u2018Don\u2019t,\u2019"]
    unclosed = "\u201cI went home. Then I slept."
    assert [unclosed[s:e] for s, e in quote_regions(unclosed)] == [unclosed]


def test_alice_keeps_its_book_id_and_quote_ids():
    path = FIXTURES / "alice.epub"
    assert epub.book_id(path) == "cc6548c401743280"
    book = epub.parse(path)
    assert len(book["chapters"]) == 14
    assert sum(len(c["paragraphs"]) for c in book["chapters"]) == 849
    assert sum(len(c["quotes"]) for c in book["chapters"]) == 1124
    assert book["cover"] and book["cover"][:3] == b"\xff\xd8\xff"
    ch7 = next(c for c in book["chapters"] if "Mad Tea-Party" in c["title"])
    first = ch7["quotes"][0]
    assert first[0] == f"c{book['chapters'].index(ch7)}p{first[1]}q{first[2]}"
    assert io.BytesIO(book["cover"]).read(2) == b"\xff\xd8"
