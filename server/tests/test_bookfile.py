import json
import shutil
import zipfile

import pytest

from elvenreader_server import bookfile
from elvenreader_server.app import CACHE_KEY, CONFIG_KEY, make_app
from elvenreader_server.audio import wav_bytes
from elvenreader_server.config import ServerConfig
from elvenreader_server.dramatize import DRAMA_STORE_KEY
from elvenreader_server.engines.registry import FAKE_ENGINE
from elvenreader_server.voices import VoiceStore

TOKEN = "testtoken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BOOK_ID = "0123456789abcdef"
Q, E = "\u201c", "\u201d"
PARAS = [
    "One",
    f"{Q}Hello there,{E} Anna said. {Q}Come in.{E}",
    f"It was a {Q}thing{E}, really.",
    f"{Q}Exit{E}",
]


def quotes() -> list:
    out = []
    for pi, text in enumerate(PARAS):
        start, n = 0, 0
        while (s := text.find(Q, start)) >= 0:
            e = text.index(E, s) + 1
            out.append([f"c0p{pi}q{n}", pi, s, e])
            start, n = e, n + 1
    return out


BOOK = {"title": "Test Book Volume 1", "author": "Someone", "language": "en",
        "chapters": [{"title": "One", "paragraphs": PARAS, "quotes": quotes()}]}
SCRIPT = {
    "status": "done", "progress": 1.0, "detail": "", "model": "",
    "characters": {"anna": {"name": "Anna", "voice": "anna"}, "bob": {"name": "Bob", "voice": "bob"}},
    "lines": {"c0p1q0": {"speaker": "anna", "delivery": "warm"}, "c0p1q1": {"speaker": "anna", "delivery": "warm"},
              "c0p2q0": {"speaker": None, "delivery": None, "mention": True},
              "c0p3q0": {"speaker": None, "delivery": None}},
    "corrections": {}, "chunks_done": [],
    "narration": [[0, 0, "bob"]],
}


def make_cfg(tmp_path) -> ServerConfig:
    cfg = ServerConfig(token=TOKEN, default_engine="fake", preload=False, trim_silence=False, level_speech=False, sentence_pause_ms=0,
                       synth_timeout_s=20, load_timeout_s=30, idle_after_s=0.5, design_pause_s=0.5,
                       book_file_sync_s=0, llm_url="http://127.0.0.1:9/v1")  # never this PC's own LLM
    cfg.data_dir = str(tmp_path)
    cfg.models_dir = str(tmp_path / "models")
    return cfg


def add_voices(root):
    store = VoiceStore(root / "voices")
    for vid in ("nar", "anna", "bob"):
        store.add(vid.title(), wav_bytes(bytes([len(vid)]) * 4800, 24000), "Hello.", voice_id=vid)


@pytest.fixture(autouse=True)
def fake_opus(monkeypatch):
    """No ffmpeg in the tests: "Opus" is the PCM with a marker."""
    monkeypatch.setattr(bookfile, "encode_opus", lambda pcm, rate: b"OPUS" + pcm)
    monkeypatch.setattr(bookfile, "decode_opus", lambda data, rate: data[4:])


@pytest.fixture
async def client(aiohttp_client, tmp_path):
    add_voices(tmp_path)
    app = make_app(make_cfg(tmp_path), specs={"fake": FAKE_ENGINE})
    store = app[DRAMA_STORE_KEY]
    store.save_book(BOOK_ID, BOOK)
    store.save_script(BOOK_ID, json.loads(json.dumps(SCRIPT)))
    return await aiohttp_client(app)


def unit_request(seg: dict) -> dict:
    """What the viewer sends for a segment (viewer.html unitRequest())."""
    body = {"text": seg["text"], "book": BOOK_ID, "language": "English"}
    if seg.get("voice"):
        body["voice"] = seg["voice"]
    if seg.get("delivery"):
        body["delivery"] = seg["delivery"]
    if seg.get("guidance") is not None:
        body["guidance"] = seg["guidance"]
    if seg.get("temperature") is not None:
        body["temperature"] = seg["temperature"]
    return body


async def speak(client, seg: dict):
    """Ask for a segment's speech as the viewer does, and wait for all of it (it is cached at the end)."""
    resp = await client.post("/api/tts", json=unit_request(seg), headers=AUTH)
    body = await resp.read()
    assert resp.status == 200, body
    return resp, body


def public():
    from elvenreader_server.dramatize import ScriptStore

    return ScriptStore.public(json.loads(json.dumps(SCRIPT)))


def test_segments_split_and_voice_paragraphs_like_the_viewer():
    segs = bookfile.segments(BOOK, public())
    assert [(s["id"], s["kind"], s["text"], s["speaker"], s["voice"], s["delivery"]) for s in segs] == [
        ("c0p0n0", "narration", "One", "bob", "bob", None),
        ("c0p1q0", "quote", f"{Q}Hello there,{E}", "anna", "anna", "warm"),
        ("c0p1n0", "narration", "Anna said.", "bob", "bob", None),
        ("c0p1q1", "quote", f"{Q}Come in.{E}", "anna", "anna", "warm"),
        # a quoted term inside a sentence stays in the narration
        ("c0p2n0", "narration", f"It was a {Q}thing{E}, really.", "bob", "bob", None),
        # a quotation given to nobody is read by the narrator
        ("c0p3q0", "quote", f"{Q}Exit{E}", "bob", "bob", None),
    ]
    # The viewer's pauses (120 ms inside a paragraph, 450 ms after it), shortened after a clause and none
    # without closing punctuation.
    assert [s["pause_ms"] for s in segs] == [0, 40, 120, 450, 450, 0]
    # a new scene (the EPUB's scene break before paragraph 3) gets a longer one
    with_break = {**BOOK, "chapters": [{**BOOK["chapters"][0], "breaks": [3]}]}
    assert [s["pause_ms"] for s in bookfile.segments(with_break, public())] == [0, 40, 120, 450, 1200, 0]


def test_a_quotation_spelled_the_way_it_sounds_is_read_as_the_analysis_says():
    from elvenreader_server.dramatize import ScriptStore

    script = json.loads(json.dumps(SCRIPT))
    script["lines"]["c0p1q0"]["say"] = f"{Q}Hello there,{E}"
    script["lines"]["c0p3q0"]["say"] = "Exit!"
    script["corrections"]["c0p3q0"] = {"speaker": "anna"}  # a correction keeps the spoken form
    pub = ScriptStore.public(script)
    assert pub["lines"]["c0p3q0"]["say"] == "Exit!"
    segs = {s["id"]: s for s in bookfile.segments(BOOK, pub)}
    assert segs["c0p3q0"]["text"] == "Exit!" and segs["c0p3q0"]["voice"] == "anna"
    assert (segs["c0p3q0"]["start"], segs["c0p3q0"]["end"]) == (0, len(PARAS[3]))  # the page's text is untouched
    back = bookfile._script_from_public(pub)  # and it survives a book file
    assert back["lines"]["c0p3q0"]["say"] == "Exit!" and back["corrections"]["c0p3q0"]["speaker"] == "anna"


async def test_speech_goes_into_the_book_file_and_comes_back_from_it(client, tmp_path):
    app = client.server.app
    segs = bookfile.segments(BOOK, public())
    for seg in segs[:3]:
        resp, _ = await speak(client, seg)
        assert resp.headers["X-Cache"] == "miss"
    assert BOOK_ID in app[bookfile.BOOKFILES_KEY].dirty

    r = await (await client.post(f"/api/books/{BOOK_ID}/file/sync", headers=AUTH)).json()
    assert r["clips"] == 3 and r["voiced"] == 3 and r["segments"] == 6
    path = tmp_path / "books" / BOOK_ID / "book.elvenbook"
    with zipfile.ZipFile(path) as z:
        assert z.infolist()[0].filename == "mimetype" and z.read("mimetype") == b"application/x-elvenbook"
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["format"] == "elvenbook" and manifest["id"] == BOOK_ID and manifest["voiced"] == 3
        assert json.loads(z.read("text/book.json")) == BOOK
        stored = [json.loads(line) for line in z.read("analysis/segments.jsonl").splitlines()]
        assert [s["id"] for s in stored] == [s["id"] for s in segs]
        assert stored[0]["voice"] == "bob" and stored[0]["speakable"]
        index = json.loads(z.read("audio/index.json"))
        assert set(index) == {"c0p0n0", "c0p1q0", "c0p1n0"}
        assert z.read(index["c0p1q0"]["entry"]).startswith(b"OPUS")
        assert {n for n in z.namelist() if n.startswith("voices/")} == {
            "voices/anna.json", "voices/anna.wav", "voices/bob.json", "voices/bob.wav"}

    status = await (await client.get(f"/api/books/{BOOK_ID}/file", headers=AUTH)).json()
    assert status["exists"] and status["voiced"] == 3 and status["segments"] == 6

    # Nothing changed: nothing is written.
    size = path.stat().st_size
    r = await (await client.post(f"/api/books/{BOOK_ID}/file/sync", headers=AUTH)).json()
    assert r["written"] == 0 and path.stat().st_size == size

    # The cache forgot the clip; the book file still has it.
    expected = app[CACHE_KEY].peek(index["c0p1q0"]["key"])[1]
    shutil.rmtree(tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    resp, body = await speak(client, segs[1])
    assert resp.headers["X-Cache"] == "file" and body == expected
    resp, _ = await speak(client, segs[1])
    assert resp.headers["X-Cache"] == "hit"  # and the cache has it again


async def test_a_passages_speech_can_be_forgotten_and_is_made_again(client, tmp_path):
    app = client.server.app
    segs = bookfile.segments(BOOK, public())
    for seg in segs[:2]:
        await speak(client, seg)
    bookfile.sync(app, BOOK_ID)
    path = tmp_path / "books" / BOOK_ID / "book.elvenbook"
    with zipfile.ZipFile(path) as z:
        key = json.loads(z.read("audio/index.json"))["c0p1q0"]["key"]

    resp = await client.post(f"/api/books/{BOOK_ID}/file/forget", json={"segments": ["c0p1q0", "c0p3q0"]},
                             headers=AUTH)
    assert (await resp.json()) == {"forgotten": ["c0p1q0"]}  # c0p3q0 had no speech
    assert app[CACHE_KEY].peek(key) is None
    with zipfile.ZipFile(path) as z:
        assert set(json.loads(z.read("audio/index.json"))) == {"c0p0n0"}
        assert json.loads(z.read("manifest.json"))["voiced"] == 1
        assert not any(n.endswith("c0p1q0.opus") for n in z.namelist())
    resp, _ = await speak(client, segs[1])
    assert resp.headers["X-Cache"] == "miss"  # made again, not served from the book file
    assert bookfile.sync(app, BOOK_ID)["voiced"] == 2
    bad = await client.post(f"/api/books/{BOOK_ID}/file/forget", json={"segments": "c0p1q0"}, headers=AUTH)
    assert bad.status == 400
    # Speech only in the cache so far (not yet in the book file) is forgotten too.
    await speak(client, segs[2])
    resp = await client.post(f"/api/books/{BOOK_ID}/file/forget", json={"segments": [segs[2]["id"]]}, headers=AUTH)
    assert (await resp.json()) == {"forgotten": [segs[2]["id"]]}
    resp, _ = await speak(client, segs[2])
    assert resp.headers["X-Cache"] == "miss"


async def test_changed_lines_drop_their_clips_and_compaction_reclaims_the_space(client, tmp_path, monkeypatch):
    app = client.server.app
    segs = bookfile.segments(BOOK, public())
    for seg in segs[:2]:
        await speak(client, seg)
    bookfile.sync(app, BOOK_ID)

    store = app[DRAMA_STORE_KEY]
    s = store.script(BOOK_ID)
    s["corrections"]["c0p1q0"] = {"speaker": "bob", "delivery": "warm"}  # now Bob says it: Anna's clip is wrong
    store.save_script(BOOK_ID, s)
    monkeypatch.setattr(bookfile, "COMPACT_MIN_DEAD", 0)
    monkeypatch.setattr(bookfile, "COMPACT_DEAD_SHARE", 0.0001)
    r = bookfile.sync(app, BOOK_ID)
    assert r["voiced"] == 1 and r["deleted"] >= 1 and r["compacted"] and r["dead"] == 0 and r["generation"] == 2
    with zipfile.ZipFile(tmp_path / "books" / BOOK_ID / "book.elvenbook") as z:
        assert set(json.loads(z.read("audio/index.json"))) == {"c0p0n0"}
        assert not any(n.endswith("c0p1q0.opus") for n in z.namelist())


async def test_a_book_being_labelled_again_keeps_its_speech_and_takes_in_edits(client, tmp_path):
    app = client.server.app
    for seg in bookfile.segments(BOOK, public())[:3]:
        await speak(client, seg)
    assert bookfile.sync(app, BOOK_ID)["voiced"] == 3
    store = app[DRAMA_STORE_KEY]
    s = store.script(BOOK_ID)
    labels = s["lines"]
    s.update(status="queued", lines={})  # "Re-label lines": every label gone until the analysis runs
    s["characters"]["anna"]["name"] = "Anna Maria"  # an edit made meanwhile
    store.save_script(BOOK_ID, s)
    r = bookfile.sync(app, BOOK_ID)
    assert r["voiced"] == 3 and r["deleted"] == 0
    with zipfile.ZipFile(tmp_path / "books" / BOOK_ID / "book.elvenbook") as z:
        assert json.loads(z.read("x-server/script.json"))["characters"]["anna"]["name"] == "Anna Maria"
    s = store.script(BOOK_ID)
    s.update(status="done", lines=labels)  # the analysis gave the lines back as they were
    store.save_script(BOOK_ID, s)
    assert bookfile.sync(app, BOOK_ID)["voiced"] == 3


async def test_a_lines_own_guidance_is_its_own_clip_and_not_a_correction(aiohttp_client, tmp_path):
    from dataclasses import replace

    add_voices(tmp_path)
    app = make_app(make_cfg(tmp_path), specs={"fake": replace(FAKE_ENGINE, tuning=("guidance", "temperature"))})
    store = app[DRAMA_STORE_KEY]
    store.save_book(BOOK_ID, BOOK)
    store.save_script(BOOK_ID, json.loads(json.dumps(SCRIPT)))
    client = await aiohttp_client(app)
    url = f"/api/books/{BOOK_ID}/lines/c0p1q0"
    line = (await (await client.patch(url, json={"guidance": 3.0}, headers=AUTH)).json())["lines"]["c0p1q0"]
    assert line["guidance"] == 3.0 and not line["corrected"]  # tunes the speech: not a correction
    pub = await (await client.get(f"/api/books/{BOOK_ID}/script", headers=AUTH)).json()
    assert pub["tuning"]["guidance"]["default"] == 2.0 and pub["tuning"]["temperature"]["default"] == 1.0
    segs = bookfile.segments(BOOK, pub)
    assert segs[1]["id"] == "c0p1q0" and segs[1]["guidance"] == 3.0 and "guidance" not in segs[3]
    r = bookfile.Renderer(app, BOOK_ID, BOOK)
    assert r.render(segs[1]).key != r.render({**segs[1], "guidance": None}).key  # its own clip
    assert r.render({**segs[1], "guidance": 2.0}).key == r.render({**segs[1], "guidance": None}).key  # the default
    assert r.render({**segs[1], "temperature": 0.7}).key != r.render(segs[1]).key
    for seg in segs[:2]:
        await speak(client, seg)  # as the website asks for it
    assert bookfile.sync(app, BOOK_ID)["voiced"] == 2  # the file finds the clip /api/tts made
    await client.patch(url, json={"guidance": 2.0, "temperature": 0.7}, headers=AUTH)  # guidance back to the default
    s = store.script(BOOK_ID)
    assert "c0p1q0" not in s["guidance"] and s["temperature"]["c0p1q0"] == 0.7


async def test_speech_defaults_are_set_in_settings(aiohttp_client, tmp_path):
    from dataclasses import replace

    add_voices(tmp_path)
    app = make_app(make_cfg(tmp_path), specs={"fake": replace(FAKE_ENGINE, tuning=("guidance", "temperature"))})
    store = app[DRAMA_STORE_KEY]
    store.save_book(BOOK_ID, BOOK)
    store.save_script(BOOK_ID, json.loads(json.dumps(SCRIPT)))
    client = await aiohttp_client(app)
    s = await (await client.get("/api/settings", headers=AUTH)).json()
    assert s["tuning"]["guidance"]["value"] == 2.0 and s["tuning"]["temperature"]["model"] == 1.0
    seg = bookfile.segments(BOOK, public())[1]
    before = bookfile.Renderer(app, BOOK_ID, BOOK).render(seg).key
    r = await client.put("/api/settings", json={"tuning": {"guidance": 1.5, "temperature": 0.9}}, headers=AUTH)
    assert r.status == 200 and (await r.json())["tuning"]["guidance"]["value"] == 1.5
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["engine_options"]["fake"] == {"cfg_value": 1.5, "temperature": 0.9}
    assert bookfile.Renderer(app, BOOK_ID, BOOK).render(seg).key != before  # every line is made again
    assert (await client.put("/api/settings", json={"tuning": {"temperature": 0.2}}, headers=AUTH)).status == 400
    await client.put("/api/settings", json={"tuning": {"guidance": 2.0, "temperature": 1.0}}, headers=AUTH)
    assert "fake" not in app[CONFIG_KEY].engine_options  # back to the model's own
    assert bookfile.Renderer(app, BOOK_ID, BOOK).render(seg).key == before


async def test_a_line_left_open_is_closed_alike_for_the_website_and_the_book_file(aiohttp_client, tmp_path):
    from dataclasses import replace

    add_voices(tmp_path)
    app = make_app(make_cfg(tmp_path), specs={"fake": replace(FAKE_ENGINE, close_clauses=True)})
    store = app[DRAMA_STORE_KEY]
    store.save_book(BOOK_ID, BOOK)
    store.save_script(BOOK_ID, json.loads(json.dumps(SCRIPT)))
    client = await aiohttp_client(app)
    segs = bookfile.segments(BOOK, public())
    assert segs[1]["text"] == f"{Q}Hello there,{E}"  # the page's text: left as it is
    r = bookfile.Renderer(app, BOOK_ID, BOOK)
    assert r.render(segs[1]).key == r.render({**segs[1], "text": f"{Q}Hello there.{E}"}).key  # what the model gets
    for seg in segs[:2]:
        await speak(client, seg)
    assert bookfile.sync(app, BOOK_ID)["voiced"] == 2


async def test_clips_from_another_speech_model_are_kept_until_replaced(client, tmp_path):
    app = client.server.app
    seg = bookfile.segments(BOOK, public())[1]
    await speak(client, seg)
    bookfile.sync(app, BOOK_ID)
    from elvenreader_server.app import CONFIG_KEY

    app[CONFIG_KEY].engine_options = {"fake": {"pitch": 2}}  # every cache key changes
    r = bookfile.sync(app, BOOK_ID)
    assert r["voiced"] == 1 and r["clips"] == 0
    await speak(client, seg)
    r = bookfile.sync(app, BOOK_ID)
    assert r["voiced"] == 1 and r["clips"] == 1  # the new clip replaced it


async def test_a_book_file_brings_the_book_its_voices_and_speech_to_another_server(client, tmp_path, aiohttp_client):
    app = client.server.app
    segs = bookfile.segments(BOOK, public())
    _, first = await speak(client, segs[1])
    await client.post("/api/pronunciations", json={"term": "Anna", "respelling": "AH-nah", "scope": "series",
                                                   "book": BOOK_ID}, headers=AUTH)
    await speak(client, segs[2])
    bookfile.sync(app, BOOK_ID)
    exported = tmp_path / "export.elvenbook"
    shutil.copyfile(tmp_path / "books" / BOOK_ID / "book.elvenbook", exported)

    other = tmp_path / "other"
    other.mkdir()
    VoiceStore(other / "voices").add("Nar", wav_bytes(b"\x01\x00" * 2400, 24000), "Hi.", voice_id="nar")
    client2 = await aiohttp_client(make_app(make_cfg(other), specs={"fake": FAKE_ENGINE}))
    resp = await client2.post("/api/library/import", json={"path": str(exported)}, headers=AUTH)
    book = await resp.json()
    assert resp.status == 200, book
    assert book["id"] == BOOK_ID and book["status"] == "done" and book["title"] == BOOK["title"]
    app2 = client2.server.app
    assert app2[DRAMA_STORE_KEY].book(BOOK_ID) == BOOK
    assert app2[DRAMA_STORE_KEY].script(BOOK_ID)["lines"] == SCRIPT["lines"]
    assert {v.id for v in VoiceStore(other / "voices").list()} == {"nar", "anna", "bob"}
    # The book's series' pronunciation comes along as the series'.
    entries = (await (await client2.get(f"/api/pronunciations?book={BOOK_ID}", headers=AUTH)).json())
    assert [e["term"] for e in entries["series"]] == ["Anna"] and not entries["book"]

    resp, body = await speak(client2, segs[1])
    assert resp.headers["X-Cache"] == "file" and body == first
    resp, _ = await speak(client2, segs[2])
    assert resp.headers["X-Cache"] == "file"  # its pronunciation came along, so the key is the same

    again = await client2.post("/api/library/import", json={"path": str(exported)}, headers=AUTH)
    assert again.status == 409
    again = await client2.post("/api/library/import", json={"path": str(exported), "replace": True}, headers=AUTH)
    assert again.status == 200


async def test_import_rejects_other_files(client, tmp_path):
    bad = tmp_path / "x.elvenbook"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("hello.txt", "hi")
    resp = await client.post("/api/library/import", json={"path": str(bad)}, headers=AUTH)
    assert resp.status == 400 and "book file" in (await resp.json())["error"]


async def test_sweep_syncs_changed_books_and_skips_ones_being_analysed(client, tmp_path):
    app = client.server.app
    path = tmp_path / "books" / BOOK_ID / "book.elvenbook"
    store = app[DRAMA_STORE_KEY]
    s = store.script(BOOK_ID)
    s["status"] = "running"
    store.save_script(BOOK_ID, s)
    await bookfile.sweep(app)
    assert not path.exists()
    s["status"] = "done"
    store.save_script(BOOK_ID, s)
    await bookfile.sweep(app)
    assert path.exists()
    size = path.stat().st_size
    await bookfile.sweep(app)  # nothing changed: not even opened
    assert path.stat().st_size == size
    await speak(client, bookfile.segments(BOOK, public())[0])
    await bookfile.sweep(app)
    with zipfile.ZipFile(path) as z:
        assert set(json.loads(z.read("audio/index.json"))) == {"c0p0n0"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_opus_round_trip(monkeypatch):
    monkeypatch.undo()
    import math
    import struct

    rate = 24000
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(rate))
    data = bookfile.encode_opus(pcm, rate)
    assert data.startswith(b"OggS") and len(data) < len(pcm) // 4
    back = bookfile.decode_opus(data, rate)
    assert abs(len(back) - len(pcm)) < rate // 10


async def test_each_user_keeps_their_own_place_in_a_book(client, tmp_path):
    app = client.server.app
    users = (await (await client.get("/api/users", headers=AUTH)).json())["users"]
    assert [(u["name"], u["owner"], u["you"]) for u in users] == [("Me", True, True)]  # the server's token: made on first use
    owner = users[0]
    me = AUTH
    ana = await (await client.post("/api/users", json={"name": "Ana"}, headers=AUTH)).json()
    assert (await client.post("/api/users", json={"name": "ana"}, headers=AUTH)).status == 409
    her = {"Authorization": f"Bearer {ana['token']}"}
    url = f"/api/books/{BOOK_ID}/bookmark"
    assert await (await client.get(url, headers=me)).json() == {}
    r = await client.put(url, json={"chapter": 0, "paragraph": 2}, headers=me)
    assert r.status == 200 and (await r.json())["percent"] > 0
    assert (await (await client.get(url, headers=me)).json())["paragraph"] == 2
    assert await (await client.get(url, headers=her)).json() == {}  # Ana hasn't started it
    assert (await client.put(url, json={"chapter": 0, "paragraph": 3}, headers=her)).status == 200  # anyone keeps a place
    assert (await (await client.get(url, headers=me)).json())["paragraph"] == 2  # still mine
    books = (await (await client.get("/api/books", headers=me)).json())["books"]
    assert books[0]["bookmark"]["chapter"] == 0 and books[0]["bookmark"]["title"] == "One"
    hers = (await (await client.get("/api/books", headers=her)).json())["books"][0]["bookmark"]
    assert hers["percent"] > books[0]["bookmark"]["percent"]  # further along than me
    # Not a place in the book: refused.
    assert (await client.put(url, json={"chapter": 5, "paragraph": 0}, headers=me)).status == 400
    assert (await client.put(url, json={"chapter": 0, "paragraph": 99}, headers=me)).status == 400
    # The book file keeps them, and an import keeps each user's newer place.
    bookfile.sync(app, BOOK_ID)
    with zipfile.ZipFile(tmp_path / "books" / BOOK_ID / "book.elvenbook") as z:
        assert set(json.loads(z.read("x-server/bookmarks.json"))) == {owner["id"], ana["id"]}
    from elvenreader_server import bookmark

    bookmark.merge(app, BOOK_ID, {ana["id"]: {"chapter": 0, "paragraph": 1, "offset": 0, "updated": 1.0}})
    assert (await (await client.get(url, headers=her)).json())["paragraph"] == 3  # hers here is newer
    # Deleting a user forgets their places and their link stops working; the owner can't go.
    r = await client.delete(f"/api/users/{ana['id']}", headers=AUTH)
    assert r.status == 200 and [u["name"] for u in (await r.json())["users"]] == ["Me"]
    assert set(bookmark.read_all(app, BOOK_ID)) == {owner["id"]}
    assert (await client.get(url, headers=her)).status == 401
    assert (await client.delete(f"/api/users/{owner['id']}", headers=AUTH)).status == 400
