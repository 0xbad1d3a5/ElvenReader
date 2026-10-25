import asyncio

import pytest

from androidreader_server import dramatize as dz
from androidreader_server.app import make_app
from test_server import AUTH, SPECS, make_cfg
from androidreader_server.audio import wav_bytes
from androidreader_server.voices import VoiceStore

PARAS = [
    "Chapter One",
    "\u201cStop!\u201d she shouted. \u201cDon\u2019t move.\u201d",
    "Tom froze. \u201cWhy?\u201d he whispered.",
    "The sign said \u201cKeep Out\u201d in red letters.",
]


def book_payload():
    quotes = []
    for pi, text in enumerate(PARAS):
        for s, e in dz.quote_regions(text):
            quotes.append([f"c0p{pi}q{s}", pi, s, e])
    return {"title": "Test", "author": "Me", "language": "en",
            "chapters": [{"title": "One", "paragraphs": PARAS, "quotes": quotes}]}


def test_quote_regions_match_the_app():
    text = "\u201cStop!\u201d she said. He said \"no\" twice. \u2018Don\u2019t,\u2019 she begged. It\u2019s 6\" long."
    assert [text[s:e] for s, e in dz.quote_regions(text)] == ["\u201cStop!\u201d", "\"no\"", "\u2018Don\u2019t,\u2019"]


def test_chunks_and_rendering_keep_text_and_number_quotes():
    book = book_payload()
    chunks = dz.make_chunks(book)
    assert len(chunks) == 1 and len(chunks[0].quotes) == 4
    text, local = dz.render_chunk(book, chunks[0])
    assert "[1]\u201cStop!\u201d she shouted. [2]\u201cDon\u2019t move.\u201d" in text
    assert "[3]\u201cWhy?\u201d he whispered." in text
    assert local == {"1": ["c0p1q0"], "2": ["c0p1q21"], "3": ["c0p2q11"], "4": ["c0p3q14"]}
    # Removing the markers gives back the original paragraphs exactly.
    import re

    assert re.sub(r"\[\d+\]", "", text) == "\n\n".join(PARAS)


def test_interrupted_quotation_is_one_numbered_turn():
    para = "\u201cYes,\u201d he said, \u201cI will.\u201d \u201cNo!\u201d she cried."
    quotes = [[f"c0p0q{s}", 0, s, e] for s, e in dz.quote_regions(para)]
    book = {"chapters": [{"title": "x", "paragraphs": [para], "quotes": quotes}]}
    text, local = dz.render_chunk(book, dz.make_chunks(book)[0])
    assert text == "[1]\u201cYes,\u201d he said, [1]\u201cI will.\u201d [2]\u201cNo!\u201d she cried."
    assert local == {"1": ["c0p0q0", "c0p0q16"], "2": ["c0p0q26"]}
    script = {"characters": {}, "lines": {}}
    dz.apply_chunk_result(script, {"speakers": [], "quotes": [
        {"id": "1", "starts": "Yes", "speaker": "Tom", "delivery": "calm"},
        {"id": "2", "starts": "No", "speaker": "the Hatter and the March Hare", "delivery": "shouting"},
    ]}, local)
    assert script["lines"]["c0p0q16"]["speaker"] == "tom"
    assert script["lines"]["c0p0q26"]["speaker"] == "hatter"


def test_chunk_schema_forces_every_id():
    schema = dz.chunk_schema(["1", "2"])
    q = schema["properties"]["quotes"]
    assert q["minItems"] == q["maxItems"] == 2
    assert q["items"]["properties"]["id"]["enum"] == ["1", "2"]


def test_apply_result_creates_characters_and_matches_aliases():
    script = {"characters": {}, "lines": {}}
    local = {"1": "a", "2": "b", "3": "c", "4": "d"}
    dz.apply_chunk_result(script, {
        "speakers": [{"name": "Mrs. Bennet", "gender": "female", "age": "middle-aged"}],
        "quotes": [
            {"id": "1", "speaker": "Mrs. Bennet", "delivery": "excited"},
            {"id": "2", "speaker": "mrs bennet", "delivery": "neutral"},
            {"id": "3", "speaker": "Mr. Bennet", "delivery": "dry, sarcastic"},
            {"id": "4", "speaker": "NARRATOR", "delivery": "neutral"},
        ],
    }, local)
    assert script["lines"]["a"] == {"speaker": "mrs-bennet", "delivery": "excited"}
    assert script["lines"]["b"] == {"speaker": "mrs-bennet", "delivery": None}
    assert script["lines"]["c"]["speaker"] == "mr-bennet"
    assert script["lines"]["d"]["speaker"] is None
    assert script["characters"]["mrs-bennet"]["gender"] == "female"
    assert "Mrs. Bennet (female, middle-aged)" in dz.roster_text(script)


@pytest.fixture
async def drama_client(aiohttp_client, tmp_path, monkeypatch):
    VoiceStore(tmp_path / "voices").add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")

    async def fake_complete(self, session, system, user, schema, max_tokens=2048):
        if "quotes" in schema["properties"]:
            ids = schema["properties"]["quotes"]["items"]["properties"]["id"]["enum"]
            names = {"1": "Alice", "2": "Alice", "3": "Tom", "4": "NARRATOR"}
            deliveries = {"1": "shouting, urgent", "3": "whispering"}
            return {"quotes": [{"id": i, "starts": "", "speaker": names[i], "delivery": deliveries.get(i, "neutral")} for i in ids],
                    "speakers": [{"name": "Alice", "gender": "female", "age": "young adult"},
                                 {"name": "Tom", "gender": "male", "age": "child"}],
                    "_usage": {"total_tokens": 100}}
        return {"characters": [{"name": "Alice", "gender": "female", "age": "young adult", "accent": "british",
                                "temperament": "brave", "voice": "A young British woman, clear and bold."}]}

    monkeypatch.setattr(dz.LocalLlm, "complete", fake_complete)
    cfg = make_cfg(tmp_path, voice_design_engine="fake")
    return await aiohttp_client(make_app(cfg, specs=SPECS))


async def wait_done(client, book="book1"):
    for _ in range(100):
        s = await (await client.get(f"/api/books/{book}/script", headers=AUTH)).json()
        if s["status"] in ("done", "error"):
            return s
        await asyncio.sleep(0.05)
    raise AssertionError("analysis did not finish")


async def test_upload_analyse_correct_and_cast(drama_client):
    c = drama_client
    assert (await c.get("/api/books/book1/script", headers=AUTH)).status == 404
    r = await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    assert r.status == 200
    s = await wait_done(c)
    assert s["status"] == "done", s
    assert s["lines"]["c0p1q0"] == {"speaker": "alice", "delivery": "shouting, urgent", "corrected": False}
    assert s["lines"]["c0p3q14"]["speaker"] is None
    alice = next(ch for ch in s["characters"] if ch["id"] == "alice")
    assert alice["lines"] == 2 and alice["description"].startswith("A young British woman")

    # A user correction wins over the model and survives.
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "alice", "delivery": "scared"}, headers=AUTH)
    line = (await r.json())["lines"]["c0p2q11"]
    assert line == {"speaker": "alice", "delivery": "scared", "corrected": True}

    # Merging Tom into Alice moves his lines and keeps his name as an alias.
    r = await c.patch("/api/books/book1/characters/tom", json={"merge_into": "alice"}, headers=AUTH)
    s = await r.json()
    assert [ch["id"] for ch in s["characters"]] == ["alice"]
    assert "Tom" in s["characters"][0]["aliases"]

    # Casting designs voices (main characters) or pool voices (minor ones) in the background.
    r = await c.post("/api/books/book1/cast", json={}, headers=AUTH)
    assert r.status == 202, await r.text()
    assert (await r.json())["status"] == "casting"
    s = await wait_done(c)
    assert "voices created" in s["detail"], s["detail"]
    vid = s["characters"][0]["voice"]
    assert vid
    voices = await (await c.get("/api/voices?all=1", headers=AUTH)).json()
    assert vid in [v["id"] for v in voices["voices"]]
    # Character voices are hidden from the normal voice list.
    plain = await (await c.get("/api/voices", headers=AUTH)).json()
    assert vid not in [v["id"] for v in plain["voices"]]


async def test_analysis_error_is_reported(aiohttp_client, tmp_path, monkeypatch):
    async def broken(self, *a, **k):
        raise dz.LlmError("No local LLM configured")

    monkeypatch.setattr(dz.LocalLlm, "complete", broken)
    c = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert s["status"] == "error" and "No local LLM" in s["detail"]
