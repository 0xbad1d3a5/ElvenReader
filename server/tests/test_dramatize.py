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

    async def fake_complete(self, session, system, user, schema, max_tokens=2048, **kw):
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


async def test_book_list_text_cover_and_viewer(drama_client):
    import base64

    c = drama_client
    assert (await (await c.get("/api/books", headers=AUTH)).json()) == {"books": []}
    jpeg = b"\xff\xd8\xff\xe0fakejpeg"
    r = await c.post("/api/books/book1", json={**book_payload(), "cover": base64.b64encode(jpeg).decode()}, headers=AUTH)
    assert r.status == 200
    books = (await (await c.get("/api/books", headers=AUTH)).json())["books"]
    assert len(books) == 1
    b = books[0]
    assert (b["id"], b["title"], b["status"], b["cover"], b["chapters"], b["quotes"]) == ("book1", "Test", "none", True, 1, 4)
    assert b["words"] == sum(len(p.split()) for p in PARAS)
    text = await (await c.get("/api/books/book1", headers=AUTH)).json()
    assert text["chapters"][0]["paragraphs"] == PARAS and "cover" not in text
    assert await (await c.get("/api/books/book1/cover", headers=AUTH)).read() == jpeg
    assert (await c.get("/api/books/nope/cover", headers=AUTH)).status == 404
    assert (await c.get("/api/books/nope", headers=AUTH)).status == 404
    assert (await c.get("/api/books", headers={})).status == 401

    # On this PC the viewer pages carry the token for their API calls.
    for path in ("/books", "/books/book1"):
        page = await (await c.get(path)).text()
        assert "AndroidReader books" in page and '"testtoken"' in page and "__TOKEN__" not in page


async def test_viewer_is_served_on_the_lan_without_the_token(drama_client):
    from unittest import mock

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from androidreader_server.app import auth_middleware, viewer_page

    app = drama_client.server.app
    lan = mock.Mock(get_extra_info=lambda *a, **k: ("192.168.1.9", 1234))

    def req(path, headers=None):
        return make_mocked_request("GET", path, headers=headers or {}, app=app, transport=lan)

    async def handler(r):
        return web.Response(text="ok")

    page = await auth_middleware(req("/books/book1"), viewer_page)
    assert page.status == 200 and "const EMBEDDED_TOKEN = null;" in page.text and "testtoken" not in page.text
    assert (await auth_middleware(req("/books"), viewer_page)).status == 200
    # its API calls still need the token
    assert (await auth_middleware(req("/api/books"), handler)).status == 401
    assert (await auth_middleware(req("/api/books", {"Authorization": "Bearer wrong"}), handler)).status == 401
    assert (await auth_middleware(req("/api/books", {"Authorization": "Bearer testtoken"}), handler)).status == 200
    # the pairing page with the token stays on the PC
    with pytest.raises(web.HTTPNotFound):
        await auth_middleware(req("/"), handler)
    index = await (await drama_client.get("/")).text()
    assert "/books#token=testtoken" in index


async def test_reanalyze_keeps_corrections(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    await c.patch("/api/books/book1/lines/c0p1q0", json={"speaker": "tom", "delivery": "calm"}, headers=AUTH)
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True, "reanalyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert s["status"] == "done"
    assert s["lines"]["c0p1q0"] == {"speaker": "tom", "delivery": "calm", "corrected": True}
    assert s["lines"]["c0p2q11"]["speaker"] == "tom"


def test_first_person_narrator_is_tracked_and_prompted():
    script = {"characters": {}, "lines": {}}
    local = {"1": ["c0p0q0"]}
    dz.apply_chunk_result(script, {"narrator": "Sasaki", "speakers": [],
                                   "quotes": [{"id": "1", "starts": "No", "speaker": "Sasaki", "delivery": "firm"}]}, local)
    assert script["narrator"] == "Sasaki" and "sasaki" in script["characters"]
    dz.apply_chunk_result(script, {"narrator": "third person", "speakers": [], "quotes": []}, {})
    assert script["narrator"] == "third person" and list(script["characters"]) == ["sasaki"]
    assert "narrator" in dz.chunk_schema(["1"])["required"]
    assert "(Mary's POV)" in dz.SYSTEM_PROMPT


def test_speaker_resolution_first_person_parentheses_and_aliases():
    script = {"characters": {}, "lines": {}, "narrator": "Sasaki"}
    local = {str(i): [f"q{i}"] for i in range(1, 7)}
    dz.apply_chunk_result(script, {
        "narrator": "I",  # the model restating "I" keeps the known narrator
        "speakers": [
            {"name": "Miss Hoshizaki", "aliases": ["Makeup", "the girl"], "gender": "female", "age": "teen"},
            {"name": "Nick", "aliases": ["Zhang Lee"], "gender": "male", "age": "adult"},
        ],
        "quotes": [
            {"id": "1", "starts": "", "speaker": "I", "delivery": "nervous"},
            {"id": "2", "starts": "", "speaker": "man (Zhang Lee)", "delivery": ""},
            {"id": "3", "starts": "", "speaker": "Makeup", "delivery": ""},
            {"id": "4", "starts": "", "speaker": "I (narrator)", "delivery": ""},
            {"id": "5", "starts": "", "speaker": "a guard", "delivery": ""},
            {"id": "6", "starts": "", "speaker": "NARRATOR", "delivery": ""},
        ],
    }, local)
    speakers = [script["lines"][f"q{i}"]["speaker"] for i in range(1, 7)]
    assert speakers == ["sasaki", "nick", "miss-hoshizaki", "sasaki", "guard", None]
    assert script["characters"]["miss-hoshizaki"]["aliases"] == ["Makeup"]  # "the girl" is a description
    assert script["narrator"] == "Sasaki"
    # An alias never takes over another character's name.
    dz.apply_chunk_result(script, {"narrator": "third person", "quotes": [],
                                   "speakers": [{"name": "Nick", "aliases": ["Miss Hoshizaki"], "gender": "male", "age": "adult"}]}, {})
    assert "Miss Hoshizaki" not in script["characters"]["nick"]["aliases"]


def test_pov_marker_starts_a_new_chunk():
    paras = ["Chapter", "\u201cHi,\u201d I said.", "(The Neighbor\u2019s POV)", "\u201cHello,\u201d I say."]
    quotes = [[f"c0p{pi}q{s}", pi, s, e] for pi, t in enumerate(paras) for s, e in dz.quote_regions(t)]
    book = {"chapters": [{"title": "One", "paragraphs": paras, "quotes": quotes}]}
    chunks = dz.make_chunks(book)
    assert [c.paragraphs for c in chunks] == [[0, 1], [2, 3]]
    assert dz.pov_change(book, chunks[0]) is None
    assert dz.pov_change(book, chunks[1]) == "(The Neighbor\u2019s POV)"
    assert not dz.is_pov_marker("He looked at the point where the road ended and kept walking for a long time.")


def test_narration_follows_first_person_evidence_and_pov_changes():
    Q, E = "\u201c", "\u201d"
    chapters = [
        {"title": "One", "paragraphs": [f"{Q}Hello,{E} I said.", f"{Q}Hi,{E} said Tom.", f"I asked, {Q}Why?{E}"]},
        {"title": "Two", "paragraphs": ["Nothing is said here.", "(The Neighbor\u2019s POV)", f"{Q}Oh,{E} I say.",
                                        "More of her narration."]},
        {"title": "Three", "paragraphs": ["Back to the main narrator, no dialogue."]},
    ]
    for ch in chapters:
        ch["quotes"] = []
        for pi, t in enumerate(ch["paragraphs"]):
            ch["quotes"] += [[f"q{len(ch['quotes'])}-{t[:3]}", pi, s, e] for s, e in dz.quote_regions(t)]
    book = {"chapters": chapters}
    speakers = {"One": ["sasaki", "tom", "sasaki"], "Two": ["neighbor"]}
    lines = {}
    for ch in chapters:
        for (qid, *_), who in zip(ch["quotes"], speakers.get(ch["title"], [])):
            lines[qid] = {"speaker": who, "delivery": None}
    script = {"characters": {}, "lines": lines, "corrections": {}}
    third_person = {"chapters": [{"title": "X", "paragraphs": [f"{Q}Hi,{E} I said."], "quotes": [["a", 0, 0, 5]]}]}
    assert dz.compute_narration(third_person, {"lines": {"a": {"speaker": "tom"}}, "corrections": {}}) == []
    lines["extra"] = {"speaker": "sasaki", "delivery": None}
    chapters[0]["paragraphs"].append(f"{Q}Fine,{E} I replied.")
    chapters[0]["quotes"].append(["extra", 3, 0, 7])
    assert dz.compute_narration(book, script) == [[0, 0, "sasaki"], [1, 1, "neighbor"], [2, 0, "sasaki"]]


def test_reconcile_narrators_by_tense():
    Q, E = "\u201c", "\u201d"
    paras = [f"{Q}A{i},{E} I said." for i in range(6)] + [f"{Q}B{i},{E} I say." for i in range(4)]
    ch = {"title": "One", "paragraphs": paras, "quotes": []}
    for pi, t in enumerate(paras):
        ch["quotes"] += [[f"q{pi}", pi, s, e] for s, e in dz.quote_regions(t)]
    book = {"chapters": [ch]}
    # the LLM kept the point-of-view narrator for two of the hero's past-tense lines
    who = ["hero"] * 4 + ["pov"] * 2 + ["pov"] * 3 + ["hero"]
    script = {"lines": {f"q{i}": {"speaker": w, "delivery": None} for i, w in enumerate(who)}}
    assert dz.reconcile_narrators(book, script) == 3
    assert [script["lines"][f"q{i}"]["speaker"] for i in range(10)] == ["hero"] * 6 + ["pov"] * 4
    # a single narrator who sometimes slips into the present tense is left alone
    single = {"lines": {f"q{i}": {"speaker": "hero", "delivery": None} for i in range(10)}}
    assert dz.reconcile_narrators(book, single) == 0
    # the rest of an interrupted quotation follows its tagged part
    paras.append(f"{Q}Wait,{E} I said. {Q}Come back.{E}")
    ch["quotes"] += [[f"r{k}", 10, s, e] for k, (s, e) in enumerate(dz.quote_regions(paras[10]))]
    script["lines"].update({"r0": {"speaker": "pov"}, "r1": {"speaker": "pov"}})
    assert dz.reconcile_narrators(book, script) == 2
    assert script["lines"]["r1"]["speaker"] == "hero"


def test_two_narrators_are_split_by_narration_tense():
    Q, E = "\u201c", "\u201d"
    past = [f"{Q}Line {i},{E} I said. The room was quiet and we were tired." for i in range(20)]
    marker = ["(The Neighbor\u2019s POV)"]
    present = [f"{Q}Line {i},{E} I say. The room is quiet, we are tired, and it is late." for i in range(15)]
    paras = past + marker + present + past[:15]
    ch = {"title": "One", "paragraphs": paras, "quotes": []}
    for pi, t in enumerate(paras):
        ch["quotes"] += [[f"q{pi}", pi, s, e] for s, e in dz.quote_regions(t)]
    front = {"title": "Copyright", "paragraphs": ["This book is a work of fiction."] * 3, "quotes": []}
    book = {"chapters": [front, ch]}
    who = {f"q{pi}": ("pov" if 20 < pi <= 35 else "hero") for pi in range(len(paras)) if pi != 20}
    script = {"lines": {q: {"speaker": w, "delivery": None} for q, w in who.items()}, "corrections": {}}
    assert dz.compute_narration(book, script) == [[0, 0, "hero"], [1, 20, "pov"], [1, 36, "hero"]]


def test_mixed_tense_books_are_chunked_by_tense_and_hint_the_narrator():
    Q, E = "\u201c", "\u201d"
    past = [f"{Q}Line {i},{E} I said. The room was quiet, we were tired, and it was late." for i in range(15)]
    present = [f"{Q}Line {i},{E} I say. The room is quiet, we are tired, it is late, and he is here." for i in range(15)]
    paras = past + present + past + present
    ch = {"title": "(The Neighbor\u2019s POV)", "paragraphs": paras, "quotes": []}
    for pi, t in enumerate(paras):
        ch["quotes"] += [[f"q{pi}", pi, s, e] for s, e in dz.quote_regions(t)]
    book = {"chapters": [ch]}
    runs = dz.mixed_tense_runs(book)
    assert runs == [(0, 0, "past"), (0, 15, "present"), (0, 30, "past"), (0, 45, "present")]
    chunks = dz.make_chunks(book, runs=runs)
    assert [(c.paragraphs[0], c.tense) for c in chunks] == [(0, "past"), (15, "present"), (30, "past"), (45, "present")]
    # one present-tense stretch (back matter) is not a narrator switch
    assert dz.mixed_tense_runs({"chapters": [{**ch, "paragraphs": past + present}]}) is None

    seen = []

    class FakeLlm:
        async def complete(self, session, system, user, schema, max_tokens=0, **kw):
            seen.append(user)
            return {"narrator": "", "speakers": [], "quotes": []}

    script = {"characters": {}, "lines": {}, "narrator": "Hero", "narrator_by_tense": {"past": "Hero", "present": "Neighbor"},
              "last_tense": "past"}
    asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, chunks[1]))
    assert "earlier present-tense passages were narrated by Neighbor" in seen[-1]
    assert "point-of-view change" not in seen[-1]  # the chapter title is not trusted when tense is known
    script["narrator_by_tense"] = {"past": "Hero"}
    asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, chunks[1]))
    assert "switches from the past to the present tense" in seen[-1]


# ---------------------------------------------------------------- dossier (pass 1) and series memory


def test_dossier_merge_revises_and_tracks_narrator_and_scenes():
    from androidreader_server import dossier as ds

    script = {"characters": {}, "lines": {}}
    first = {"narrator": "Sasaki", "present": ["Sasaki", "the old woman"], "characters": [
        {"name": "Futarishizuka", "aliases": ["Grandmother"], "gender": "female", "apparent_age": "elderly",
         "speech_style": "calls people 'dearie'", "facts": [{"fact": "talks like a granny", "quote": "dearie"}]}]}
    present = ds.merge_update(script, first, (3, 0), "past")
    c = script["characters"]["futarishizuka"]
    assert c["age"] == "elderly" and c["dossier"]["speech_style"] == "calls people 'dearie'"
    assert "Grandmother" in c["aliases"] and "sasaki" in present
    # a later passage corrects the apparent age; the revision is kept with where it happened
    later = {"narrator": "third person", "present": [], "characters": [
        {"name": "Grandmother", "apparent_age": "child", "form": "human, looks like a girl of ten",
         "true_nature": "centuries old", "revised": "she only talks old; she looks like a little girl"}]}
    ds.merge_update(script, later, (9, 40), None)
    assert c["age"] == "child" and c["dossier"]["true_nature"] == "centuries old"
    assert c["dossier"]["revisions"][0]["at"] == [9, 40]
    # the user's word wins over the model's, and survives later model updates
    c["dossier_user"] = {"apparent_age": "teen"}
    ds.sync_character(c)
    ds.merge_update(script, later, (10, 0), None)
    assert ds.effective(c)["apparent_age"] == "teen"
    assert ds.narrators(script) == {"past": "sasaki"}
    text = ds.render_dossier(script)
    assert "Futarishizuka (also: Grandmother): female, teen" in text and "really: centuries old" in text
    script["dossier_state"]["scenes"] = [[3, 0, 20, ["futarishizuka"]], [4, 0, 9, ["sasaki"]]]
    assert ds.scene_cast(script, 3, [15, 30]) == ["futarishizuka"]
    assert "teen female" in ds.voice_description(c)


async def test_two_pass_analysis_uses_the_dossier_and_keeps_user_edits(aiohttp_client, tmp_path, monkeypatch):
    prompts = {"dossier": 0, "label": []}

    async def fake(self, session, system, user, schema, max_tokens=2048, **kw):
        if "characters" in schema["properties"] and "present" in schema["properties"]:
            prompts["dossier"] += 1
            assert kw.get("thinking") is True
            return {"narrator": "third person", "present": ["Alice", "Tom"], "characters": [
                {"name": "Tom", "aliases": ["the boy"], "gender": "male", "apparent_age": "child",
                 "speech_style": "says 'gosh' a lot", "introduced_as": "a boy by the river"}]}
        if "quotes" in schema["properties"]:
            prompts["label"].append(user)
            ids = schema["properties"]["quotes"]["items"]["properties"]["id"]["enum"]
            return {"narrator": "third person", "speakers": [],
                    "quotes": [{"id": i, "starts": "", "speaker": "Tom", "delivery": "neutral"} for i in ids]}
        return {"characters": []}

    monkeypatch.setattr(dz.LocalLlm, "complete", fake)
    c = await aiohttp_client(make_app(make_cfg(tmp_path, llm_dossier_model="reasoner.gguf"), specs=SPECS))
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert s["status"] == "done" and prompts["dossier"] == 1
    assert "says 'gosh' a lot" in prompts["label"][0] and "Present in this part of the story: Alice, Tom" in prompts["label"][0]
    tom = next(ch for ch in s["characters"] if ch["id"] == "tom")
    assert tom["age"] == "child" and tom["dossier"]["introduced_as"] == "a boy by the river"
    assert "alice" not in [ch["id"] for ch in s["characters"]]  # present but never speaks: pruned

    # Edit the dossier; relabel runs pass 2 only, with the edit; reanalyze reads again but keeps the edit.
    r = await c.patch("/api/books/book1/characters/tom", json={"dossier": {"speech_style": "stammers"}}, headers=AUTH)
    assert next(ch for ch in (await r.json())["characters"] if ch["id"] == "tom")["dossier_user"] == {"speech_style": "stammers"}
    await c.post("/api/books/book1/analyze", json={"relabel": True}, headers=AUTH)
    await wait_done(c)
    assert prompts["dossier"] == 1 and "stammers" in prompts["label"][-1]
    await c.post("/api/books/book1/analyze", json={"reanalyze": True}, headers=AUTH)
    s = await wait_done(c)
    tom = next(ch for ch in s["characters"] if ch["id"] == "tom")
    assert prompts["dossier"] == 2 and tom["dossier_user"] == {"speech_style": "stammers"}


def test_series_name_from_titles():
    from androidreader_server.series import series_name

    assert series_name("Sasaki and Peeps, Vol. 10") == "Sasaki and Peeps"
    assert series_name("Sasaki and Peeps v10 [Yen Press]") == "Sasaki and Peeps"
    assert series_name("The Wandering Inn: Volume 3") == "The Wandering Inn"
    assert series_name("Mistborn Book 2") == "Mistborn"
    assert series_name("Alice's Adventures in Wonderland") == "Alice's Adventures in Wonderland"


async def test_series_memory_carries_cast_and_voices_to_the_next_volume(drama_client):
    from androidreader_server.dramatize import DRAMA_STORE_KEY

    c = drama_client
    app = c.server.app
    await c.post("/api/books/vol1", json={**book_payload(), "title": "Tea Saga, Vol. 1", "analyze": True}, headers=AUTH)
    await wait_done(c, "vol1")
    r = await c.patch("/api/books/vol1/characters/alice", json={"voice": "nar", "dossier": {"apparent_age": "child"}},
                      headers=AUTH)
    assert r.status == 200
    series = await (await c.get("/api/series", headers=AUTH)).json()
    entry = series["series"][0]
    assert entry["name"] == "Tea Saga" and entry["characters"]["alice"]["voice"] == "nar"

    store = app[DRAMA_STORE_KEY]
    store.save_book("vol2", {**book_payload(), "title": "Tea Saga, Vol. 2"})
    from androidreader_server import series as sr

    s2 = store.script("vol2")
    assert sr.seed(app, "vol2", s2) >= 1
    assert s2["characters"]["alice"]["voice"] == "nar"
    assert s2["characters"]["alice"]["dossier_user"]["apparent_age"] == "child"
    # analysing the next volume starts from the series cast (pass 1 off by default)
    await c.post("/api/books/vol3", json={**book_payload(), "title": "Tea Saga, Vol. 3", "analyze": True}, headers=AUTH)
    s3 = await wait_done(c, "vol3")
    assert next(ch for ch in s3["characters"] if ch["id"] == "alice")["voice"] == "nar"
    # a book's series can be set by hand (e.g. a spin-off)
    r = await c.post("/api/books/vol2/series", json={"series": "Tea Saga Side Stories"}, headers=AUTH)
    assert (await r.json())["series"] == "Tea Saga Side Stories"
    assert store.summary("vol2")["series"] == "Tea Saga Side Stories"


def test_prune_keeps_speakers_narrators_user_edits_and_chosen_voices():
    script = {"characters": {
        "a": {"name": "A"}, "b": {"name": "B"}, "n": {"name": "N"}, "u": {"name": "U", "user": True},
        "v": {"name": "V", "voice": "560fa464"}, "p": {"name": "P", "voice": "pool-male-adult"}},
        "lines": {"q1": {"speaker": "a"}}, "corrections": {}, "narrator": "N"}
    assert sorted(dz.prune_characters(script)) == ["b", "p"]


def test_consolidation_merges_duplicates_drops_groups_and_fills_gaps():
    from androidreader_server import dossier as ds

    script = {"characters": {}, "lines": {}, "corrections": {}}
    ds.merge_update(script, {"narrator": "", "present": ["Robot Girl"], "characters": [
        {"name": "Type Twelve", "form": "mechanical life-form"}, {"name": "Robot Girl", "speech_style": "formal"},
        {"name": "Four Large Men"}, {"name": "Makeup", "gender": "female"}]}, (1, 0), None)
    script["characters"]["makeup"]["user"] = True  # the user's own entry is never folded away

    class FakeLlm:
        cfg = type("C", (), {"llm_dossier_effort": "high"})

        async def complete(self, *a, **k):
            return {"merges": [{"keep": "Type Twelve", "same_as": ["Robot Girl"], "why": "same android"},
                               {"keep": "Type Twelve", "same_as": ["Makeup"], "why": "wrong"}],
                    "drop": ["Four Large Men"],
                    "fixes": [{"name": "Type Twelve", "gender": "female", "apparent_age": "child"}]}

    async def step(make):
        return await make()

    done = asyncio.run(ds.consolidate(FakeLlm(), None, {"title": "T"}, script, "m.gguf", step))
    t = script["characters"]["type-twelve"]
    assert done["merged"] == ["Robot Girl -> Type Twelve"] and done["dropped"] == ["Four Large Men"]
    assert "Robot Girl" in t["aliases"] and t["dossier"]["speech_style"] == "formal"
    assert (t["gender"], t["age"]) == ("female", "child")
    assert script["dossier_state"]["scenes"] if "scenes" in script["dossier_state"] else True
    assert "makeup" in script["characters"] and "robot-girl" not in script["characters"]
