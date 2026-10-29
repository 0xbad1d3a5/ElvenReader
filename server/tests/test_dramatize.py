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
        item = (schema["properties"].get("characters") or {}).get("items", {}).get("properties", {})
        if "role" in item:  # the tier question
            roles = {"Alice": ("main", "the heroine"), "Tom": ("extra", "one scene")}
            return {"characters": [{"name": n, "role": roles.get(n, ("extra", ""))[0], "reason": roles.get(n, ("", "?"))[1]}
                                   for n in item["name"]["enum"]]}
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
    assert "Story just before this passage" not in seen[-1]

    # more context: the preceding text and more recent lines; a review lists the proposed labels
    script["lines"] = {f"q{pi}": {"speaker": None} for pi in range(15)}
    asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, chunks[1], context_words=30, recent=12))
    before = seen[-1].split("Story just before this passage (for context only):\n")[1].split("\n\nMost recent")[0]
    assert before.split("\n\n") == paras[13:15]  # whole paragraphs (15 words each) until >= 30 words, in order
    assert seen[-1].count("NARRATOR: ") == 12
    _, local = dz.render_chunk(book, chunks[1])
    asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, chunks[1], proposed={"1": "Neighbor"}))
    assert dz.REVIEW_NOTE in seen[-1] and "[1] Neighbor\n[2] ?" in seen[-1] and len(local) > 1


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

    # Edit the profile (for voice design); relabel runs pass 2 only; the edit shapes voices, not who says
    # what, so it is not in the labelling prompt; reanalyze reads again but keeps the edit.
    r = await c.patch("/api/books/book1/characters/tom", json={"profile": {"speech_style": "stammers"}}, headers=AUTH)
    assert next(ch for ch in (await r.json())["characters"] if ch["id"] == "tom")["profile_user"] == {"speech_style": "stammers"}
    await c.post("/api/books/book1/analyze", json={"relabel": True}, headers=AUTH)
    await wait_done(c)
    assert prompts["dossier"] == 1 and "stammers" not in prompts["label"][-1]
    await c.post("/api/books/book1/analyze", json={"reanalyze": True}, headers=AUTH)
    s = await wait_done(c)
    tom = next(ch for ch in s["characters"] if ch["id"] == "tom")
    assert prompts["dossier"] == 2 and tom["profile_user"] == {"speech_style": "stammers"}


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
    r = await c.patch("/api/books/vol1/characters/alice", json={"voice": "nar", "profile": {"apparent_age": "child"}},
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
    assert s2["characters"]["alice"]["profile_user"]["apparent_age"] == "child"
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


def test_pov_markers_are_headings_not_sentences():
    assert dz.is_pov_marker("(The Neighbor\u2019s POV)")
    assert dz.is_pov_marker("Suiran\u2019s Side")
    assert dz.is_pov_marker("(Suiran's side)")
    assert not dz.is_pov_marker("And as I watched, he slammed straight into Suiran\u2019s side.")
    assert not dz.is_pov_marker("She stood by my side.")


def test_a_book_re_split_by_a_newer_parser_keeps_its_analysis():
    Q, E = "\u201c", "\u201d"
    paras = [f"{Q}Hi,{E} said Tom.", "Narration.", f"{Q}Bye,{E} said Ann."]

    def book(split):
        chapters, start = [], 0
        for n in split:
            ps = paras[start:start + n]
            quotes = [[f"c{len(chapters)}p{i}q{s}", i, s, e] for i, t in enumerate(ps) for s, e in dz.quote_regions(t)]
            chapters.append({"title": f"T{len(chapters)}", "paragraphs": ps, "quotes": quotes})
            start += n
        return {"chapters": chapters}

    old, new = book([1, 2]), book([3])
    script = {"status": "done", "characters": {"tom": {"name": "Tom"}, "ann": {"name": "Ann"}},
              "lines": {"c0p0q0": {"speaker": "tom"}, "c1p1q0": {"speaker": "ann"}},
              "corrections": {"c1p1q0": {"speaker": "ann", "delivery": "sad"}}, "chunks_done": ["c0k0"]}
    assert dz.migrate_ids(old, new, script)
    assert set(script["lines"]) == {"c0p0q0", "c0p2q0"} and script["corrections"] == {"c0p2q0": {"speaker": "ann", "delivery": "sad"}}
    assert script["chunks_done"] == []
    assert not dz.migrate_ids(new, new, script)  # same split
    assert not dz.migrate_ids(old, {"chapters": [{"paragraphs": ["other"]}]}, script)  # different text


async def test_upload_with_a_new_split_migrates_and_new_speakers_get_voices(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    await c.post("/api/books/book1/cast", json={}, headers=AUTH)
    s = await wait_done(c)
    assert all(ch["voice"] for ch in s["characters"])
    # a new speaker appears in a re-analysis: it gets a voice without casting again by hand
    store = c.server.app[dz.DRAMA_STORE_KEY]
    raw = store.script("book1")
    raw["characters"]["zed"] = {"name": "Zed", "aliases": [], "gender": "male", "age": "adult", "voice": None}
    raw["corrections"]["c0p2q11"] = {"speaker": "zed"}
    store.save_script("book1", raw)
    await c.post("/api/books/book1/analyze", json={"relabel": True}, headers=AUTH)
    s = await wait_done(c)
    for _ in range(50):
        s = await (await c.get("/api/books/book1/script", headers=AUTH)).json()
        if next(ch for ch in s["characters"] if ch["id"] == "zed").get("voice"):
            break
        await asyncio.sleep(0.05)
    assert next(ch for ch in s["characters"] if ch["id"] == "zed")["voice"]


def test_unmarked_end_of_a_pov_section_is_found_exactly():
    Q, E = "\u201c", "\u201d"
    past = [f"The room was quiet and we were tired, it was late {i}." for i in range(20)]
    present = [f"The room is quiet and we are tired, it is late {i}." for i in range(20)]
    talk = [f"{Q}Line {i}.{E}" for i in range(3)]  # dialogue only: no tense of its own
    paras = past + ["(The Neighbor\u2019s POV)"] + present + talk + past
    book = {"chapters": [{"title": "One", "paragraphs": paras, "quotes": []}]}
    runs = dz.tense_runs(book)
    assert runs == [(0, 0, "past"), (0, 20, "present"), (0, 44, "past")]


# ---------------------------------------------------------------- quoted terms are not dialogue


def test_quoted_terms_inside_a_sentence_are_mentions_not_speech():
    Q, E = "\u201c", "\u201d"

    def m(text):
        s = text.index(Q)
        return dz.is_mention(text, s, text.index(E, s) + 1)

    assert m(f"I remembered that mao was Chinese for {Q}cat.{E} Perhaps")
    assert m(f"Because of all the {Q}inspections{E} we\u2019d been conducting")
    assert m(f"his pet Java sparrow is {Q}OP.{E} He told me")
    assert m(f"I\u2019d heard them say something about {Q}ink,{E} but I hadn\u2019t")
    assert not m(f"{Q}Call me Nick.{E}")  # starts the paragraph
    assert not m(f"The man tensed. {Q}Where did you hear that name?{E}")  # after a full stop
    assert not m(f"Alice considered a little, and then said {Q}The fourth.{E}")  # speech verb, no comma
    assert not m(f"the March Hare went {Q}Sh! sh!{E} and the Dormouse")  # exclamation inside
    assert not m(f"then, saying to herself {Q}This is Bill,{E} she gave one sharp kick")  # verb to someone
    assert not m(f"he said softly {Q}Go home{E} and left")
    assert not m(f"singing in its sleep {Q}Twinkle, twinkle, twinkle{E}")  # inner commas


def test_mentions_are_narration_and_corrections_still_win():
    Q, E = "\u201c", "\u201d"
    paras = [f"{Q}Hello,{E} said Bob.", f"Mao is Chinese for {Q}cat.{E} I knew that."]
    ch = {"title": "One", "paragraphs": paras, "quotes": []}
    for pi, t in enumerate(paras):
        ch["quotes"] += [[f"q{pi}", pi, s, e] for s, e in dz.quote_regions(t)]
    book = {"chapters": [ch]}
    chunk = dz.make_chunks(book)[0]

    class FakeLlm:  # the model gets the term wrong
        async def complete(self, session, system, user, schema, max_tokens=0, **kw):
            return {"narrator": "Sasaki", "speakers": [],
                    "quotes": [{"id": "1", "speaker": "Bob", "delivery": "neutral"},
                               {"id": "2", "speaker": "Sasaki", "delivery": "neutral"}]}

    script = {"characters": {}, "lines": {}, "corrections": {}}
    result, local = asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, chunk))
    dz.apply_chunk_result(script, result, local)
    assert script["lines"]["q1"] == {"speaker": None, "delivery": None, "mention": True}
    assert script["lines"]["q0"]["speaker"] == dz.find_character(script, "Bob")
    pub = dz.ScriptStore.public(script)["lines"]
    assert pub["q1"]["mention"] and "mention" not in pub["q0"]

    # books analysed before: mark_mentions fixes them; a reader's correction still wins
    script["lines"]["q1"] = {"speaker": dz.find_character(script, "Sasaki"), "delivery": None}
    assert dz.mark_mentions(book, script) == 1 and script["lines"]["q1"]["speaker"] is None
    script["corrections"]["q1"] = {"speaker": dz.find_character(script, "Bob")}
    pub = dz.ScriptStore.public(script)["lines"]
    assert pub["q1"]["speaker"] == dz.find_character(script, "Bob") and "mention" not in pub["q1"]


async def test_book_pictures_are_listed_and_served(drama_client):
    c = drama_client
    payload = book_payload()
    payload["chapters"][0]["images"] = {"1": ["OEBPS/images/p1.jpg"]}
    payload["chapters"][0]["end_images"] = ["OEBPS/images/end.png", "OEBPS/images/p1.jpg"]
    assert (await c.post("/api/books/book1", json=payload, headers=AUTH)).status == 200
    assert dz.picture_refs((await (await c.get("/api/books/book1", headers=AUTH)).json())) == \
        ["OEBPS/images/p1.jpg", "OEBPS/images/end.png"]
    lst = await (await c.get("/api/books/book1/pictures", headers=AUTH)).json()
    assert lst == {"have": [], "need": ["OEBPS/images/p1.jpg", "OEBPS/images/end.png"]}
    q = "/api/books/book1/picture?path=OEBPS%2Fimages%2Fp1.jpg"
    assert (await c.get(q, headers=AUTH)).status == 404  # not stored yet
    f = c.server.app[dz.DRAMA_STORE_KEY].picture_file("book1", "OEBPS/images/p1.jpg")  # as an import stores it
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\xff\xd8jpeg")
    r = await c.get(q, headers=AUTH)
    assert r.status == 200 and r.headers["Content-Type"] == "image/jpeg" and await r.read() == b"\xff\xd8jpeg"
    assert (await c.get(q)).status == 401  # token needed
    # only pictures the book refers to, no path tricks
    assert (await c.get("/api/books/book1/picture?path=..%2F..%2Fconfig.json", headers=AUTH)).status == 404
    lst = await (await c.get("/api/books/book1/pictures", headers=AUTH)).json()
    assert lst == {"have": ["OEBPS/images/p1.jpg"], "need": ["OEBPS/images/end.png"]}


def test_later_passages_may_revise_earlier_lines_but_never_corrections():
    Q, E = "\u201c", "\u201d"
    paras = [f"{Q}Who are you?{E} the man asked.", f"{Q}Nobody,{E} I said.", f"{Q}Where from?{E}",
             f"{Q}I go by Nick here,{E} said the man."]
    ch = {"title": "One", "paragraphs": paras, "quotes": []}
    for pi, t in enumerate(paras):
        ch["quotes"] += [[f"q{pi}", pi, s, e] for s, e in dz.quote_regions(t)]
    book = {"chapters": [ch]}
    later = dz.Chunk("c0k1", 0, [3], [tuple(ch["quotes"][3])])
    script = {"characters": {}, "lines": {}, "corrections": {}, "narrator": "Sasaki"}
    prisoner, me = dz.ensure_character(script, "a prisoner"), dz.ensure_character(script, "Sasaki")
    script["lines"] = {"q0": {"speaker": prisoner, "delivery": None}, "q1": {"speaker": me, "delivery": None},
                       "q2": {"speaker": prisoner, "delivery": None}}
    script["corrections"] = {"q2": {"speaker": prisoner}}  # the reader decided this one
    seen = []

    class FakeLlm:
        async def complete(self, session, system, user, schema, max_tokens=0, **kw):
            seen.append((user, schema))
            return {"narrator": "Sasaki", "speakers": [{"name": "Nick", "aliases": [], "gender": "male", "age": "adult"}],
                    "quotes": [{"id": "1", "speaker": "Nick", "delivery": "neutral"}],
                    "revisions": [{"id": "R1", "starts": "Who are you", "speaker": "Nick", "reason": "he gives his name: 'I go by Nick'"},
                                  {"id": "R2", "starts": "I go by", "speaker": "Nick", "reason": "aimed at the wrong line"},
                                  {"id": "R2", "starts": "Nobody", "speaker": "Sasaki", "reason": "unchanged"}]}

    result, local = asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, later, revise=12))
    user, schema = seen[-1]
    assert "[R1] Prisoner: \u201cWho are you?\u201d" in user and "[R2] Sasaki" in user
    assert "Where from" not in user.split(dz.REVISE_NOTE)[1]  # corrected lines are not offered
    assert schema["properties"]["revisions"]["items"]["properties"]["id"]["enum"] == ["R1", "R2"]
    dz.apply_chunk_result(script, result, local)
    nick = dz.find_character(script, "Nick")
    assert script["lines"]["q0"]["speaker"] == nick and script["lines"]["q0"]["revised"]
    assert script["lines"]["q1"]["speaker"] == me and "revised" not in script["lines"]["q1"]
    assert script["revisions"] == [{"id": "q0", "from": prisoner, "to": nick, "reason": "he gives his name: 'I go by Nick'"}]
    assert dz.ScriptStore.public(script)["lines"]["q0"]["revised"]
    # unnamed-only: Sasaki's line is not offered; off by default: no revisions in the schema
    asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, later, revise=12, revise_unnamed=True))
    assert "[R1]" not in seen[-1][0]  # q0 is now Nick, q1 Sasaki: nothing unnamed left
    asyncio.run(dz.label_chunk(FakeLlm(), None, book, script, later))
    assert "revisions" not in seen[-1][1]["properties"]


# ---------------------------------------------------------------- character tiers (main / supporting / extra)


def test_tiers_follow_the_model_within_guard_rails_and_the_reader_wins():
    from androidreader_server import roles

    ev = {"name": "X", "lines": 10, "share": 0.002, "chapters_with_lines": 1, "chapters_mentioned": 1,
          "first": 1, "last": 1, "named": True, "narrates": False, "chapters": 10}
    c = {"name": "X"}
    roles.settle(c, ev, "supporting", "recurring")
    assert (c["role"], c["role_auto"], c["role_source"], c["role_note"]) == ("supporting", "supporting", "llm", "recurring")
    roles.settle(c, {**ev, "share": 0.06}, "extra")  # never an extra with 5% of the lines
    assert c["role"] == "supporting" and "6%" in c["role_note"]
    roles.settle(c, {**ev, "narrates": True}, "supporting")  # narrators are main
    assert c["role"] == "main"
    roles.settle(c, {**ev, "lines": 1, "share": 0.09}, "extra")  # far from the counts: flagged
    assert c["role"] == "supporting" and c["role_flag"] == "the line counts suggest main"
    roles.settle(c, {**ev, "named": False}, "supporting")  # unnamed walk-ons are extras
    assert c["role"] == "extra" and "unnamed" in c["role_note"]
    roles.settle(c, ev, None)  # no answer: the rules
    assert c["role"] == "extra" and c["role_source"] == "rule" and "role_flag" not in c
    c["role_series"] = "main"  # the series keeps a main character main
    roles.settle(c, ev, "extra")
    assert c["role"] == "main"
    roles.set_user_role(c, "extra")  # ...unless the reader says otherwise
    roles.settle(c, ev, "main")
    assert c["role"] == "extra" and c["role_auto"] == "main"
    roles.set_user_role(c, None)
    assert c["role"] == "main" and "role_user" not in c
    assert roles.rule_role({**ev, "share": 0.1}) == "main" and roles.rule_role({**ev, "named": False}) == "extra"
    assert roles.rule_role({**ev, "chapters_with_lines": 2}) == "supporting"


def test_casting_plan_by_tier():
    s = {"characters": {"a": {"name": "A", "role": "main"}, "b": {"name": "B", "role": "supporting"},
                        "c": {"name": "C", "role": "supporting"}, "d": {"name": "D", "role": "extra"},
                        "e": {"name": "E"}, "n": {"name": "N", "role": "extra"}},
         "lines": {**{f"b{i}": {"speaker": "b"} for i in range(3)}, "c1": {"speaker": "c"},
                   **{f"d{i}": {"speaker": "d"} for i in range(9)}, **{f"e{i}": {"speaker": "e"} for i in range(5)}},
         "narration": [[0, 0, "n"]]}
    assert dz.own_voice_plan(s, 1) == {"a", "b", "e", "n"}  # supporting by lines within budget; untiered by count
    assert dz.own_voice_plan(s, 5) == {"a", "b", "c", "e", "n"}  # extras never, however many lines


async def test_tiers_from_analysis_override_series_and_bulk_voices(drama_client):
    from androidreader_server import series as sr
    from androidreader_server.dramatize import DRAMA_STORE_KEY

    c = drama_client
    app = c.server.app
    await c.post("/api/books/vol1", json={**book_payload(), "title": "Tea Saga, Vol. 1", "analyze": True}, headers=AUTH)
    s = await wait_done(c, "vol1")
    tiers = {ch["id"]: (ch["role"], ch["role_source"]) for ch in s["characters"]}
    # the model says extra, but Tom speaks a third of this tiny book's lines: never an extra then
    assert tiers == {"alice": ("main", "llm"), "tom": ("supporting", "llm")}
    # the reader's tier wins and survives a re-analysis
    r = await c.patch("/api/books/vol1/characters/tom", json={"role": "extra"}, headers=AUTH)
    assert next(ch for ch in (await r.json())["characters"] if ch["id"] == "tom")["role"] == "extra"
    await c.post("/api/books/vol1/analyze", json={"relabel": True}, headers=AUTH)
    s = await wait_done(c, "vol1")
    tom = next(ch for ch in s["characters"] if ch["id"] == "tom")
    assert (tom["role"], tom["role_user"], tom["role_auto"]) == ("extra", "extra", "supporting")
    # sorting the cast on its own (no re-labelling) writes the tiers to the series
    r = await c.post("/api/books/vol1/roles", headers=AUTH)
    assert r.status == 202
    for _ in range(100):
        s = await (await c.get("/api/books/vol1/script", headers=AUTH)).json()
        if not (s.get("roles_state") or {}).get("running"):
            break
        await asyncio.sleep(0.05)
    assert s["roles_state"].get("done"), s.get("roles_state")
    assert next(ch for ch in s["characters"] if ch["id"] == "tom")["role"] == "extra"
    # the series remembers: Alice stays main, Tom keeps the reader's tier
    entry = (await (await c.get("/api/series", headers=AUTH)).json())["series"][0]["characters"]
    assert entry["alice"]["role"] == "main" and entry["tom"]["role_user"] == "extra"
    store = app[DRAMA_STORE_KEY]
    store.save_book("vol2", {**book_payload(), "title": "Tea Saga, Vol. 2"})
    s2 = store.script("vol2")
    sr.seed(app, "vol2", s2)
    assert s2["characters"]["alice"]["role_series"] == "main" and s2["characters"]["tom"]["role"] == "extra"
    # bulk "own voices" for supporting characters; tiers on the Voices page
    await c.patch("/api/books/vol1/characters/tom", json={"role": "supporting"}, headers=AUTH)
    s1 = store.script("vol1")
    s1["characters"]["tom"]["voice"] = "pool-male-child"
    store.save_script("vol1", s1)
    await c.post("/api/books/vol1/cast", json={"own_voices_for": ["supporting"]}, headers=AUTH)
    s = await wait_done(c, "vol1")
    assert next(ch for ch in s["characters"] if ch["id"] == "tom")["voice"].startswith("ch-")
    v = await (await c.get(f"/api/voices/{next(ch for ch in s['characters'] if ch['id'] == 'alice')['voice']}?details=1",
                           headers=AUTH)).json()
    assert any(u.get("role") == "main" for u in v.get("used_by", [])), v


def test_series_tier_floor_comes_from_other_volumes_only():
    from androidreader_server.series import _higher

    assert _higher("extra", "main") == "main"  # main in an earlier volume stays main
    assert _higher("main", "supporting") == "main"
    assert _higher("extra", None) == "extra"  # this volume re-sorted: just replaced
    assert _higher("extra", "main", user=True) == "extra"  # the reader's choice wins


def test_descriptive_names_are_shown_capitalised_and_still_match():
    assert dz.normalize_character_name("a prisoner") == ("Prisoner", True)
    assert dz.normalize_character_name("the tattoo artist") == ("Tattoo Artist", True)
    assert dz.normalize_character_name("crowd") == ("Crowd", True)
    assert dz.normalize_character_name("man in the hat") == ("Man in the Hat", True)
    assert dz.normalize_character_name("the Neighbor") == ("The Neighbor", False)
    for kept in ("Type Twelve", "Mr. Akutsu", "Magical Pink", "TV Announcer"):
        assert dz.normalize_character_name(kept) == (kept, False)
    script = {"characters": {}, "lines": {}}
    cid = dz.ensure_character(script, "a prisoner")
    c = script["characters"][cid]
    assert (c["name"], c["aliases"], c["unnamed"]) == ("Prisoner", ["a prisoner"], True)
    assert dz.find_character(script, "a prisoner") == cid == dz.find_character(script, "the prisoner")
    # books analysed earlier are migrated; the tier evidence still sees an unnamed walk-on
    old = {"characters": {"guard": {"name": "a guard", "aliases": []}, "akutsu": {"name": "Mr. Akutsu", "aliases": []}},
           "lines": {}}
    assert dz.normalize_names(old) == 1
    assert old["characters"]["guard"] == {"name": "Guard", "aliases": ["a guard"], "unnamed": True}
    assert old["characters"]["akutsu"]["name"] == "Mr. Akutsu" and "unnamed" not in old["characters"]["akutsu"]
    from androidreader_server import roles

    ev = roles.evidence({"chapters": []}, {**old, "characters": old["characters"]})
    assert ev["guard"]["named"] is False and ev["akutsu"]["named"] is True


def test_model_names_are_normalised_on_the_analysis_path():
    script = {"characters": {}, "lines": {}}
    local = {"1": ["a"], "2": ["b"], "3": ["c"], "4": ["d"]}
    dz.apply_chunk_result(script, {
        "narrator": "third person",
        "speakers": [{"name": "a prisoner", "aliases": [], "gender": "male", "age": "adult"},
                     {"name": "the tattoo artist", "aliases": [], "gender": "male", "age": "adult"}],
        "quotes": [{"id": "1", "speaker": "a prisoner", "delivery": "neutral"},
                   {"id": "2", "speaker": "the tattoo artist", "delivery": "neutral"},
                   {"id": "3", "speaker": "crowd", "delivery": "shouting"},
                   {"id": "4", "speaker": "the prisoner", "delivery": "neutral"}]}, local)
    names = {cid: c["name"] for cid, c in script["characters"].items()}
    assert sorted(names.values()) == ["Crowd", "Prisoner", "Tattoo Artist"]
    assert script["lines"]["a"]["speaker"] == script["lines"]["d"]["speaker"]  # still matched by description
    assert all(c["unnamed"] for c in script["characters"].values())


async def test_names_the_reader_types_are_kept_exactly(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "the butler", "delivery": ""}, headers=AUTH)
    s = await r.json()
    assert s["characters"][[ch["id"] for ch in s["characters"]].index(s["lines"]["c0p2q11"]["speaker"])]["name"] == "the butler"
    r = await c.patch("/api/books/book1/characters/tom", json={"name": "tom the piper's son"}, headers=AUTH)
    assert any(ch["name"] == "tom the piper's son" for ch in (await r.json())["characters"])
    s = c.server.app[dz.DRAMA_STORE_KEY].script("book1")
    s.pop("names_v", None)  # the one-time migration leaves reader-typed names alone
    dz.normalize_names(s)
    assert {"the butler", "tom the piper's son"} <= {ch["name"] for ch in s["characters"].values()}


async def test_characters_are_described_by_their_own_voice(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    # a voice chosen later replaces the stale description
    r = await c.patch("/api/voices/nar", json={"description": "A calm, warm storyteller's voice."}, headers=AUTH)
    assert r.status == 200
    r = await c.patch("/api/books/book1/characters/alice", json={"voice": "nar"}, headers=AUTH)
    alice = next(ch for ch in (await r.json())["characters"] if ch["id"] == "alice")
    assert alice["description"] == "A calm, warm storyteller's voice."
    # editing that voice later updates everyone it voices
    await c.patch("/api/voices/nar", json={"description": "A young girl's voice, old-fashioned and teasing."}, headers=AUTH)
    s = await (await c.get("/api/books/book1/script", headers=AUTH)).json()
    assert next(ch for ch in s["characters"] if ch["id"] == "alice")["description"].startswith("A young girl's voice")
    # shared pool voices don't overwrite a character's own description
    pool = {"name": "x", "voice": "pool-male-adult", "description": "A gruff old sailor."}

    class V:
        description, instruct = "An adult man with an ordinary voice.", ""
    assert dz.follow_voice(pool, {"pool-male-adult": V()}) is False and pool["description"] == "A gruff old sailor."


# ---------------------------------------------------------------- voice profiles


def test_profile_updates_merge_specifics_and_never_touch_the_readers_fields():
    from androidreader_server import profile as pf

    c = {"name": "F"}
    assert pf.merge(c, {"gender": "female", "apparent_age": "unknown", "appearance": "young girl",
                        "speech_style": "teasing", "notes": "centuries old"})
    assert pf.merge(c, {"apparent_age": "child", "appearance": "young girl in a kimono",  # more specific: replaces
                        "speech_style": "calls people dearie", "notes": "Centuries old; sly"})
    assert not pf.merge(c, {"gender": "male", "appearance": "tall woman"})  # set fields are not overturned
    assert c["profile"] == {"gender": "female", "apparent_age": "child", "appearance": "young girl in a kimono",
                            "speech_style": "teasing; calls people dearie", "notes": "centuries old; sly"}
    pf.set_user_fields(c, {"appearance": "girl in a kimono, bob cut", "gender": "female"})  # same as model: not kept
    assert c["profile_user"] == {"appearance": "girl in a kimono, bob cut"}
    pf.merge(c, {"appearance": "girl in a kimono, bob cut, holding a fan"})
    assert pf.effective(c)["appearance"] == "girl in a kimono, bob cut"  # the reader's field wins
    d = pf.voice_description(c)
    assert d.startswith("A child female voice for a character who appears as girl in a kimono, bob cut.")
    assert "Way of speaking: teasing; calls people dearie." in d and "Centuries old; sly." in d
    assert pf.voice_description({"profile": {"apparent_age": "adult", "gender": "male"}}) == "An adult male voice."
    long = "; ".join(f"trait number {i}" for i in range(40))
    assert len(pf.compact_text(long)) <= pf.NOTES_MAX


def test_old_dossiers_become_profiles():
    from androidreader_server import profile as pf

    s = {"characters": {
        "f": {"name": "Futarishizuka", "temperament": "calm, perceptive", "accent": "neutral",
              "dossier_user": {"gender": "female", "apparent_age": "child", "form": "human, looks like a young girl in a kimono",
                               "true_nature": "centuries old", "speech_style": "old-fashioned, teasing, calls people dearie"}},
        "t": {"name": "TV Announcer", "appearance": "A polished broadcast figure.", "accent": "american",
              "temperament": "professional"}}}
    assert pf.migrate(s) == 2
    f, t = s["characters"]["f"], s["characters"]["t"]
    assert f["profile_user"] == {"gender": "female", "apparent_age": "child", "speech_style": "old-fashioned, teasing, calls people dearie",
                                 "appearance": "human, looks like a young girl in a kimono", "notes": "centuries old"}
    assert f["profile"] == {"notes": "calm, perceptive"}
    assert t["profile"] == {"appearance": "A polished broadcast figure.", "notes": "professional; american accent"}
    assert s["profile_v"] == 2


async def test_the_labelling_pass_fills_profiles(aiohttp_client, tmp_path, monkeypatch):
    seen = []

    async def fake(self, session, system, user, schema, max_tokens=2048, **kw):
        if "quotes" in schema["properties"]:
            seen.append((system, schema))
            ids = schema["properties"]["quotes"]["items"]["properties"]["id"]["enum"]
            return {"narrator": "third person", "speakers": [{"name": "Tom", "aliases": [], "gender": "male", "age": "child"}],
                    "quotes": [{"id": i, "starts": "", "speaker": "Tom", "delivery": "neutral"} for i in ids],
                    "profile_updates": [{"character": "Tom", "appearance": "a small boy in a straw hat",
                                         "speech_style": "stammers", "notes": "shy"}]}
        return {"characters": [{"name": "Tom", "appearance": "a boy", "gender": "male", "age": "child", "accent": "irish",
                                "temperament": "timid", "voice": "A shy boy's voice."}]}

    monkeypatch.setattr(dz.LocalLlm, "complete", fake)
    c = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert "profile_updates" in seen[0][1]["properties"] and "Character profiles" in seen[0][0]
    tom = next(ch for ch in s["characters"] if ch["id"] == "tom")
    # noted while reading; the roster step doesn't overwrite it (its appearance only fills gaps)
    assert tom["profile"] == {"appearance": "a small boy in a straw hat", "speech_style": "stammers", "notes": "shy",
                              "gender": "male", "apparent_age": "child"}


# ---------------------------------------------------------------- adding books from the website


async def test_an_uploaded_epub_is_parsed_analysed_and_cast(drama_client, tmp_path):
    import aiohttp

    from test_epub import make_epub

    path = make_epub(tmp_path / "tea.epub", {
        "META-INF/container.xml": '<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>',
        "content.opf": """<package><metadata><dc:title>Tea Party</dc:title><dc:creator>L. C.</dc:creator></metadata><manifest>
            <item id="c" href="c.xhtml" media-type="application/xhtml+xml"/>
            <item id="pic" href="p.jpg" media-type="image/jpeg" properties="cover-image"/></manifest>
            <spine><itemref idref="c"/></spine></package>""",
        "c.xhtml": "<html><body><p><img src='p.jpg'/></p>" + "".join(f"<p>{p}</p>" for p in PARAS) + "</body></html>",
        "p.jpg": "\xff\xd8\xff fake jpeg",
    })
    c = drama_client
    form = aiohttp.FormData()
    form.add_field("epub", path.read_bytes(), filename="tea.epub", content_type="application/epub+zip")
    r = await c.post("/api/library/import", data=form, headers=AUTH)
    assert r.status == 200, await r.text()
    book = await r.json()
    from androidreader_server import epub

    assert book["id"] == epub.book_id(path) and book["title"] == "Tea Party"
    s = await wait_done(c, book["id"])
    for _ in range(100):  # analysis, then the cast, without being asked
        s = await (await c.get(f"/api/books/{book['id']}/script", headers=AUTH)).json()
        if s["status"] == "done" and "voices created" in s["detail"]:
            break
        await asyncio.sleep(0.05)
    assert "voices created" in s["detail"], s["detail"]
    assert all(ch.get("voice") and ch.get("role") for ch in s["characters"] if ch["lines"])
    stored = await (await c.get(f"/api/books/{book['id']}", headers=AUTH)).json()
    assert stored["chapters"][0]["quotes"][0][0].startswith("c0p")
    assert (await c.get(f"/api/books/{book['id']}/cover", headers=AUTH)).status == 200
    pics = await (await c.get(f"/api/books/{book['id']}/pictures", headers=AUTH)).json()
    assert pics == {"have": ["p.jpg"], "need": []}
    # adding it again (here by path, allowed from this PC) keeps the analysis
    r = await c.post("/api/library/import", json={"path": str(path), "series": "Tea Books"}, headers=AUTH)
    again = await r.json()
    assert again["status"] == "done" and (again["series"], again["series_auto"]) == ("Tea Books", False)
    r = await c.post("/api/library/import", json={"path": str(tmp_path / "missing.epub")}, headers=AUTH)
    assert r.status == 400
    assert (await c.post("/api/library/import", data=aiohttp.FormData({"x": "1"}), headers=AUTH)).status == 400


def test_llm_restarts_after_crash_mid_request(monkeypatch, tmp_path):
    """A local llama-server that dies during a request is restarted and the request retried."""
    import aiohttp

    llm = dz.LocalLlm(make_cfg(tmp_path))
    starts, stops = [], []

    async def ensure(self, session, model=None, context=None):
        starts.append(1)
        self.proc = object()

    async def stop(self):
        stops.append(1)
        self.proc = None

    monkeypatch.setattr(dz.LocalLlm, "ensure", ensure)
    monkeypatch.setattr(dz.LocalLlm, "stop", stop)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(dz.asyncio, "sleep", lambda s: real_sleep(0))

    class Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self, content_type=None):
            return {"choices": [{"message": {"content": '{"ok": 1}'}}]}

    class Session:
        calls = 0

        def post(self, *a, **k):
            Session.calls += 1
            if Session.calls == 1:
                raise aiohttp.ClientOSError(64, "The specified network name is no longer available")
            return Resp()

    out = asyncio.run(llm.complete(Session(), "s", "u", {}))
    assert out["ok"] == 1 and Session.calls == 2 and len(starts) == 2 and len(stops) == 1

def test_roster_batches_fit_the_context():
    """A big cast with long evidence is split so every roster request fits the LLM context."""
    blocks = [f"## C{i}\n" + "x" * 3800 for i in range(12)]  # about 13k tokens in one request
    batches = dz._roster_batches(blocks, 4000, 12288)
    assert len(batches) > 1 and sum(batches, []) == blocks
    for b in batches:
        prompt = (4000 + sum(len(x) + 2 for x in b)) // 3
        assert prompt + 1024 + 300 * len(b) <= 12288
    assert dz._roster_batches(blocks[:3], 4000, 12288) == [blocks[:3]]  # a small cast stays one request

def test_capitalised_descriptions_get_title_case():
    norm = lambda n: dz.normalize_character_name(n)[0]
    assert norm("Bookstore employee") == "Bookstore Employee"
    assert norm("Urano's mother") == "Urano's Mother"
    assert norm("Ludwig van Beethoven") == "Ludwig van Beethoven"
    assert norm("Queen of Hearts") == "Queen of Hearts"
    assert norm("Pat (a footman)") == "Pat (a footman)"
    assert norm("Type Twelve") == "Type Twelve"


def test_canonical_names_follow_the_text():
    paras = (["Myne smiled."] * 12 + ["Urano read."] * 3 + ["Mom, said Effa."] * 4 + ["Dad, Gunther and Otto."] * 4
             + ["Futarishizuka bowed."] * 5 + ["Ms. Futarishizuka bowed."] * 3)
    book = {"chapters": [{"title": "1", "paragraphs": paras}]}
    s = {"characters": {}}
    ids = {}
    for name, aliases in (("Urano", ["Myne"]), ("Urano's Mother", ["Mom", "Effa"]), ("Dad", ["Captain", "Gunther"]),
                          ("Ms. Futarishizuka", ["Futarishizuka"]), ("Otto", [])):
        ids[name] = dz.ensure_character(s, name, aliases=aliases)
    typed = dz.ensure_character(s, "Mother dear", aliases=["Effa2"], exact=True)
    s["characters"][typed]["name_user"] = True
    renamed = dict(dz.canonical_names(book, s))
    assert renamed == {"Urano": "Myne", "Urano's Mother": "Effa", "Dad": "Gunther"}
    c = s["characters"][ids["Urano"]]
    assert c["name"] == "Myne" and "Urano" in c["aliases"]
    assert dz.find_character(s, "Urano") == ids["Urano"]  # old names still match
    assert s["characters"][ids["Ms. Futarishizuka"]]["name"] == "Ms. Futarishizuka"  # under 3x: kept
    assert s["characters"][typed]["name"] == "Mother dear"

def test_profile_placeholders_are_not_descriptions():
    from androidreader_server import profile

    for t in ("none described", "Not mentioned", "N/A", "unknown", "nothing specified", "No details given"):
        assert profile.blank(t), t
    for t in ("pale skin, wears glasses", "none too friendly", "Nothing escapes her"):
        assert not profile.blank(t), t
    c = {"profile": {"gender": "female", "appearance": "none described", "notes": "kind; not mentioned"}}
    assert profile.clean(c) and c["profile"] == {"gender": "female", "notes": "kind"}
    assert not profile.merge(c, {"appearance": "Not described", "notes": "N/A"})
    assert "appears as" not in profile.voice_description({"profile": {"appearance": "none described"}})

def test_volume_key_orders_volumes():
    from androidreader_server.series import volume_key

    assert volume_key("Sasaki and Peeps, Vol. 10") == [10]
    assert volume_key("Ascendance of a Bookworm: Part 2 Volume 3") == [2, 3]
    assert volume_key("Mistborn Book 2") == [2]
    assert volume_key("Sasaki and Peeps v10 [Yen Press]") == [10]
    assert volume_key("The Wandering Inn 4") == [4]
    assert volume_key("Alice's Adventures in Wonderland") == []


def test_applying_a_series_keeps_voices_the_reader_chose():
    from androidreader_server import series as sr
    from androidreader_server.dramatize import DRAMA_STORE_KEY

    data = {"name": "Saga", "books": {"v1": "Saga 1"}, "characters": {
        "alice": {"name": "Alice", "aliases": [], "voice": "series-alice", "books": ["v1"]},
        "tom": {"name": "Tom", "aliases": [], "voice": "series-tom", "books": ["v1"]},
        "ann": {"name": "Ann", "aliases": [], "voice": "pool-female-adult", "books": ["v1"]}}}

    class Store:
        def get(self, name):
            return data

    class Drama:
        def summary(self, book_id):
            return {"title": "Saga 2"}

    app = {sr.SERIES_KEY: Store(), DRAMA_STORE_KEY: Drama()}
    script = {"characters": {
        "alice": {"name": "Alice", "voice": "ch-v2v2v2v2-alice"},
        "tom": {"name": "Tom", "voice": "my-pick", "voice_user": True}}, "lines": {}}
    assert sr.seed(app, "v2v2v2v2", script) == 3
    assert script["characters"]["alice"]["voice"] == "ch-v2v2v2v2-alice"  # a plain seed fills gaps only
    sr.seed(app, "v2v2v2v2", script, replace_generated=True)
    assert script["characters"]["alice"]["voice"] == "series-alice"
    assert script["characters"]["tom"]["voice"] == "my-pick"
    assert script["characters"]["ann"]["voice"] == "pool-female-adult"
    assert sr.unseed(script, "Saga") == 3  # none of them has a line here
    assert script["characters"] == {}


async def wait_cast(c, book):
    for _ in range(200):
        s = await (await c.get(f"/api/books/{book}/script", headers=AUTH)).json()
        if s["status"] == "done" and "voices" in s["detail"]:
            return s
        await asyncio.sleep(0.05)
    raise AssertionError(s["detail"])


async def test_series_can_be_changed_applied_renamed_and_merged(drama_client):
    from androidreader_server.dramatize import DRAMA_STORE_KEY

    c = drama_client
    app = c.server.app
    await c.post("/api/books/vol1", json={**book_payload(), "title": "Tea Saga, Vol. 1", "analyze": True}, headers=AUTH)
    await wait_done(c, "vol1")
    await c.patch("/api/books/vol1/characters/alice", json={"voice": "nar"}, headers=AUTH)

    # volume 2 is detected as its own series ("Tea Saga Gaiden"), analysed and cast with voices of its own
    await c.post("/api/books/gaiden02", json={**book_payload(), "title": "Tea Saga Gaiden 2", "analyze": True}, headers=AUTH)
    await wait_done(c, "gaiden02")
    await c.post("/api/books/gaiden02/cast", json={}, headers=AUTH)
    s = await wait_cast(c, "gaiden02")
    alice = next(ch for ch in s["characters"] if ch["id"] == "alice")
    assert alice["voice"].startswith("ch-gaiden02-")
    info = await (await c.get("/api/books/gaiden02/series", headers=AUTH)).json()
    assert (info["series"], info["auto"], info["can_apply"], info["volume"]) == ("Tea Saga Gaiden", True, False, [2])

    # moving it to "Tea Saga": it leaves the old series; the series' cast is offered, not yet written over
    r = await (await c.post("/api/books/gaiden02/series", json={"series": "Tea Saga"}, headers=AUTH)).json()
    assert (r["series"], r["auto"], r["can_apply"], r["applied"]) == ("Tea Saga", False, True, None)
    assert "gaiden02" not in r["data"]["books"] and r["data"]["characters"]["alice"]["voice"] == "nar"
    names = [g["name"] for g in (await (await c.get("/api/series", headers=AUTH)).json())["series"]]
    assert "Tea Saga Gaiden" not in names
    # applying it gives the book the series' voices without labelling it again
    lines_before = (await (await c.get("/api/books/gaiden02/script", headers=AUTH)).json())["lines"]
    r = await (await c.post("/api/books/gaiden02/series", json={"apply": True}, headers=AUTH)).json()
    assert r["applied"] >= 1 and not r["can_apply"] and "gaiden02" in r["data"]["books"]
    s = await (await c.get("/api/books/gaiden02/script", headers=AUTH)).json()
    assert next(ch for ch in s["characters"] if ch["id"] == "alice")["voice"] == "nar"
    assert s["lines"] == lines_before and s["status"] == "done"

    # the series page: volumes in order
    series = (await (await c.get("/api/series", headers=AUTH)).json())["series"]
    saga = next(g for g in series if g["name"] == "Tea Saga")
    assert [v["id"] for v in saga["volumes"]] == ["vol1", "gaiden02"] and saga["cast"] >= 1

    # rename: every book follows
    r = await c.post("/api/series/rename", json={"from": "Tea Saga", "to": "Tea Chronicles"}, headers=AUTH)
    assert sorted((await r.json())["books"]) == ["gaiden02", "vol1"]
    store = app[DRAMA_STORE_KEY]
    assert store.summary("vol1")["series"] == store.summary("gaiden02")["series"] == "Tea Chronicles"
    # merge: a third book's series folds into it; the same characters become one, the existing voice wins
    await c.post("/api/books/side", json={**book_payload(), "title": "Tea Side Stories", "analyze": True}, headers=AUTH)
    await wait_done(c, "side")
    await c.post("/api/books/side/series", json={}, headers=AUTH)
    r = await (await c.post("/api/series/rename", json={"from": "Tea Side Stories", "to": "Tea Chronicles"},
                            headers=AUTH)).json()
    assert r["merged"] and r["books"] == ["side"]
    assert r["data"]["characters"]["alice"]["voice"] == "nar" and "side" in r["data"]["characters"]["alice"]["books"]
    names = [g["name"] for g in (await (await c.get("/api/series", headers=AUTH)).json())["series"]]
    assert "Tea Side Stories" not in names and store.summary("side")["series"] == "Tea Chronicles"
    assert (await c.post("/api/series/rename", json={"from": "", "to": "x"}, headers=AUTH)).status == 400

async def test_setting_a_series_keeps_the_title_of_a_book_without_meta(drama_client):
    from androidreader_server.dramatize import DRAMA_STORE_KEY

    c = drama_client
    store = c.server.app[DRAMA_STORE_KEY]
    store.save_book("old", {**book_payload(), "title": "Old Tale, Vol. 3"})
    (store.dir("old") / "meta.json").unlink()  # stored before meta.json existed
    r = await (await c.post("/api/books/old/series", json={"series": "Old Tales"}, headers=AUTH)).json()
    assert r["series"] == "Old Tales" and r["title"] == "Old Tale, Vol. 3"
    assert store.summary("old")["title"] == "Old Tale, Vol. 3"