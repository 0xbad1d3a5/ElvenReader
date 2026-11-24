import asyncio
import json

import pytest

from elvenreader_server import dramatize as dz
from elvenreader_server.app import make_app
from test_server import AUTH, SPECS, make_cfg
from elvenreader_server.audio import wav_bytes
from elvenreader_server.voices import VoiceStore

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


def test_a_spoken_form_is_kept_only_for_a_quotation_in_one_piece():
    script = {"characters": {}, "lines": {}}
    wh = "\u201cWh\u2026 whaaaaaaat?!\u201d"
    dz.apply_chunk_result(script, {"speakers": [], "quotes": [
        {"id": "1", "starts": "Wh", "speaker": "Ronie", "delivery": "shouting, drawn out", "_text": wh,
         "respell": [{"written": "whaaaaaaat", "said": "what"}]},
        {"id": "2", "starts": "Yes", "speaker": "Tom", "delivery": "calm", "_text": "\u201cYesss,\u201d",
         "respell": [{"written": "Yesss", "said": "yes"}]},
        {"id": "3", "starts": "No", "speaker": "Tom", "delivery": "calm", "_text": "\u201cNo.\u201d", "respell": []},
        {"id": "4", "starts": "Krrr", "speaker": "Tom", "delivery": "growling", "_text": "\u201cKrrrr!\u201d",
         "respell": [{"written": "Grrr", "said": "grr"}]},  # not in the quotation: ignored
    ]}, {"1": ["c0p0q0"], "2": ["c0p1q0", "c0p1q9"], "3": ["c0p2q0"], "4": ["c0p3q0"]})
    assert script["lines"]["c0p0q0"]["say"] == "\u201cWh\u2026 what?!\u201d"
    assert "say" not in script["lines"]["c0p1q0"] and "say" not in script["lines"]["c0p1q9"]
    assert "say" not in script["lines"]["c0p2q0"] and "say" not in script["lines"]["c0p3q0"]
    assert "respell" in dz.chunk_schema(["1"])["properties"]["quotes"]["items"]["required"]


def test_respelling_swaps_only_the_named_words():
    text = "\u201cY-yeah\u2026 Nooooo, please, I mean\u2026 how do you build trust that way\u2026?\u201d"
    assert dz.respelled(text, [{"written": "nooooo", "said": "no"}]) == \
        "\u201cY-yeah\u2026 No, please, I mean\u2026 how do you build trust that way\u2026?\u201d"
    assert dz.respelled(text, [{"written": "Y-yeah", "said": "Y-yeah"}]) == ""  # unchanged
    assert dz.respelled(text, [{"written": "Y-yeah", "said": "Yeah"}]) == ""  # stutters stay
    assert dz.respelled("\u201cSo-o-o tired\u201d", [{"written": "So-o-o", "said": "so"}]) == "\u201cSo tired\u201d"
    wh = "\u201cWh\u2026whaaaaaaat?!\u201d"
    assert dz.respelled(wh, [{"written": "Wh\u2026whaaaaaaat?!", "said": "What?"}]) == ""  # one word only
    # A word cut off mid-way stays cut off.
    assert dz.respelled("\u201cJust drive safely, plea\u2014,\u201d", [{"written": "plea", "said": "please"}]) == ""
    assert dz.respelled(text, [{"written": "oo", "said": "o"}]) == ""  # only whole words
    assert dz.respelled(text, []) == "" and dz.respelled(text, None) == ""


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
            {"id": "1", "speaker": "Mrs. Bennet", "delivery": ["very happy", "loud", "shout", "made-up"]},
            {"id": "2", "speaker": "mrs bennet", "delivery": []},
            {"id": "3", "speaker": "Mr. Bennet", "delivery": "dry, sarcastic"},
            {"id": "4", "speaker": "NARRATOR", "delivery": "neutral"},
        ],
    }, local)
    assert script["lines"]["a"] == {"speaker": "mrs-bennet", "delivery": "very happy, loud"}  # one volume
    assert script["lines"]["b"] == {"speaker": "mrs-bennet", "delivery": None}
    assert script["lines"]["c"] == {"speaker": "mr-bennet", "delivery": "sarcastic"}  # a cached free-text answer
    assert script["lines"]["d"]["speaker"] is None
    assert script["characters"]["mrs-bennet"]["gender"] == "female"
    assert "Mrs. Bennet (female, middle-aged)" in dz.roster_text(script)


@pytest.fixture
async def drama_client(aiohttp_client, tmp_path, monkeypatch):
    VoiceStore(tmp_path / "voices").add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")

    async def fake_complete(self, session, system, user, schema, max_tokens=2048, **kw):
        item = (schema["properties"].get("characters") or {}).get("items", {}).get("properties", {})
        if "voice" in item and "id" in item:  # which generic voice fits
            return {"characters": [{"id": i, "voice": {"tom": "Male Child"}.get(i, "Female Adult")} for i in item["id"]["enum"]]}
        if "role" in item:  # the tier question
            roles = {"Alice": ("main", "the heroine"), "Tom": ("extra", "one scene")}
            return {"characters": [{"name": n, "role": roles.get(n, ("extra", ""))[0], "reason": roles.get(n, ("", "?"))[1]}
                                   for n in item["name"]["enum"]]}
        if "quotes" in schema["properties"]:
            ids = schema["properties"]["quotes"]["items"]["properties"]["id"]["enum"]
            names = {"1": "Alice", "2": "Alice", "3": "Tom", "4": "NARRATOR"}
            deliveries = {"1": ["angry", "shout"], "3": ["whisper"]}
            return {"quotes": [{"id": i, "starts": "", "speaker": names[i], "delivery": deliveries.get(i, [])} for i in ids],
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
    assert s["lines"]["c0p1q0"] == {"speaker": "alice", "delivery": "angry, shout", "corrected": False}
    assert s["lines"]["c0p3q14"]["speaker"] is None
    alice = next(ch for ch in s["characters"] if ch["id"] == "alice")
    assert alice["lines"] == 2 and alice["description"].startswith("A young British woman")

    # A user correction wins over the model and survives.
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "alice", "delivery": "scared"}, headers=AUTH)
    line = (await r.json())["lines"]["c0p2q11"]
    assert line == {"speaker": "alice", "delivery": "scared", "corrected": True,
                    "analysis": {"speaker": "tom", "delivery": "whisper"}}  # what the model had said

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


async def test_books_can_be_analysed_on_an_inference_server(aiohttp_client, aiohttp_server, tmp_path, monkeypatch):
    """Settings' "remote" analysis: the same requests go to an OpenAI-compatible server (llama-swap). Like vLLM with
    speculative decoding there, this one fails JSON-schema requests, so the schema moves into the prompt; and like
    llama-swap at its request limit, it sometimes answers 429, and the request waits for a free slot."""
    from aiohttp import web

    monkeypatch.setattr(dz, "REMOTE_BUSY_DELAY_S", 0.01)
    seen, busy = [], [2]

    async def chat(request):
        body = await request.json()
        seen.append(body)
        if busy[0]:
            busy[0] -= 1
            return web.json_response({"error": {"message": "Too many requests", "code": "concurrency_limit"}}, status=429)
        if "response_format" in body:
            return web.json_response({"error": {"message": "Internal server error"}}, status=500)
        schema = json.loads(body["messages"][0]["content"].rsplit("\n", 1)[1])
        props = schema["properties"]
        if "quotes" in props:
            ids = props["quotes"]["items"]["properties"]["id"]["enum"]
            names = {"1": "Alice", "2": "Alice", "3": "Tom", "4": "NARRATOR"}
            answer = {"quotes": [{"id": i, "starts": "", "speaker": names[i], "delivery": []} for i in ids],
                      "speakers": [{"name": "Alice", "gender": "female", "age": "young adult", "aliases": []},
                                   {"name": "Tom", "gender": "male", "age": "child", "aliases": []}]}
        else:
            answer = {k: [] for k in props}
        return web.json_response({"choices": [{"message": {"content": "```json\n" + json.dumps(answer) + "\n```"}}],
                                  "usage": {"total_tokens": 50}})

    remote = web.Application()
    remote.router.add_post("/v1/chat/completions", chat)
    server = await aiohttp_server(remote)
    cfg = make_cfg(tmp_path, llm_backend="remote", llm_remote_url=str(server.make_url("/v1")), llm_remote_model="qwen-x")
    c = await aiohttp_client(make_app(cfg, specs=SPECS))
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert s["status"] == "done", s
    assert s["lines"]["c0p1q0"]["speaker"] == "alice" and s["lines"]["c0p2q11"]["speaker"] == "tom"
    assert seen[2].get("response_format") and not any("response_format" in b for b in seen[3:])
    label = next(b for b in seen[1:] if "quotes" in b["messages"][0]["content"])
    assert label["model"] == "qwen-x" and label["chat_template_kwargs"] == {"enable_thinking": False}
    assert label["logprobs"] and label["top_logprobs"]
    analyzer = c.server.app[dz.ANALYZER_KEY]
    assert analyzer.llm.proc is None  # this PC's llama-server never started
    assert c.server.app[dz.DRAMA_STORE_KEY].script("book1")["model"] == "qwen-x"


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
        assert "ElvenReader books" in page and '"testtoken"' in page and "__TOKEN__" not in page


async def test_viewer_is_served_on_the_lan_without_the_token(drama_client):
    from unittest import mock

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from elvenreader_server.app import auth_middleware, viewer_page

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
    # the PC's start page is the Users page (each user's link), with the admin's token
    r = await drama_client.get("/")
    assert r.url.path == "/books/users" and 'const EMBEDDED_TOKEN = "testtoken";' in await r.text()


async def test_reanalyze_keeps_corrections(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    await c.patch("/api/books/book1/lines/c0p1q0", json={"speaker": "tom", "delivery": "calm"}, headers=AUTH)
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True, "reanalyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert s["status"] == "done"
    assert s["lines"]["c0p1q0"] == {"speaker": "tom", "delivery": "calm", "corrected": True,
                                    "analysis": {"speaker": "alice", "delivery": "angry, shout"}}
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
    # An alias unrelated to the name resolves its own passage's labels, but waits for a second passage to stick
    # ("the girl" is a description and never counts).
    assert script["characters"]["miss-hoshizaki"]["aliases"] == []
    assert list(script["characters"]["miss-hoshizaki"]["alias_votes"]) == ["Makeup"]
    assert script["narrator"] == "Sasaki"
    dz.apply_chunk_result(script, {"narrator": "third person", "quotes": [], "speakers": [
        {"name": "Miss Hoshizaki", "aliases": ["Makeup"], "gender": "female", "age": "teen"}]}, {}, vote="chunk-2")
    assert script["characters"]["miss-hoshizaki"]["aliases"] == ["Makeup"]
    # An alias never takes over another character's name.
    dz.apply_chunk_result(script, {"narrator": "third person", "quotes": [],
                                   "speakers": [{"name": "Nick", "aliases": ["Miss Hoshizaki"], "gender": "male", "age": "adult"}]}, {})
    assert "Miss Hoshizaki" not in script["characters"]["nick"]["aliases"]


def test_a_shared_title_never_folds_a_new_speaker_into_someone_else():
    """Asuna once picked up "Commander"; then "Fanatio" (also called Commander) joined her, and every one of
    Fanatio's lines went to Asuna."""
    script = {"characters": {}, "lines": {}, "narrator": "third person"}
    dz.ensure_character(script, "Asuna", "female", "young adult", ["Commander"])  # series memory: taken at once
    dz.apply_chunk_result(script, {"narrator": "third person", "speakers": [
        {"name": "Fanatio", "aliases": ["Commander", "Fanatio Synthesis Two"], "gender": "female", "age": "adult"}],
        "quotes": [{"id": "1", "starts": "", "speaker": "Fanatio", "delivery": "calm"}]}, {"1": ["q1"]}, vote="c1")
    assert script["lines"]["q1"]["speaker"] == "fanatio"
    assert "Fanatio" not in script["characters"]["asuna"]["aliases"]
    assert "Fanatio Synthesis Two" in script["characters"]["fanatio"]["aliases"]  # shares her name: taken at once
    # A real name given with the character's own name still joins them.
    dz.apply_chunk_result(script, {"narrator": "third person", "quotes": [], "speakers": [
        {"name": "Asuna Yuuki", "aliases": ["Asuna"], "gender": "female", "age": "young adult"}]}, {}, vote="c2")
    assert "Asuna Yuuki" in script["characters"]["asuna"]["aliases"] and len(script["characters"]) == 2
    # The cast list shown to the model never offers an alias another character answers to.
    script["characters"]["fanatio"]["aliases"].append("Commander")
    assert "Commander" not in dz.roster_text(script).split("Fanatio")[1]


def test_a_cover_name_linked_once_still_finds_the_character_later():
    """Sasaki undercover: one passage says "Sasaki, also called Kenta Nakano"; later ones only say "Kenta
    Nakano". He stays one character, and the cover name becomes his alias."""
    script = {"characters": {}, "lines": {}, "narrator": "third person"}
    dz.ensure_character(script, "Sasaki", "male", "middle-aged")
    male = {"gender": "male", "age": "middle-aged"}
    passages = [([{"name": "Sasaki", "aliases": ["Kenta Nakano"], **male}], "Kenta Nakano"),
                ([], "Kenta Nakano"),  # the model just labels the cover name
                ([{"name": "Kenta Nakano", "aliases": [], **male}], "Kenta Nakano")]
    for i, (speakers, label) in enumerate(passages):
        dz.apply_chunk_result(script, {"narrator": "third person", "speakers": speakers, "quotes": [
            {"id": "1", "starts": "", "speaker": label, "delivery": ""}]}, {"1": [f"q{i}"]}, vote=f"p{i}")
    assert [script["lines"][f"q{i}"]["speaker"] for i in range(3)] == ["sasaki"] * 3
    assert list(script["characters"]) == ["sasaki"] and script["characters"]["sasaki"]["aliases"] == ["Kenta Nakano"]


def test_pronouns_are_never_names():
    assert not dz.is_proper_alias("I") and not dz.is_proper_alias("She")
    book = {"chapters": [{"title": "x", "paragraphs": ["I said I would. I did. I know Kirito."], "quotes": []}]}
    script = {"characters": {"kirito": {"name": "Kirito", "aliases": ["I"]}}, "lines": {}}
    dz.canonical_names(book, script, ratio=1)
    assert script["characters"]["kirito"]["name"] == "Kirito"


def test_speaker_odds_come_from_the_token_probabilities():
    import math

    def tok(t, p, alts=()):
        return {"token": t, "logprob": math.log(p), "top_logprobs": [{"token": t, "logprob": math.log(p)},
                                                                     *({"token": a, "logprob": math.log(q)} for a, q in alts)]}
    tokens = [tok('{"quotes":[{"id":"1","starts":"That","speaker":"', 1.0), tok("As", 0.6, [("Fan", 0.3), ("NAR", 0.05)]),
              tok("una", 0.95), tok('","delivery":"calm"}]}', 1.0)]
    [(p, alts)] = dz.speaker_odds(tokens)
    assert abs(p - 0.57) < 0.01 and [(t, round(q, 3)) for t, q in alts] == [("Fan", 0.3), ("NAR", 0.05)]
    script = {"characters": {"asuna": {"name": "Asuna", "aliases": []}, "fanatio": {"name": "Fanatio", "aliases": []}},
              "lines": {}}
    odds = dz.line_odds(script, {"_p": p, "_alts": alts}, "asuna")
    assert odds == {"conf": 0.57, "alts": [["fanatio", 0.3], [None, 0.05]]}


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
    # The reader's corrections hold to the end of their section (the next POV heading or chapter), and survive
    # every recompute.
    script["narration_fixed"] = {"1:1": "tom"}
    assert dz.compute_narration(book, script) == [[0, 0, "sasaki"], [1, 1, "tom"], [2, 0, "sasaki"]]
    script["narration_fixed"] = {"1:0": None}  # chapter Two's opening to the book's narrator voice, up to the POV
    assert dz.compute_narration(book, script) == [[0, 0, "sasaki"], [1, 0, None], [1, 1, "neighbor"], [2, 0, "sasaki"]]
    script["narration_fixed"] = {"2:0": "neighbor"}  # the last section: to the end of the book
    assert dz.compute_narration(book, script) == [[0, 0, "sasaki"], [1, 1, "neighbor"]]
    # a third-person book can be given a narrator too
    assert dz.compute_narration(third_person, {"lines": {"a": {"speaker": "tom"}}, "narration_fixed": {"0:0": "tom"}}) \
        == [[0, 0, "tom"]]
    # a third-person scene after a scene break goes to the book's narrator voice; the next one to its "I"
    Q, E = "\u201c", "\u201d"
    third = " ".join(["Asuna walked to class and thought about the cabin, and she sat down."] * 12)
    ch = {"title": "Nine", "paragraphs": [third, f"{Q}Who are you?{E} Asuna asked.", f"{Q}Um,{E} I said.",
                                          "I yawned and rubbed my eyes."], "breaks": [2]}
    ch["quotes"] = [[f"z{pi}-{s}", pi, s, e] for pi, t in enumerate(ch["paragraphs"]) for s, e in dz.quote_regions(t)]
    book["chapters"].append(ch)
    lines.update({ch["quotes"][0][0]: {"speaker": "asuna"}, ch["quotes"][1][0]: {"speaker": "sasaki"}})
    script["narration_fixed"] = {}
    assert dz.compute_narration(book, script)[-2:] == [[3, 0, None], [3, 2, "sasaki"]]
    assert dz.section_end(book, 3, 0) == (3, 2)
    # a first-person scene after a break with no "I said" in it (a flashback) goes on with the narrator before it
    flash = " ".join(["I remembered the old days and walked home slowly with my sister."] * 12)
    ch2 = {"title": "Ten", "paragraphs": [f"{Q}Wait,{E} I said.", flash], "breaks": [1]}
    ch2["quotes"] = [["w0", 0, 0, 7]]
    book["chapters"].append(ch2)
    lines["w0"] = {"speaker": "tom"}
    assert dz.compute_narration(book, script)[-1:] == [[4, 0, "tom"]]
    # first person carried by "we" (one "I" among them) stays first person; "we" with no "I" (reported speech) doesn't
    we = " ".join(["We walked on and our packs grew heavy as the road climbed past us."] * 12) + " I sighed."
    ours = " ".join(["The elder said our city stood in the north before the flood took us."] * 12)
    ch2["paragraphs"][1], ch2["breaks"] = we, [1]
    assert dz.compute_narration(book, script)[-1:] == [[4, 0, "tom"]]
    ch2["paragraphs"].append(ours)
    ch2["breaks"] = [1, 2]
    assert dz.compute_narration(book, script)[-2:] == [[4, 0, "tom"], [4, 2, None]]
    book["chapters"].pop()
    del lines["w0"]
    # and the LLM is told: the passage and its context show the break on a line of its own
    text, local = dz.render_chunk(book, dz.Chunk("c3k0", 3, [0, 1, 2, 3], [tuple(q) for q in ch["quotes"]]))
    assert text.count(dz.SCENE_BREAK) == 1 and text.index(dz.SCENE_BREAK) < text.index("[2]")
    assert list(local) == ["1", "2"]  # the numbering is as before
    ch["silent"] = {"3": ["\u201c\u2026?\u201d"]}  # a wordless line before paragraph 3: shown, unnumbered
    text, local = dz.render_chunk(book, dz.Chunk("c3k0", 3, [0, 1, 2, 3], [tuple(q) for q in ch["quotes"]]))
    assert "\u201c\u2026?\u201d\n\nI yawned" in text and list(local) == ["1", "2"]


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
    from elvenreader_server import dossier as ds

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
    from elvenreader_server.series import series_name

    assert series_name("Sasaki and Peeps, Vol. 10") == "Sasaki and Peeps"
    assert series_name("Sasaki and Peeps v10 [Yen Press]") == "Sasaki and Peeps"
    assert series_name("The Wandering Inn: Volume 3") == "The Wandering Inn"
    assert series_name("Mistborn Book 2") == "Mistborn"
    assert series_name("Alice's Adventures in Wonderland") == "Alice's Adventures in Wonderland"


async def test_series_memory_carries_cast_and_voices_to_the_next_volume(drama_client):
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

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
    from elvenreader_server import series as sr

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
    from elvenreader_server import dossier as ds

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
    from elvenreader_server import roles

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
    from elvenreader_server import series as sr
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

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
    from elvenreader_server.series import _higher

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
    from elvenreader_server import roles

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


async def test_a_new_name_that_was_filed_under_someone_else_becomes_its_own_character(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    app = c.server.app
    store = app[dz.DRAMA_STORE_KEY]
    s = store.script("book1")
    s["characters"]["tom"]["aliases"] += ["Fanatio", "Fanatio Synthesis Two", "Tommy"]  # the analysis mixed them up
    store.save_script("book1", s)
    from elvenreader_server.series import SERIES_KEY, book_series, remember

    remember(app, "book1")
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "Fanatio", "delivery": ""}, headers=AUTH)
    s = await r.json()
    chars = {ch["id"]: ch for ch in s["characters"]}
    new = chars[s["lines"]["c0p2q11"]["speaker"]]
    assert new["id"] != "tom" and new["name"] == "Fanatio"
    assert "Tommy" in chars["tom"]["aliases"] and not any("Fanatio" in a for a in chars["tom"]["aliases"])
    series = app[SERIES_KEY].get(book_series(app, "book1"))
    assert not any("Fanatio" in a for e in series["characters"].values() for a in e.get("aliases", []))
    # A character's own name still picks that character.
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "Tom", "delivery": ""}, headers=AUTH)
    assert (await r.json())["lines"]["c0p2q11"]["speaker"] == "tom"


async def test_manual_edits_and_flags_are_kept_as_reports_of_incorrect_analysis(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "alice", "delivery": "scared"}, headers=AUTH)
    await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "alice", "delivery": "scared"}, headers=AUTH)  # no change
    await c.patch("/api/books/book1/lines/c0p1q0", json={"speaker": "The Butler"}, headers=AUTH)
    await c.patch("/api/books/book1/characters/tom", json={"name": "Tom Thumb", "role": "main"}, headers=AUTH)
    r = await c.post("/api/books/book1/reports", json={"line": "c0p3q14", "note": "this is Alice thinking"}, headers=AUTH)
    assert r.status == 201
    assert (await c.post("/api/books/book1/reports", json={}, headers=AUTH)).status == 400
    reports = (await (await c.get("/api/books/book1/reports", headers=AUTH)).json())["reports"]
    kinds = [r["type"] for r in reports]
    assert kinds == ["speaker", "delivery", "new_character", "name", "role", "flag"]
    first = reports[0]
    assert first["line"] == "c0p2q11" and first["before"]["speaker"] == "tom" and first["after"]["name"] == "Alice"
    assert first["text"].startswith("\u201c")
    assert reports[-1]["note"] == "this is Alice thinking" and reports[-1]["before"]["speaker"] is None
    everything = (await (await c.get("/api/reports?type=flag", headers=AUTH)).json())["reports"]
    assert [r["type"] for r in everything] == ["flag"]
    # Voice-editor changes are casting fixes: reported in every book whose characters use the voice.
    await c.patch("/api/books/book1/characters/alice", json={"voice": "nar"}, headers=AUTH)
    r = await c.patch("/api/voices/nar", json={"description": "older, slower"}, headers=AUTH)
    assert r.status == 200
    voice_reports = [x for x in (await (await c.get("/api/books/book1/reports", headers=AUTH)).json())["reports"]
                     if x["type"] == "voice" and x.get("edit") == "details"]
    assert [(x["character"], x["after"]["description"]) for x in voice_reports] == [("alice", "older, slower")]


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


def test_unused_character_voices_are_deleted(tmp_path):
    from elvenreader_server import reports
    from elvenreader_server.app import VOICES_KEY

    voices, store = VoiceStore(tmp_path / "voices"), dz.ScriptStore(tmp_path / "books")
    wav = b"RIFF" + b"\0" * 64
    for vid, source in (("used", "character"), ("orphan", "character"), ("past", "character"), ("mine", "upload"),
                        ("pool-m-adult", "character")):
        voices.add(vid, wav, "hi", source=source, voice_id=vid)
    voices.update("orphan", description="edited")  # its earlier versions go with it
    (tmp_path / "books" / "b1").mkdir(parents=True)
    (tmp_path / "books" / "b1" / "book.json").write_text("{}")
    store.save_script("b1", {"characters": {"tom": {"name": "Tom", "voice": "used"}}})
    app = {dz.DRAMA_STORE_KEY: store, VOICES_KEY: voices}
    reports.record(app, "b1", "voice", character="tom", before={"voice": "past"}, after={"voice": "used"})
    assert dz.delete_unused_voices(app) == ["orphan"]
    assert voices.get("orphan") is None and not (tmp_path / "voices" / ".history" / "orphan").exists()
    assert all(voices.get(v) for v in ("used", "past", "mine", "pool-m-adult"))
    assert dz.delete_unused_voices(app) == []


def test_name_voices_follows_renamed_characters(tmp_path):
    from elvenreader_server.app import VOICES_KEY
    from elvenreader_server.voices import VoiceStore

    voices, store = VoiceStore(tmp_path / "voices"), dz.ScriptStore(tmp_path / "books")
    wav = b"RIFF" + b"\0" * 64
    for vid, name, source in (("ch-dad", "Dad", "character"), ("ch-ortwin", "Lord Ortwin", "character"),
                              ("ch-mine", "Grumpy Guard", "character"), ("studio", "Mother", "voicestudio")):
        voices.add(name, wav, "hi", source=source, voice_id=vid)

    def book(book_id, **characters):
        (tmp_path / "books" / book_id).mkdir(parents=True)
        (tmp_path / "books" / book_id / "book.json").write_text("{}")
        store.save_script(book_id, {"characters": characters})

    book("vol1", dad={"name": "Gunther", "aliases": ["Dad", "Captain"], "voice": "ch-dad"},
         ortwin={"name": "Ortwin", "aliases": ["Lord Ortwin"], "voice": "ch-ortwin"},
         guard={"name": "Guard", "aliases": ["a guard"], "voice": "ch-mine"},
         mother={"name": "Effa", "aliases": ["Mother"], "voice": "studio"})
    book("vol2", gunther={"name": "Gunther", "aliases": ["Dad"], "voice": "ch-dad"},
         ortwin={"name": "Lord Ortwin", "aliases": ["Ortwin"], "voice": "ch-ortwin"})
    app = {dz.DRAMA_STORE_KEY: store, VOICES_KEY: voices}
    assert dz.name_voices(app) == [("Dad", "Gunther")]
    # still "Lord Ortwin" in vol2: the voice keeps that name rather than flip between books;
    # a name the reader gave the voice, and their own voices, are kept
    assert [voices.get(v).name for v in ("ch-dad", "ch-ortwin", "ch-mine", "studio")] == \
        ["Gunther", "Lord Ortwin", "Grumpy Guard", "Mother"]
    assert dz.name_voices(app) == []


# ---------------------------------------------------------------- the voice pool


def test_pool_options_follow_gender_and_age():
    teen = {"name": "Kid", "gender": "female", "age": "teen"}
    assert dz.pool_options(teen) == ["pool-f-teen", "pool-f-teen-strong", "pool-f-teen-soft"]
    assert dz.pool_voice(teen) == "pool-f-teen"
    assert dz.pool_voice({**teen, "pool_pick": "pool-f-teen-soft"}) == "pool-f-teen-soft"
    # a pick that no longer suits them (the profile now says elderly) gives way to the plain voice
    old = {**teen, "pool_pick": "pool-f-teen-soft", "profile_user": {"apparent_age": "elderly"}}
    assert dz.pool_voice(old) == "pool-f-elderly"
    assert dz.pool_options({"name": "Guard", "gender": "male", "age": "young adult"})[0] == "pool-m-adult-young"
    # unknown gender: both; unknown age: every age
    assert {"pool-f-child", "pool-m-child"} <= set(dz.pool_options({"name": "Voice", "gender": "unknown", "age": "child"}))
    assert len(dz.pool_options({"name": "Voice"})) == len(dz.POOL)
    assert dz.pool_voice({"name": "Voice"}) == "pool-m-adult"
    assert all(vid.startswith("pool-") for vid in dz.POOL)


async def test_pick_pool_voices_keeps_only_fitting_picks():
    script = {"characters": {"tom": {"name": "Tom", "gender": "male", "age": "child"},
                             "ann": {"name": "Ann", "gender": "female", "age": "adult",
                                     "profile": {"speech_style": "loud, bossy"}}}}

    class FakeLlm:
        async def complete(self, session, system, user, schema, max_tokens=0, **kw):
            assert "Female Adult - Strong" in user and "speaks: loud, bossy" in user
            return {"characters": [{"id": "tom", "voice": "Female Child"},  # wrong gender: dropped
                                   {"id": "ann", "voice": "Female Adult - Strong"}]}

    picks = await dz.pick_pool_voices(FakeLlm(), None, {"title": "B"}, script, ["tom", "ann"])
    assert picks == {"ann": "pool-f-adult-strong"}


async def test_generic_voices_are_picked_and_old_pool_voices_replaced(drama_client, monkeypatch):
    import aiohttp

    from elvenreader_server.app import VOICES_KEY

    c = drama_client
    app = c.server.app
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    assert app[dz.DRAMA_STORE_KEY].script("book1")["characters"]["tom"]["pool_pick"] == "pool-m-child"
    # A book cast with the old pool: its characters move to the new one, and the old voices go.
    voices = app[VOICES_KEY]
    voices.add("Pool: unknown child", wav_bytes(b"\x10\x00" * 2400, 24000), "Hi.", source="character",
               voice_id="pool-unknown-child")
    store = app[dz.DRAMA_STORE_KEY]
    script = store.script("book1")
    script["characters"]["tom"]["voice"] = "pool-unknown-child"
    script["characters"]["tom"].pop("pool_pick")
    script.pop("pool_v", None)
    store.save_script("book1", script)
    async with aiohttp.ClientSession() as session:
        await app[dz.ANALYZER_KEY]._backfill_pool(session)
    script = store.script("book1")
    assert script["characters"]["tom"]["voice"] == "pool-m-child" and script["pool_v"] == dz.POOL_V
    assert voices.get("pool-unknown-child") is None
    assert all(voices.get(vid) for vid in dz.POOL)  # every generic voice exists, for switching to
    # A pool voice whose description changed is designed again in place; one the reader edited is kept.
    analyzer = app[dz.ANALYZER_KEY]
    monkeypatch.setitem(dz.POOL, "pool-m-child", ("Male Child", "A boy of about nine with a bright voice."))
    monkeypatch.setitem(dz.POOL, "pool-f-child", ("Female Child", "A girl of about nine with a bright voice."))
    voices.update("pool-f-child", description="My own girl.")  # edited: it has a history now
    assert await analyzer._ensure_pool()
    assert voices.get("pool-m-child").instruct == "A boy of about nine with a bright voice."
    assert voices.get("pool-f-child").description == "My own girl."
    assert store.script("book1")["characters"]["tom"]["voice"] == "pool-m-child"

async def test_past_voices_list_earlier_versions_and_voices_and_bring_one_back(drama_client):
    from elvenreader_server.app import VOICES_KEY

    c = drama_client
    app = c.server.app
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    voices = app[VOICES_KEY]
    voices.add("Second", wav_bytes(b"\x20\x00" * 2400, 24000), "Hi.", voice_id="v2")
    await c.patch("/api/books/book1/characters/alice", json={"voice": "nar"}, headers=AUTH)
    await c.patch("/api/books/book1/characters/alice", json={"voice": "v2"}, headers=AUTH)  # nar: a voice she had
    await asyncio.sleep(0.2)  # reports keep the time to 0.1 s: the change below comes after
    first_audio = voices.get("v2").audio_sha
    voices.replace_audio("v2", wav_bytes(b"\x30\x00" * 2400, 24000), "New.")  # v2 as it was: an earlier version
    past = (await (await c.get("/api/books/book1/characters/alice/past-voices", headers=AUTH)).json())["past"]
    assert [(p["kind"], p["voice"]) for p in past] == [("version", "v2"), ("voice", "nar")]
    audio = await c.get(f"/api/voices/v2/history/{past[0]['stamp']}/audio", headers=AUTH)
    assert audio.status == 200 and (await audio.read())[:4] == b"RIFF"
    r = await c.post(f"/api/voices/v2/history/{past[0]['stamp']}/restore", headers=AUTH)
    assert r.status == 200 and voices.get("v2").audio_sha == first_audio
    # the version it replaced is a past voice now (so bringing one back can be undone the same way)
    past = (await (await c.get("/api/books/book1/characters/alice/past-voices", headers=AUTH)).json())["past"]
    assert past[0]["kind"] == "version" and len(past) <= 3
    assert (await c.get("/api/books/book1/characters/nobody/past-voices", headers=AUTH)).status == 404


# ---------------------------------------------------------------- voice profiles


def test_profile_updates_merge_specifics_and_never_touch_the_readers_fields():
    from elvenreader_server import profile as pf

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
    from elvenreader_server import profile as pf

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
    from elvenreader_server import epub

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


async def test_books_labelled_before_spoken_forms_get_them_between_analyses(aiohttp_client, tmp_path, monkeypatch):
    asked = []

    async def fake(self, session, system, user, schema, max_tokens=2048, **kw):
        assert "spoken" in schema["properties"], "only the spoken forms are asked for"
        asked.append(user)
        return {"spoken": [{"id": n, "written": "whaaaaaaat", "said": "what"}
                           for n, line in (ln.split("] ", 1) for ln in (x[1:] for x in user.splitlines()))
                           if "whaaa" in line]}

    monkeypatch.setattr(dz.LocalLlm, "complete", fake)
    app = make_app(make_cfg(tmp_path), specs=SPECS)
    store = app[dz.DRAMA_STORE_KEY]
    paras = ["\u201cWh\u2026 whaaaaaaat?!\u201d she cried.", "\u201cIt\u2019s fine,\u201d he said."]
    store.save_book("old1", {"title": "Old", "chapters": [{"title": "One", "paragraphs": paras, "quotes": [
        ["c0p0q0", 0, 0, paras[0].index("\u201d") + 1], ["c0p1q0", 1, 0, paras[1].index("\u201d") + 1]]}]})
    store.save_script("old1", {**store.script("old1"), "status": "done", "lines": {
        "c0p0q0": {"speaker": "ann", "delivery": "shouting"}, "c0p1q0": {"speaker": "tom", "delivery": None}}})
    c = await aiohttp_client(app)
    for _ in range(100):
        if store.script("old1").get("spoken_forms") == dz.SPOKEN_FORMS:
            break
        await asyncio.sleep(0.05)
    s = store.script("old1")
    assert s.get("spoken_forms") == dz.SPOKEN_FORMS and len(asked) == 1
    assert s["lines"]["c0p0q0"]["say"] == "\u201cWh\u2026 what?!\u201d" and "say" not in s["lines"]["c0p1q0"]
    pub = await (await c.get("/api/books/old1/script", headers=AUTH)).json()
    assert pub["lines"]["c0p0q0"]["say"] == "\u201cWh\u2026 what?!\u201d"


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
    from elvenreader_server import profile

    for t in ("none described", "Not mentioned", "N/A", "unknown", "nothing specified", "No details given"):
        assert profile.blank(t), t
    for t in ("pale skin, wears glasses", "none too friendly", "Nothing escapes her"):
        assert not profile.blank(t), t
    c = {"profile": {"gender": "female", "appearance": "none described", "notes": "kind; not mentioned"}}
    assert profile.clean(c) and c["profile"] == {"gender": "female", "notes": "kind"}
    assert not profile.merge(c, {"appearance": "Not described", "notes": "N/A"})
    assert "appears as" not in profile.voice_description({"profile": {"appearance": "none described"}})

def test_volume_key_orders_volumes():
    from elvenreader_server.series import volume_key

    assert volume_key("Sasaki and Peeps, Vol. 10") == [10]
    assert volume_key("Ascendance of a Bookworm: Part 2 Volume 3") == [2, 3]
    assert volume_key("Mistborn Book 2") == [2]
    assert volume_key("Sasaki and Peeps v10 [Yen Press]") == [10]
    assert volume_key("The Wandering Inn 4") == [4]
    assert volume_key("Alice's Adventures in Wonderland") == []


def test_applying_a_series_keeps_voices_the_reader_chose():
    from elvenreader_server import series as sr
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

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
    # Only characters of earlier volumes come in; a walk-on of one of them doesn't, one who recurs does.
    data["books"].update({"v0": "Saga 0", "v3": "Saga 3"})
    data["characters"]["toto"] = {"name": "Toto", "aliases": [], "role": "extra", "books": ["v1"]}
    data["characters"]["guard"] = {"name": "Guard", "aliases": [], "role": "extra", "books": ["v0", "v1"]}
    data["characters"]["kirito"] = {"name": "Kirito", "aliases": [], "role": "main", "books": ["v3"]}
    sr.seed(app, "v2v2v2v2", script)
    names = [c["name"] for c in script["characters"].values()]
    assert "Guard" in names and "Toto" not in names and "Kirito" not in names
    sr.seed(app, "v2v2v2v2", script, earlier_only=False)  # a labelled book taking the series' voices
    assert "Kirito" in [c["name"] for c in script["characters"].values()]


async def wait_cast(c, book):
    for _ in range(200):
        s = await (await c.get(f"/api/books/{book}/script", headers=AUTH)).json()
        if s["status"] == "done" and "voices" in s["detail"]:
            return s
        await asyncio.sleep(0.05)
    raise AssertionError(s["detail"])


async def test_series_can_be_changed_applied_renamed_and_merged(drama_client):
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

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
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

    c = drama_client
    store = c.server.app[DRAMA_STORE_KEY]
    store.save_book("old", {**book_payload(), "title": "Old Tale, Vol. 3"})
    (store.dir("old") / "meta.json").unlink()  # stored before meta.json existed
    r = await (await c.post("/api/books/old/series", json={"series": "Old Tales"}, headers=AUTH)).json()
    assert r["series"] == "Old Tales" and r["title"] == "Old Tale, Vol. 3"
    assert store.summary("old")["title"] == "Old Tale, Vol. 3"

async def test_a_characters_voice_follows_across_the_series(drama_client):
    from elvenreader_server.app import VOICES_KEY
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

    c = drama_client
    app = c.server.app
    store = app[DRAMA_STORE_KEY]
    app[VOICES_KEY].add("Other", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="other")
    for book in ("v1", "v2", "v3"):
        await c.post(f"/api/books/{book}", json={**book_payload(), "title": f"Tea Saga, Vol. {book[1]}", "analyze": True},
                     headers=AUTH)
        await wait_done(c, book)
    for book in ("v1", "v2"):
        s = store.script(book)
        s["characters"]["alice"]["voice"] = "nar"
        s["characters"]["tom"]["voice"] = "nar" if book == "v1" else None
        store.save_script(book, s)
    s = store.script("v3")
    s["characters"]["alice"]["voice"] = "my-v3-pick"  # chosen differently for this volume: left alone
    store.save_script("v3", s)

    # the editor's copy for Alice (the voice is shared with Tom) goes to Alice in every volume, not to Tom
    used = (await (await c.get("/api/voices/nar", headers=AUTH)).json())["used_by"]
    assert {u["series"] for u in used} == {"Tea Saga"} and all("aliases" in u for u in used)
    r = await c.post("/api/voices/nar/duplicate", json={"book": "v1", "character": "alice"}, headers=AUTH)
    copy = (await r.json())["id"]
    assert store.script("v1")["characters"]["alice"]["voice"] == copy
    assert store.script("v2")["characters"]["alice"]["voice"] == copy
    assert store.script("v1")["characters"]["tom"]["voice"] == "nar"
    assert store.script("v3")["characters"]["alice"]["voice"] == "my-v3-pick"
    series = (await (await c.get("/api/series", headers=AUTH)).json())["series"]
    assert next(g for g in series if g["name"] == "Tea Saga")["characters"]["alice"]["voice"] == copy

    # picking another voice in the drawer does the same
    r = await c.patch("/api/books/v2/characters/alice", json={"voice": "other"}, headers=AUTH)
    assert r.status == 200
    assert store.script("v1")["characters"]["alice"]["voice"] == "other"
    assert store.script("v3")["characters"]["alice"]["voice"] == "my-v3-pick"

async def test_automatic_casting_waits_while_someone_listens(drama_client, monkeypatch):
    import time

    from elvenreader_server.app import ENGINES_KEY

    app = drama_client.server.app
    manager = app[ENGINES_KEY]
    from elvenreader_server.app import CONFIG_KEY

    monkeypatch.setattr(app[CONFIG_KEY], "llm_mode", "shared")  # the listening wait of the other modes
    monkeypatch.setattr(dz, "LISTEN_QUIET_S", 0.6)
    monkeypatch.setattr(dz.asyncio, "sleep", _fast_sleep)
    t0 = time.monotonic()
    await dz._wait_for_background_turn(app, "nobook")  # nobody has listened: goes straight on
    assert time.monotonic() - t0 < 0.2
    manager._last_speech = time.monotonic()  # someone is listening
    t0 = time.monotonic()
    await dz._wait_for_background_turn(app, "nobook")
    assert time.monotonic() - t0 >= 0.5


_real_sleep = asyncio.sleep


async def _fast_sleep(s):
    await _real_sleep(min(s, 0.05))


async def test_analysis_runs_while_reading_and_waits_while_designing(drama_client):
    from elvenreader_server.activity import ACTIVITY_KEY

    c = drama_client
    analyzer = c.server.app[dz.ANALYZER_KEY]
    activity = c.server.app[ACTIVITY_KEY]
    activity.idle_after_s, activity.design_pause_s = 30, 0.8
    activity.touch()  # someone is reading: analysis goes on next to the speech model
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    s = await wait_done(c)
    assert s["status"] == "done" and analyzer.llm.shared
    activity.touch_design()  # someone designs voices: analysis waits until a pause after the last design
    await c.post("/api/books/book1/analyze", json={"relabel": True}, headers=AUTH)
    await asyncio.sleep(0.3)
    s = await (await c.get("/api/books/book1/script", headers=AUTH)).json()
    assert s["status"] in ("queued", "running") and "paused while you design voices" in s["detail"]
    s = await wait_done(c)
    assert s["status"] == "done"


async def test_steps_follow_the_state_of_the_site(drama_client):
    from elvenreader_server.activity import ACTIVITY_KEY

    app = drama_client.server.app
    analyzer, activity = app[dz.ANALYZER_KEY], app[ACTIVITY_KEY]
    activity.idle_after_s, activity.design_pause_s = 30, 0.4
    calls = []

    async def step():
        calls.append(analyzer.llm.shared)
        if len(calls) == 1:
            activity.touch()  # idle -> someone comes back mid-step: the whole-GPU LLM is stopped
            raise dz.LlmPreempted()
        return 42

    assert not activity.in_use
    assert await analyzer._gpu(step) == 42
    assert calls == [False, True]  # retried at once, next to the speech model
    activity.touch_design()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await analyzer._gpu(lambda: asyncio.sleep(0, 7)) == 7
    assert loop.time() - t0 >= 0.3  # waited for the design pause
    activity.touch_design()
    t0 = loop.time()
    assert await analyzer._gpu(lambda: asyncio.sleep(0, 8), interactive=True) == 8  # asked for: no wait
    assert loop.time() - t0 < 0.3


async def test_using_the_site_is_tracked(drama_client):
    from elvenreader_server.activity import ACTIVITY_KEY

    c = drama_client
    activity = c.server.app[ACTIVITY_KEY]
    activity.design_pause_s = 30
    assert activity.state == "idle"
    r = await c.post("/api/tts", json={"text": "Hello there."}, headers=AUTH)
    await r.read()
    s = await (await c.get("/api/settings", headers=AUTH)).json()
    assert s["activity"]["state"] == "reading" and s["llm_mode"] == "adaptive"
    await c.post("/api/voices/nar/design", json={"description": "deep", "engine": "fake", "count": 1}, headers=AUTH)
    assert (await (await c.post("/api/activity", headers=AUTH)).json())["state"] == "designing"
    r = await c.put("/api/settings", json={"design_pause_s": 120, "idle_after_s": 60}, headers=AUTH)
    assert (await r.json())["design_pause_s"] == 120 and activity.idle_after_s == 60


async def test_gpu_status_for_the_banner(drama_client):
    g = await (await drama_client.get("/api/gpu", headers=AUTH)).json()
    assert g["mode"] == "adaptive" and g["activity"]["state"] == "idle"
    assert set(g["engines"]["speech"]) >= {"state", "loading_s", "expected_s"} and "voice_designer" in g["engines"]
    assert g["llm"]["state"] == "stopped" and g["llm"]["book"] is None and g["llm"]["backend"] == "local"

def test_book_queue_reorders_removes_and_persists(tmp_path):
    q = dz.BookQueue(tmp_path / "q.json")

    async def fill():
        for b in ("a", "b", "c", "d"):
            await q.put(b)
        await q.put("b")  # already waiting: no duplicate

    asyncio.run(fill())
    assert q.items == ["a", "b", "c", "d"]
    q.reorder(["c", "zzz", "a"])  # named ones first; unknown ids ignored; the rest keep their order
    assert q.items == ["c", "a", "b", "d"]
    assert q.remove("b") and not q.remove("b")
    q.set_current("x")
    assert dz.BookQueue(tmp_path / "q.json").saved_order() == ["x", "c", "a", "d"]
    assert asyncio.run(q.get()) == "c"
    assert q.saved_order() == ["x", "a", "d"]


async def test_queue_api_reorders_removes_and_survives_a_restart(aiohttp_client, tmp_path, monkeypatch):
    release = asyncio.Event()

    async def slow(self, *a, **k):  # the first book stays "being analysed"
        await release.wait()
        raise dz.LlmError("stop")

    monkeypatch.setattr(dz.LocalLlm, "complete", slow)
    cfg = make_cfg(tmp_path)
    c = await aiohttp_client(make_app(cfg, specs=SPECS))
    for b in ("b1", "b2", "b3", "b4"):
        r = await c.post(f"/api/books/{b}", json={**book_payload(), "title": f"Book {b}", "analyze": True}, headers=AUTH)
        assert r.status == 200
    for _ in range(50):
        q = await (await c.get("/api/queue", headers=AUTH)).json()
        if q["current"]:
            break
        await asyncio.sleep(0.02)
    assert q["current"]["id"] == "b1" and q["current"]["title"] == "Book b1"
    assert [b["id"] for b in q["queued"]] == ["b2", "b3", "b4"]

    r = await c.put("/api/queue", json={"order": ["b4", "b2"]}, headers=AUTH)
    assert [b["id"] for b in (await r.json())["queued"]] == ["b4", "b2", "b3"]
    assert (await c.put("/api/queue", json={"order": "b4"}, headers=AUTH)).status == 400

    r = await c.delete("/api/queue/b2", headers=AUTH)
    assert [b["id"] for b in (await r.json())["queued"]] == ["b4", "b3"]
    assert (await c.delete("/api/queue/b2", headers=AUTH)).status == 404
    s = await (await c.get("/api/books/b2/script", headers=AUTH)).json()
    assert s["status"] == "none"
    books = {b["id"]: b for b in (await (await c.get("/api/books", headers=AUTH)).json())["books"]}
    assert books["b2"]["status"] == "none"
    # Analysing it again puts it at the back of the queue.
    assert (await c.post("/api/books/b2/analyze", headers=AUTH)).status == 200
    q = await (await c.get("/api/queue", headers=AUTH)).json()
    assert [b["id"] for b in q["queued"]] == ["b4", "b3", "b2"]

    # A restart resumes in the same order, the interrupted book first.
    await c.close()
    c = await aiohttp_client(make_app(cfg, specs=SPECS))
    for _ in range(50):
        q = await (await c.get("/api/queue", headers=AUTH)).json()
        if q["current"]:
            break
        await asyncio.sleep(0.02)
    assert q["current"]["id"] == "b1"
    assert [b["id"] for b in q["queued"]] == ["b4", "b3", "b2"]

    # A series volume queues ahead of its later volumes (they inherit its cast), not at the back.
    for b, title in (("s3", "Saga, Vol. 3"), ("s1", "Saga, Vol. 1"), ("s2", "Saga, Vol. 2: The Middle")):
        await c.post(f"/api/books/{b}", json={**book_payload(), "title": title, "analyze": True}, headers=AUTH)
    q = await (await c.get("/api/queue", headers=AUTH)).json()
    assert [b["id"] for b in q["queued"]] == ["b4", "b3", "b2", "s1", "s2", "s3"]
    release.set()

async def test_rename_book_sets_volume_order_keeps_series_and_survives_reupload(drama_client):
    c = drama_client
    for b, title in (("v1", "Saga, Vol. 1"), ("v2", "Saga 2: The Middle")):
        await c.post(f"/api/books/{b}", json={**book_payload(), "title": title}, headers=AUTH)
    await c.post("/api/books/v2/series", json={"series": "Saga"}, headers=AUTH)
    assert (await c.patch("/api/books/v2", json={"title": "  "}, headers=AUTH)).status == 400
    r = await c.patch("/api/books/v2", json={"title": "Saga, Vol. 2: The Middle"}, headers=AUTH)
    info = await r.json()
    assert info["title"] == "Saga, Vol. 2: The Middle" and info["volume"] == [2] and info["series"] == "Saga"
    assert (await (await c.get("/api/books/v2", headers=AUTH)).json())["title"] == "Saga, Vol. 2: The Middle"
    # An auto-detected series isn't lost when the new title wouldn't detect it.
    r = await c.patch("/api/books/v1", json={"title": "The Beginning"}, headers=AUTH)
    assert (await r.json())["series"] == "Saga"
    # Uploading the same file again keeps the reader's title.
    await c.post("/api/books/v2", json={**book_payload(), "title": "Saga 2: The Middle"}, headers=AUTH)
    books = {b["id"]: b for b in (await (await c.get("/api/books", headers=AUTH)).json())["books"]}
    assert books["v2"]["title"] == "Saga, Vol. 2: The Middle" and books["v2"]["series"] == "Saga"

async def test_queue_analyse_now_and_send_to_bottom_set_the_current_book_aside(aiohttp_client, tmp_path, monkeypatch):
    calls = []

    async def blocking(self, *a, **k):  # every book stays "being analysed" until set aside
        calls.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr(dz.LocalLlm, "complete", blocking)
    c = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))
    for b in ("b1", "b2", "b3"):
        await c.post(f"/api/books/{b}", json={**book_payload(), "title": f"Book {b}", "analyze": True}, headers=AUTH)

    async def queue():
        return await (await c.get("/api/queue", headers=AUTH)).json()

    async def current_is(book):
        for _ in range(100):
            q = await queue()
            if q["current"] and q["current"]["id"] == book:
                return q
            await asyncio.sleep(0.02)
        raise AssertionError(f"{book} never started: {q}")

    await current_is("b1")
    r = await c.post("/api/queue/b3/now", headers=AUTH)
    assert r.status == 200
    q = await current_is("b3")
    assert [b["id"] for b in q["queued"]] == ["b1", "b2"]  # the set-aside book goes next
    s = await (await c.get("/api/books/b1/script", headers=AUTH)).json()
    assert s["status"] == "queued" and "set aside" in s["detail"]

    assert (await c.post("/api/queue/b1/bottom", headers=AUTH)).status == 200
    assert [b["id"] for b in (await queue())["queued"]] == ["b2", "b1"]
    assert (await c.post("/api/queue/b3/bottom", headers=AUTH)).status == 200  # the current one
    q = await current_is("b2")
    assert [b["id"] for b in q["queued"]] == ["b1", "b3"]
    assert (await c.post("/api/queue/nope/now", headers=AUTH)).status in (400, 404)
    import json
    saved = next(tmp_path.rglob("analysis-queue.json"))
    assert json.loads(saved.read_text())["order"] == ["b2", "b1", "b3"]  # no duplicates of the set-aside book


async def test_series_volumes_are_always_analysed_in_order(aiohttp_client, tmp_path, monkeypatch):
    async def blocking(self, *a, **k):  # every book stays "being analysed" until set aside
        await asyncio.Event().wait()

    monkeypatch.setattr(dz.LocalLlm, "complete", blocking)
    c = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))

    async def add(b, title):
        await c.post(f"/api/books/{b}", json={**book_payload(), "title": title, "analyze": True}, headers=AUTH)

    async def current_is(book):
        for _ in range(100):
            q = await (await c.get("/api/queue", headers=AUTH)).json()
            if q["current"] and q["current"]["id"] == book:
                return q
            await asyncio.sleep(0.02)
        raise AssertionError(f"{book} never started: {q}")

    await add("s3", "Saga, Vol. 3")
    await current_is("s3")
    await add("s1", "Saga, Vol. 1")  # an earlier volume: the later one steps aside for it
    q = await current_is("s1")
    assert [b["id"] for b in q["queued"]] == ["s3"]
    await add("s2", "Saga, Vol. 2")
    assert [b["id"] for b in (await current_is("s1"))["queued"]] == ["s2", "s3"]
    # "Analyse now" on a later volume doesn't stop an earlier one; a hand-made order is followed by volume
    await c.post("/api/queue/s3/now", headers=AUTH)
    await c.put("/api/queue", json={"order": ["s3", "s2"]}, headers=AUTH)
    await asyncio.sleep(0.2)
    assert (await current_is("s1"))["current"]["id"] == "s1"
    await c.post("/api/queue/s1/bottom", headers=AUTH)  # sent to the back, it still goes first: the earliest volume
    await current_is("s1")

def test_labelling_prompt_shows_the_text_after_the_passage():
    paras = [f"Paragraph {i} \u201cLine {i}.\u201d" if i % 2 else f"Narration {i} " + "word " * 40 for i in range(30)]
    quotes = [[f"c0p{i}q0", i, *dz.quote_regions(p)[0]] for i, p in enumerate(paras) if dz.quote_regions(p)]
    book = {"title": "T", "author": "A", "chapters": [{"title": "One", "paragraphs": paras, "quotes": quotes}]}
    chunks = dz.make_chunks(book, max_words=200)
    first, last = chunks[0], chunks[-1]
    after = dz.following_text(book, first, 60)
    assert after.startswith(paras[first.paragraphs[-1] + 1]) and len(after.split()) >= 60
    assert dz.following_text(book, last, 60) == ""
    seen = {}

    class Llm:
        async def complete(self, session, system, user, schema, max_tokens=0, **kw):
            seen["user"] = user
            return {"quotes": [], "speakers": []}

    asyncio.run(dz.label_chunk(Llm(), None, book, {"characters": {}, "lines": {}}, first, after_words=60))
    assert "Story just after this passage" in seen["user"]
    assert seen["user"].index("Passage:") < seen["user"].index(paras[first.paragraphs[-1] + 1])
    asyncio.run(dz.label_chunk(Llm(), None, book, {"characters": {}, "lines": {}}, first))
    assert "Story just after" not in seen["user"]

async def test_cover_thumbnails_and_series_sheets(drama_client):
    import base64
    import io

    from PIL import Image

    def jpeg(color, size=(1400, 1980)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, "JPEG", quality=95)
        return buf.getvalue()

    c = drama_client
    for b, color in (("b1", (255, 0, 0)), ("b2", (0, 0, 255))):
        r = await c.post(f"/api/books/{b}", json={**book_payload(), "cover": base64.b64encode(jpeg(color)).decode()}, headers=AUTH)
        assert r.status == 200
    await c.post("/api/books/b3", json=book_payload(), headers=AUTH)  # no cover
    books = {b["id"]: b for b in (await (await c.get("/api/books", headers=AUTH)).json())["books"]}
    assert books["b1"]["cover_v"] > 0 and books["b3"]["cover_v"] == 0

    r = await c.get("/api/books/b1/cover?w=200&v=1", headers=AUTH)
    thumb = Image.open(io.BytesIO(await r.read()))
    assert thumb.width == 240 and "immutable" in r.headers["Cache-Control"]  # rounded up to a stored width

    r = await c.get("/api/covers?w=120&ids=b1,b3,b2", headers=AUTH)
    assert r.status == 200
    sheet = Image.open(io.BytesIO(await r.read())).convert("RGB")
    assert sheet.size == (360, 180)  # three 2:3 cells, left to right in the order asked
    assert sheet.getpixel((60, 90))[0] > 200 and sheet.getpixel((300, 90))[2] > 200
    assert max(sheet.getpixel((180, 90))) - min(sheet.getpixel((180, 90))) < 20  # no cover: a grey cell
    assert (await c.get("/api/covers?ids=../x", headers=AUTH)).status == 400
    assert (await c.get("/api/covers", headers=AUTH)).status == 400

def test_describing_the_cast_copes_with_a_profile_stored_as_null():
    class Llm:
        class cfg:
            llm_context = 32768

        async def complete(self, session, system, user, schema, max_tokens=2048, **kw):
            return {"characters": [{"name": "Tom", "appearance": "a small boy", "gender": "male", "age": "child",
                                    "accent": "neutral", "temperament": "shy", "voice": "A shy boy."}]}

    book = {"title": "T", "chapters": [{"title": "x", "paragraphs": ["\u201cHi,\u201d Tom said.", "\u201cBye.\u201d"],
                                        "quotes": [["c0p0q0", 0, 0, 5], ["c0p1q0", 1, 0, 6]]}]}
    script = {"characters": {"tom": {"name": "Tom", "aliases": [], "profile": None}},
              "lines": {"c0p0q0": {"speaker": "tom", "delivery": None}, "c0p1q0": {"speaker": "tom", "delivery": None}}}
    asyncio.run(dz.describe_cast(Llm(), None, book, script))
    assert script["characters"]["tom"]["profile"]["appearance"] == "a small boy"


async def test_who_narrates_a_section_can_be_corrected(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    r = await c.patch("/api/books/book1/narration", json={"chapter": 0, "paragraph": 0, "narrator": "alice"}, headers=AUTH)
    assert r.status == 200
    s = await r.json()
    assert s["narration"][0] == [0, 0, "alice"] and s["narration_fixed"] == {"0:0": "alice"}
    reports = [json.loads(x) for x in (c.server.app[dz.DRAMA_STORE_KEY].dir("book1") / "reports.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    assert reports[-1]["type"] == "narration" and reports[-1]["after"]["narrator"] == "alice"
    # the analysis running again keeps it
    script = c.server.app[dz.DRAMA_STORE_KEY].script("book1")
    assert dz.compute_narration(c.server.app[dz.DRAMA_STORE_KEY].book("book1"), script)[0] == [0, 0, "alice"]
    # back to what the analysis found
    s = await (await c.patch("/api/books/book1/narration", json={"chapter": 0, "paragraph": 0, "narrator": "auto"},
                             headers=AUTH)).json()
    assert s["narration_fixed"] == {} and [0, 0, "alice"] not in s["narration"]
    assert (await c.patch("/api/books/book1/narration", json={"chapter": 0, "paragraph": 0, "narrator": "nobody"},
                          headers=AUTH)).status == 404
    assert (await c.patch("/api/books/book1/narration", json={"chapter": 9, "paragraph": 0, "narrator": None},
                          headers=AUTH)).status == 400


async def test_a_line_can_be_read_as_part_of_the_narration(drama_client):
    from elvenreader_server import bookfile

    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"mention": True}, headers=AUTH)
    line = (await r.json())["lines"]["c0p2q11"]
    assert line["mention"] and line["speaker"] is None and line["delivery"] is None and line["corrected"]
    pub = await (await c.get("/api/books/book1/script", headers=AUTH)).json()
    book = c.server.app[dz.DRAMA_STORE_KEY].book("book1")
    segs = bookfile.segments(book, pub)
    assert "c0p2q11" not in [s["id"] for s in segs]  # read inside the paragraph's narration
    assert any(s["paragraph"] == 2 and s["kind"] == "narration" for s in segs)
    # and back: a line of its own, with the speaker given
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"mention": False, "speaker": "alice"}, headers=AUTH)
    line = (await r.json())["lines"]["c0p2q11"]
    assert line["speaker"] == "alice" and not line.get("mention")
    reports = (await (await c.get("/api/books/book1/reports", headers=AUTH)).json())["reports"]
    assert [x["type"] for x in reports if x["type"] == "mention"] == ["mention", "mention"]


async def test_users_read_everything_but_change_only_their_books(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)
    await wait_done(c)
    assert (await (await c.get("/api/me", headers=AUTH)).json())["owner"] is True
    sam = await (await c.post("/api/users", json={"name": "Sam"}, headers=AUTH)).json()
    SAM = {"Authorization": f"Bearer {sam['token']}"}
    assert await (await c.get("/api/me", headers=SAM)).json() == {"id": sam["id"], "name": "Sam", "owner": False}

    # Sam reads the admin's book, read-only.
    books = (await (await c.get("/api/books", headers=SAM)).json())["books"]
    assert [(b["id"], b["can_edit"]) for b in books] == [("book1", False)]
    assert (await (await c.get("/api/books/book1/script", headers=SAM)).json())["can_edit"] is False
    assert (await (await c.get("/api/books/book1/script", headers=AUTH)).json())["can_edit"] is True
    r = await c.patch("/api/books/book1/lines/c0p2q11", json={"speaker": "alice"}, headers=SAM)
    assert r.status == 403 and "books you added" in (await r.json())["error"]
    assert (await c.patch("/api/books/book1/characters/alice", json={"role": "extra"}, headers=SAM)).status == 403
    assert (await c.post("/api/books/book1/analyze", json={}, headers=SAM)).status == 403
    assert (await c.post("/api/pronunciations", json={"term": "Alice", "respelling": "Aleese", "scope": "book",
                                                      "book": "book1"}, headers=SAM)).status == 403
    # ...but may report wrong analysis and keep a place in it.
    assert (await c.post("/api/books/book1/reports", json={"line": "c0p3q14"}, headers=SAM)).status in (200, 201)
    assert (await c.put("/api/books/book1/bookmark", json={"chapter": 0, "paragraph": 1}, headers=SAM)).status == 200

    # A book Sam adds is his to change (and the admin's).
    await c.post("/api/books/book2", json={**book_payload(), "analyze": True}, headers=SAM)
    await wait_done(c, "book2")
    books = {b["id"]: b["can_edit"] for b in (await (await c.get("/api/books", headers=SAM)).json())["books"]}
    assert books == {"book1": False, "book2": True}
    assert (await c.patch("/api/books/book2/lines/c0p2q11", json={"speaker": "alice"}, headers=SAM)).status == 200
    assert (await c.post("/api/pronunciations", json={"term": "Alice", "respelling": "Aleese", "scope": "book",
                                                      "book": "book2"}, headers=SAM)).status in (200, 201)
    assert (await c.patch("/api/books/book2/lines/c0p2q11", json={"speaker": "tom"}, headers=AUTH)).status == 200

    # Site-wide things are the admin's.
    assert (await c.get("/api/users", headers=SAM)).status == 403
    assert (await c.post("/api/users", json={"name": "Eve"}, headers=SAM)).status == 403
    assert (await c.delete("/api/voices/nar", headers=SAM)).status == 403
    assert (await c.put("/api/queue", json={"order": []}, headers=SAM)).status == 403


async def test_user_pronunciations_follow_the_users_books(drama_client):
    c = drama_client
    await c.post("/api/books/book1", json={**book_payload(), "analyze": True}, headers=AUTH)  # the admin's
    await wait_done(c)
    sam = await (await c.post("/api/users", json={"name": "Sam"}, headers=AUTH)).json()
    SAM = {"Authorization": f"Bearer {sam['token']}"}
    await c.post("/api/books/book2", json={**book_payload(), "analyze": True}, headers=SAM)
    await wait_done(c, "book2")
    r = await c.post("/api/pronunciations", json={"term": "Alice", "respelling": "Aleese", "scope": "user"}, headers=AUTH)
    assert r.status == 201 and (await r.json())["scope"] == "user"
    entry = await r.json()
    mine = await (await c.get("/api/pronunciations?book=book1", headers=SAM)).json()
    assert [e["term"] for e in mine["user"]] == ["Alice"]  # the admin's book is read with the admin's entries
    his = await (await c.get("/api/pronunciations?book=book2", headers=SAM)).json()
    assert his["user"] == [] and his["users"] == [sam["id"]]
    # Sam keeps his own, and can't change the admin's.
    assert (await c.post("/api/pronunciations", json={"term": "Tom", "respelling": "Tome", "scope": "user"},
                         headers=SAM)).status == 201
    assert [e["term"] for e in (await (await c.get("/api/pronunciations?book=book2", headers=SAM)).json())["user"]] == ["Tom"]
    assert (await c.patch(f"/api/pronunciations/{entry['id']}", json={"respelling": "x"}, headers=SAM)).status == 403
    assert (await c.delete(f"/api/pronunciations/{entry['id']}", headers=SAM)).status == 403
    for body in ({"term": "X", "respelling": "y", "scope": "global"}, {"term": "X", "respelling": "y"}):
        assert (await c.post("/api/pronunciations", json=body, headers=AUTH)).status == 400
    # The Pronunciations page: Sam sees his own and those of his books' series; the admin sees everyone's.
    await c.post("/api/pronunciations", json={"term": "Bob", "respelling": "Bawb", "scope": "series", "book": "book1"},
                 headers=AUTH)
    his = await (await c.get("/api/pronunciations?all=1", headers=SAM)).json()
    assert sorted(e["term"] for e in his["entries"]) == ["Bob", "Tom"]  # book2 is in book1's series; not the admin's own
    assert {e["term"]: e["can_edit"] for e in his["entries"]} == {"Bob": False, "Tom": True}
    assert list(his["books"]) == ["book2"] and [x["can_edit"] for x in his["series"]] == [False]
    everyone = await (await c.get("/api/pronunciations?all=1", headers=AUTH)).json()
    assert sorted(e["term"] for e in everyone["entries"]) == ["Alice", "Bob", "Tom"]
    assert everyone["users"][sam["id"]] == "Sam" and all(e["can_edit"] for e in everyone["entries"])

