import asyncio
import dataclasses
import json

import pytest

from androidreader_server.app import make_app, pair_url, pairing_link
from androidreader_server.audio import wav_bytes
from androidreader_server.config import ServerConfig, load_config
from androidreader_server.engines.manager import EngineError, EngineManager
from androidreader_server.engines.registry import FAKE_ENGINE
from androidreader_server.voices import VoiceStore

TOKEN = "testtoken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WORD_BYTES = 16000 // 20 * 2  # fake engine: 50 ms of 16 kHz PCM16 per word

SLOW = dataclasses.replace(FAKE_ENGINE, id="fake_slow", label="Slow fake", defaults={"chunk_delay": 0.3})
BROKEN = dataclasses.replace(FAKE_ENGINE, id="broken", label="Broken", defaults={"fail_load": True})
SPECS = {s.id: s for s in (FAKE_ENGINE, SLOW, BROKEN)}


def make_cfg(tmp_path, **kw) -> ServerConfig:
    cfg = ServerConfig(token=TOKEN, default_engine="fake", preload=False, trim_silence=False,
                       sentence_pause_ms=0, synth_timeout_s=20, load_timeout_s=30, **kw)
    cfg.data_dir = str(tmp_path)
    cfg.models_dir = str(tmp_path / "models")
    return cfg


@pytest.fixture
async def client(aiohttp_client, tmp_path):
    VoiceStore(tmp_path / "voices").add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")
    return await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))


async def tts(client, **body):
    resp = await client.post("/api/tts", json=body, headers=AUTH)
    return resp, await resp.read()


# ---------------------------------------------------------------- auth & status


async def test_ping_is_public(client):
    resp = await client.get("/api/ping")
    assert (await resp.json())["service"] == "androidreader-server"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": TOKEN}])
async def test_api_requires_token(client, headers):
    assert (await client.post("/api/tts", json={"text": "hi"}, headers=headers)).status == 401


async def test_pair_page_public_without_token_and_index_on_loopback(client):
    page = await (await client.get("/pair")).text()
    assert TOKEN not in page and "areader" in page and "com.androidreader" in page
    assert (await client.get("/")).status == 200


def test_pairing_links_carry_all_urls():
    urls = ["http://192.168.1.5:3950", "http://10.8.0.2:3950"]
    assert pairing_link(urls, "a b") == (
        "areader://pair?url=http%3A%2F%2F192.168.1.5%3A3950&url=http%3A%2F%2F10.8.0.2%3A3950&token=a%20b"
    )
    assert pair_url(urls[0], urls, "tok") == "http://192.168.1.5:3950/pair#t=tok&u=http://10.8.0.2:3950"


def test_server_urls_put_private_addresses_first_and_add_extras(monkeypatch):
    from androidreader_server import app as app_module

    monkeypatch.setattr(app_module, "lan_addresses", lambda: [
        {"ip": "203.0.113.5", "kind": "other"}, {"ip": "192.168.1.5", "kind": "lan"},
    ])
    assert app_module.server_urls(3950, ["http://my-pc.vpn.example:3950/", "http://192.168.1.5:3950"]) == [
        "http://192.168.1.5:3950", "http://203.0.113.5:3950", "http://my-pc.vpn.example:3950",
    ]


async def test_health_and_engines(client):
    health = await (await client.get("/api/health", headers=AUTH)).json()
    assert health["default_engine"] == "fake" and health["voices"] == 1
    data = await (await client.get("/api/engines", headers=AUTH)).json()
    fake = next(e for e in data["engines"] if e["id"] == "fake")
    assert fake["installed"] and fake["delivery"] == "instruct"


async def test_voices_listing(client):
    data = await (await client.get("/api/voices", headers=AUTH)).json()
    assert data["default"] == "nar"
    assert data["voices"][0]["name"] == "Narrator"


def test_config_creates_and_persists_token(tmp_path, monkeypatch):
    monkeypatch.delenv("ANDROIDREADER_TOKEN", raising=False)
    first = load_config(tmp_path)
    assert len(first.token) == 16
    assert load_config(tmp_path).token == first.token
    monkeypatch.setenv("ANDROIDREADER_TOKEN", "override")
    assert load_config(tmp_path).token == "override"
    assert json.loads((tmp_path / "config.json").read_text())["token"] == first.token


def test_config_tolerates_bom(tmp_path, monkeypatch):
    monkeypatch.delenv("ANDROIDREADER_TOKEN", raising=False)
    (tmp_path / "config.json").write_bytes(b'\xef\xbb\xbf{"token": "abc", "default_engine": "fish_s2"}')
    cfg = load_config(tmp_path)
    assert (cfg.token, cfg.default_engine) == ("abc", "fish_s2")


# ---------------------------------------------------------------- synthesis


async def test_tts_streams_and_caches(client):
    resp, body = await tts(client, text="one two three.")
    assert resp.status == 200
    assert resp.headers["X-Sample-Rate"] == "16000"
    assert resp.headers["X-Cache"] == "miss" and resp.headers["X-Engine"] == "fake"
    assert len(body) == 3 * WORD_BYTES
    resp2, body2 = await tts(client, text="one two three.")
    assert resp2.headers["X-Cache"] == "hit" and body2 == body


async def test_cache_key_covers_voice_delivery_and_engine(client, tmp_path):
    await tts(client, text="alpha beta")
    resp, _ = await tts(client, text="alpha beta", delivery="shouting")
    assert resp.headers["X-Cache"] == "miss"
    resp, _ = await tts(client, text="alpha beta", engine="fake_slow")
    assert resp.headers["X-Cache"] == "miss"


async def test_pronunciation_edit_invalidates_cached_sentence(client):
    await tts(client, text="Hermione waved.")
    assert (await tts(client, text="Hermione waved."))[0].headers["X-Cache"] == "hit"
    r = await client.post("/api/pronunciations", json={"term": "Hermione", "respelling": "her-MY-oh-nee"},
                          headers=AUTH)
    assert r.status == 201
    entry = await r.json()
    resp, body = await tts(client, text="Hermione waved.")
    assert resp.headers["X-Cache"] == "miss"
    assert len(body) == 2 * WORD_BYTES  # "her-my-oh-nee waved." is still two words
    version = resp.headers["X-Pronunciation-Version"]
    await client.patch(f"/api/pronunciations/{entry['id']}", json={"respelling": "her mee own"}, headers=AUTH)
    resp, body = await tts(client, text="Hermione waved.")
    assert resp.headers["X-Cache"] == "miss" and resp.headers["X-Pronunciation-Version"] != version
    assert len(body) == 4 * WORD_BYTES  # the new respelling has three words
    # Another sentence that doesn't use the term is unaffected.
    await tts(client, text="Ron waved.")
    assert (await tts(client, text="Ron waved."))[0].headers["X-Cache"] == "hit"


async def test_book_entries_apply_only_with_book(client):
    await client.post("/api/pronunciations", json={"term": "Gandalf", "respelling": "gan dalf the grey",
                                                   "scope": "book", "book": "lotr"}, headers=AUTH)
    _, plain = await tts(client, text="Gandalf", cache=False)
    _, booked = await tts(client, text="Gandalf", book="lotr", cache=False)
    assert len(plain) == WORD_BYTES and len(booked) == 4 * WORD_BYTES
    listing = await (await client.get("/api/pronunciations?book=lotr", headers=AUTH)).json()
    assert listing["global"] == [] and listing["book"][0]["term"] == "Gandalf"


async def test_preview_with_unsaved_pronunciation(client):
    _, body = await tts(client, text="Siobhan left.", cache=False,
                        extra_pronunciations=[{"term": "Siobhan", "respelling": "shiv awn"}])
    assert len(body) == 3 * WORD_BYTES
    entries = await (await client.get("/api/pronunciations", headers=AUTH)).json()
    assert entries["global"] == []


async def test_speed_and_sentence_pause(client):
    _, normal = await tts(client, text="a b c d e f g h i j k l m n o p", cache=False)
    _, fast = await tts(client, text="a b c d e f g h i j k l m n o p", cache=False, speed=2.0)
    assert abs(len(fast) / len(normal) - 0.5) < 0.1
    _, paused = await tts(client, text="a b.", cache=False, sentence_pause_ms=500)
    assert len(paused) == 2 * WORD_BYTES + 8000 * 2


async def test_unspeakable_text_is_silence(client):
    resp, body = await tts(client, text=" ... ")
    assert resp.status == 200 and body == b""


async def test_errors(client, tmp_path):
    resp, body = await tts(client, text="please explode")
    assert resp.status == 502 and "engine exploded" in json.loads(body)["error"]
    resp, body = await tts(client, text="hi", engine="nope")
    assert resp.status == 400
    resp, body = await tts(client, text="hi", engine="broken")
    assert resp.status == 503 and "fake load failure" in json.loads(body)["error"]
    resp, _ = await tts(client, text="x" * 5000)
    assert resp.status == 413
    resp = await client.post("/api/tts", data=b"nope", headers=AUTH)
    assert resp.status == 400
    # The server still works after all that.
    resp, body = await tts(client, text="still fine")
    assert resp.status == 200 and len(body) == 2 * WORD_BYTES


async def test_no_voices_is_503(aiohttp_client, tmp_path):
    client = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))
    resp, body = await tts(client, text="hello")
    assert resp.status == 503 and "No voices" in json.loads(body)["error"]


async def test_cancel_mid_stream_releases_engine(client):
    resp = await client.post("/api/tts", json={"text": "w1 w2 w3 w4 w5 w6 w7 w8", "engine": "fake_slow"},
                             headers=AUTH)
    assert resp.status == 200
    assert len(await resp.content.readexactly(WORD_BYTES)) == WORD_BYTES
    resp.close()  # phone hit stop
    t0 = asyncio.get_running_loop().time()
    resp, body = await tts(client, text="after cancel", engine="fake_slow", cache=False)
    assert resp.status == 200 and len(body) == 2 * WORD_BYTES
    assert asyncio.get_running_loop().time() - t0 < 2.0  # did not wait for all 8 words


async def test_switching_engines_restarts_worker(client):
    await tts(client, text="one", engine="fake")
    health = await (await client.get("/api/health", headers=AUTH)).json()
    assert health["engine"]["active"] == "fake"
    await tts(client, text="one", engine="fake_slow")
    health = await (await client.get("/api/health", headers=AUTH)).json()
    assert health["engine"]["active"] == "fake_slow"
    engines = await (await client.get("/api/engines", headers=AUTH)).json()
    stats = {e["id"]: e["stats"] for e in engines["engines"]}
    assert stats["fake"]["requests"] == 1 and stats["fake"]["rtf"] is not None


async def test_design_voice_creates_locked_voice(client):
    r = await client.post("/api/voices/design", json={"engine": "fake", "description": "male, elderly",
                                                      "name": "Old man"}, headers=AUTH)
    assert r.status == 201
    voice = await r.json()
    listing = await (await client.get("/api/voices", headers=AUTH)).json()
    assert voice["id"] in [v["id"] for v in listing["voices"]]
    audio = await client.get(f"/api/voices/{voice['id']}/audio", headers=AUTH)
    assert (await audio.read())[:4] == b"RIFF"


async def test_voice_upload_and_delete(client):
    wav = wav_bytes(b"\x10\x00" * 100, 16000)
    import aiohttp

    form = aiohttp.FormData()
    form.add_field("name", "Mine")
    form.add_field("ref_text", "Some words.")
    form.add_field("audio", wav, filename="v.wav", content_type="audio/wav")
    r = await client.post("/api/voices", data=form, headers=AUTH)
    assert r.status == 201
    vid = (await r.json())["id"]
    assert (await client.delete(f"/api/voices/{vid}", headers=AUTH)).status == 200
    assert (await client.delete(f"/api/voices/{vid}", headers=AUTH)).status == 404


async def test_manager_reports_missing_venv(tmp_path):
    spec = dataclasses.replace(FAKE_ENGINE, id="ghost", in_process_python=False)
    manager = EngineManager({"ghost": spec}, tmp_path)
    with pytest.raises(EngineError) as err:
        await manager.preload("ghost")
    assert err.value.status == 503 and "not installed" in err.value.detail


async def test_gpu_lease_holds_tts_until_released(tmp_path):
    from androidreader_server.engines.manager import AudioChunk

    manager = EngineManager(SPECS, tmp_path)

    async def speak():
        n = 0
        async for ev in manager.synthesize("fake", {"text": "a b"}):
            n += isinstance(ev, AudioChunk)
        return n

    try:
        async with manager.exclusive():
            task = asyncio.ensure_future(speak())
            await asyncio.sleep(0.2)
            assert manager.tts_waiting and not task.done()
            assert manager.active is None
        assert await asyncio.wait_for(task, 20) == 2
        assert not manager.tts_waiting
        assert manager.idle_seconds() < 5  # "auto" LLM mode stays on the CPU right after speech
    finally:
        await manager.close()


# ---------------------------------------------------------------- voice editor


def _wav_seconds(data: bytes) -> float:
    import io
    import wave

    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


async def _put_audio(client, vid, wav, **fields):
    import aiohttp

    form = aiohttp.FormData()
    for k, v in fields.items():
        form.add_field(k, str(v))
    form.add_field("audio", wav, filename="clip.wav", content_type="audio/wav")
    return await client.put(f"/api/voices/{vid}/audio", data=form, headers=AUTH)


async def test_cache_key_follows_the_voice_content_not_just_its_id(client):
    await tts(client, text="alpha beta", voice="nar")
    assert (await tts(client, text="alpha beta", voice="nar"))[0].headers["X-Cache"] == "hit"
    r = await _put_audio(client, "nar", wav_bytes(b"\x20\x00" * 24000, 24000), ref_text="New words.")
    assert r.status == 200
    assert (await tts(client, text="alpha beta", voice="nar"))[0].headers["X-Cache"] == "miss"
    # metadata that changes the sound (seed) also changes the key; the name does not
    await client.patch("/api/voices/nar", json={"name": "Renamed"}, headers=AUTH)
    assert (await tts(client, text="alpha beta", voice="nar"))[0].headers["X-Cache"] == "hit"
    await client.patch("/api/voices/nar", json={"seed": 7}, headers=AUTH)
    assert (await tts(client, text="alpha beta", voice="nar"))[0].headers["X-Cache"] == "miss"
    # undoing back to the original reference finds the original clips again
    for _ in range(3):
        assert (await client.post("/api/voices/nar/revert", headers=AUTH)).status == 200
    assert (await tts(client, text="alpha beta", voice="nar"))[0].headers["X-Cache"] == "hit"


async def test_voice_patch_history_and_revert(client):
    r = await client.patch("/api/voices/nar", json={"name": "Old Tom", "description": "male, elderly", "seed": "42",
                                                    "language": "English"}, headers=AUTH)
    v = await r.json()
    assert (v["name"], v["description"], v["seed"], v["language"]) == ("Old Tom", "male, elderly", 42, "English")
    assert len(v["history"]) == 1 and v["history"][0]["name"] == "Narrator"
    assert (await client.patch("/api/voices/nar", json={"seed": "abc"}, headers=AUTH)).status == 400
    unchanged = await (await client.patch("/api/voices/nar", json={"name": "Old Tom"}, headers=AUTH)).json()
    assert len(unchanged["history"]) == 1  # no-op edits are not versions
    back = await (await client.post("/api/voices/nar/revert", headers=AUTH)).json()
    assert back["name"] == "Narrator" and back["seed"] is None and back["history"] == []
    assert (await client.post("/api/voices/nar/revert", headers=AUTH)).status == 409
    assert (await client.get("/api/voices/nope", headers=AUTH)).status == 404
    assert (await client.get("/api/voices/nar")).status == 401


async def test_replace_reference_trims_converts_and_transcribes(client):
    from androidreader_server.voice_edit import TRANSCRIBE_KEY

    heard = []

    async def fake(wav, language):
        heard.append((_wav_seconds(wav), language))
        return "Transcribed words."

    client.server.app[TRANSCRIBE_KEY].impl = fake
    clip = wav_bytes(b"\x10\x00" * 16000 * 4, 16000)  # 4 s
    r = await _put_audio(client, "nar", clip, start="0.5", end="3", language="English")
    v = await r.json()
    assert r.status == 200 and v["ref_text"] == "Transcribed words." and v["language"] == "English"
    assert heard == [(2.5, "English")]
    audio = await (await client.get("/api/voices/nar/audio", headers=AUTH)).read()
    assert _wav_seconds(audio) == 2.5
    # a typed transcript wins over Whisper
    r = await _put_audio(client, "nar", clip, ref_text="My own words.")
    assert (await r.json())["ref_text"] == "My own words." and len(heard) == 1
    assert (await _put_audio(client, "nar", clip, start="3.9")).status == 400  # too short
    assert (await _put_audio(client, "nar", b"not audio at all", ref_text="x")).status == 400
    r = await client.post("/api/transcribe", data=clip, headers=AUTH)
    assert (await r.json()) == {"text": "Transcribed words."}


async def test_design_candidates_then_keep_one(client):
    r = await client.post("/api/voices/nar/design", json={"description": "female, young, bright", "engine": "fake",
                                                           "seeds": [11, 22], "count": 2}, headers=AUTH)
    cands = (await r.json())["candidates"]
    assert [c["seed"] for c in cands] == [11, 22]
    audio = await client.get(f"/api/voice-candidates/{cands[1]['id']}/audio", headers=AUTH)
    assert (await audio.read())[:4] == b"RIFF"
    v = await (await client.post(f"/api/voices/nar/design/{cands[1]['id']}", headers=AUTH)).json()
    assert (v["seed"], v["instruct"], v["description"]) == (22, "female, young, bright", "female, young, bright")
    assert len(v["history"]) == 1
    # the other candidates are gone once one is kept
    assert (await client.get(f"/api/voice-candidates/{cands[0]['id']}/audio", headers=AUTH)).status == 404


async def test_used_by_and_duplicate_for_one_character(client):
    from androidreader_server.dramatize import DRAMA_STORE_KEY

    store = client.server.app[DRAMA_STORE_KEY]
    store.save_book("book1", {"title": "A Book", "chapters": []})
    s = store.script("book1")
    s["characters"] = {"tom": {"name": "Tom", "voice": "nar"}, "ann": {"name": "Ann", "voice": "nar"}}
    store.save_script("book1", s)
    v = await (await client.get("/api/voices/nar", headers=AUTH)).json()
    assert sorted(u["name"] for u in v["used_by"]) == ["Ann", "Tom"] and v["used_by"][0]["title"] == "A Book"
    listing = await (await client.get("/api/voices?details=1", headers=AUTH)).json()
    assert len(next(x for x in listing["voices"] if x["id"] == "nar")["used_by"]) == 2

    r = await client.post("/api/voices/nar/duplicate", json={"book": "book1", "character": "tom"}, headers=AUTH)
    copy = await r.json()
    assert r.status == 201 and copy["id"] != "nar" and copy["name"] == "Tom"
    assert [u["name"] for u in copy["used_by"]] == ["Tom"]
    assert store.script("book1")["characters"]["ann"]["voice"] == "nar"
    # the script tells clients which version of each cast voice they should hear
    script = await (await client.get("/api/books/book1/script", headers=AUTH)).json()
    assert set(script["voice_versions"]) == {"nar", copy["id"]}
    before = script["voice_versions"][copy["id"]]
    await client.patch(f"/api/voices/{copy['id']}", json={"seed": 5}, headers=AUTH)
    script = await (await client.get("/api/books/book1/script", headers=AUTH)).json()
    assert script["voice_versions"][copy["id"]] != before
    assert (await client.post("/api/voices/nar/duplicate", json={"book": "book1", "character": "zed"},
                              headers=AUTH)).status == 404


async def test_option_made_from_a_shared_voice_can_be_kept_on_a_copy(client):
    r = await client.post("/api/voices/nar/design", json={"description": "deep", "engine": "fake", "count": 1},
                          headers=AUTH)
    cand = (await r.json())["candidates"][0]
    copy = await (await client.post("/api/voices/nar/duplicate", json={}, headers=AUTH)).json()
    v = await (await client.post(f"/api/voices/{copy['id']}/design/{cand['id']}", headers=AUTH)).json()
    assert v["id"] == copy["id"] and v["instruct"] == "deep"
    original = await (await client.get("/api/voices/nar", headers=AUTH)).json()
    assert original["history"] == []  # the shared voice itself is untouched


async def test_voice_clip_is_revalidated_after_edits(client):
    r = await client.get("/api/voices/nar/audio", headers=AUTH)
    assert r.headers["Cache-Control"] == "no-cache" and r.headers.get("ETag")
    before = await r.read()
    await _put_audio(client, "nar", wav_bytes(b"\x30\x00" * 24000, 24000), ref_text="Other words.")
    assert await (await client.get("/api/voices/nar/audio", headers=AUTH)).read() != before
    listing = await client.get("/api/voices?all=1", headers=AUTH)
    assert listing.headers["Cache-Control"] == "no-cache"


def test_default_voice_prefers_your_own_voices_over_designed_ones(tmp_path):
    store = VoiceStore(tmp_path)
    wav = wav_bytes(b"\x10\x00" * 2400, 24000)
    store.add("Alpha (designed for a character)", wav, "x", source="design")
    store.add("The Documentarian", wav, "x", source="voicestudio")
    store.add("Zed", wav, "x", source="upload")
    assert store.default().name == "The Documentarian"
