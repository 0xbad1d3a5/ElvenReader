import asyncio
import contextlib
import dataclasses
import json

import pytest

from elvenreader_server.app import make_app, viewer_url
from elvenreader_server.audio import wav_bytes
from elvenreader_server.config import ServerConfig, load_config
from elvenreader_server.engines.manager import EngineError, EngineManager
from elvenreader_server.engines.registry import FAKE_ENGINE
from elvenreader_server.voices import VoiceStore

TOKEN = "testtoken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WORD_BYTES = 16000 // 20 * 2  # fake engine: 50 ms of 16 kHz PCM16 per word

SLOW = dataclasses.replace(FAKE_ENGINE, id="fake_slow", label="Slow fake", defaults={"chunk_delay": 0.3})
BROKEN = dataclasses.replace(FAKE_ENGINE, id="broken", label="Broken", defaults={"fail_load": True})
SPECS = {s.id: s for s in (FAKE_ENGINE, SLOW, BROKEN)}


def make_cfg(tmp_path, **kw) -> ServerConfig:
    kw.setdefault("idle_after_s", 0.5)  # background work goes on half a second after the site was used
    kw.setdefault("design_pause_s", 0.5)  # and analysis half a second after a voice was designed
    kw.setdefault("llm_url", "http://127.0.0.1:9/v1")  # never this PC's own LLM, if one is running
    kw.setdefault("level_speech", False)  # byte-exact comparisons; test_cached_clips_are_levelled turns it on
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
    assert (await resp.json())["service"] == "elvenreader-server"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": TOKEN}])
async def test_api_requires_token(client, headers):
    assert (await client.post("/api/tts", json={"text": "hi"}, headers=headers)).status == 401


async def test_index_on_loopback_and_no_phone_pairing(client):
    assert (await client.get("/")).status == 200
    assert (await client.get("/pair")).status in (401, 404)  # the phone app's pairing page is gone


def test_viewer_links_carry_the_token_in_the_fragment():
    assert viewer_url("http://192.168.1.5:3950", "a b") == "http://192.168.1.5:3950/books#token=a%20b"


def test_server_urls_put_private_addresses_first_and_add_extras(monkeypatch):
    from elvenreader_server import app as app_module

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
    monkeypatch.delenv("ELVENREADER_TOKEN", raising=False)
    first = load_config(tmp_path)
    assert len(first.token) == 16
    assert load_config(tmp_path).token == first.token
    monkeypatch.setenv("ELVENREADER_TOKEN", "override")
    assert load_config(tmp_path).token == "override"
    assert json.loads((tmp_path / "config.json").read_text())["token"] == first.token


def test_config_tolerates_bom(tmp_path, monkeypatch):
    monkeypatch.delenv("ELVENREADER_TOKEN", raising=False)
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


async def test_cached_clips_are_levelled(aiohttp_client, tmp_path):
    from elvenreader_server.audio import TARGET_SPEECH_DB, level, normalize

    VoiceStore(tmp_path / "voices").add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")
    c = await aiohttp_client(make_app(make_cfg(tmp_path, level_speech=True), specs=SPECS))
    _, streamed = await tts(c, text="one two three.")  # sent as it is made
    resp, cached = await tts(c, text="one two three.")
    # Kept evened out (no delivery tags), and evened again as it is served (clips cached before that existed).
    assert resp.headers["X-Cache"] == "hit" and cached == normalize(normalize(streamed, 16000, None), 16000, None)
    _, whisper = await tts(c, text="one two three.", delivery="whisper")
    resp, whisper2 = await tts(c, text="one two three.", delivery="whisper")
    assert whisper2 == level(whisper, 16000, TARGET_SPEECH_DB - 6)  # tagged: levelled as a whole, under the rest


async def test_cache_key_covers_voice_delivery_and_engine(client, tmp_path):
    await tts(client, text="alpha beta")
    resp, _ = await tts(client, text="alpha beta", delivery="shouting")
    assert resp.headers["X-Cache"] == "miss"
    r = await client.put("/api/settings", json={"speech_engine": "fake_slow"}, headers=AUTH)
    assert r.status == 200
    resp, _ = await tts(client, text="alpha beta")
    assert resp.headers["X-Cache"] == "miss"


async def test_pronunciation_edit_invalidates_cached_sentence(client):
    await tts(client, text="Hermione waved.", book="hp")
    assert (await tts(client, text="Hermione waved.", book="hp"))[0].headers["X-Cache"] == "hit"
    r = await client.post("/api/pronunciations", json={"term": "Hermione", "respelling": "her-MY-oh-nee",
                                                       "scope": "book", "book": "hp"}, headers=AUTH)
    assert r.status == 201
    entry = await r.json()
    resp, body = await tts(client, text="Hermione waved.", book="hp")
    assert resp.headers["X-Cache"] == "miss"
    assert len(body) == 2 * WORD_BYTES  # "her-my-oh-nee waved." is still two words
    version = resp.headers["X-Pronunciation-Version"]
    await client.patch(f"/api/pronunciations/{entry['id']}", json={"respelling": "her mee own"}, headers=AUTH)
    resp, body = await tts(client, text="Hermione waved.", book="hp")
    assert resp.headers["X-Cache"] == "miss" and resp.headers["X-Pronunciation-Version"] != version
    assert len(body) == 4 * WORD_BYTES  # the new respelling has three words
    # Another sentence that doesn't use the term is unaffected.
    await tts(client, text="Ron waved.", book="hp")
    assert (await tts(client, text="Ron waved.", book="hp"))[0].headers["X-Cache"] == "hit"


async def test_book_entries_apply_only_with_book(client):
    await client.post("/api/pronunciations", json={"term": "Gandalf", "respelling": "gan dalf the grey",
                                                   "scope": "book", "book": "lotr"}, headers=AUTH)
    _, plain = await tts(client, text="Gandalf", cache=False)
    _, booked = await tts(client, text="Gandalf", book="lotr", cache=False)
    assert len(plain) == WORD_BYTES and len(booked) == 4 * WORD_BYTES
    listing = await (await client.get("/api/pronunciations?book=lotr", headers=AUTH)).json()
    assert listing["series"] == [] and listing["book"][0]["term"] == "Gandalf"


async def test_preview_with_unsaved_pronunciation(client):
    _, body = await tts(client, text="Siobhan left.", cache=False,
                        extra_pronunciations=[{"term": "Siobhan", "respelling": "shiv awn"}])
    assert len(body) == 3 * WORD_BYTES
    entries = await (await client.get("/api/pronunciations", headers=AUTH)).json()
    assert entries["series"] == [] and entries["book"] == [] and "global" not in entries


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
    assert resp.status == 409
    assert (await client.put("/api/settings", json={"speech_engine": "broken"}, headers=AUTH)).status == 200
    resp, body = await tts(client, text="hi")
    assert resp.status == 503 and "fake load failure" in json.loads(body)["error"]
    assert (await client.put("/api/settings", json={"speech_engine": "fake"}, headers=AUTH)).status == 200
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


async def test_cancel_mid_stream_releases_engine(aiohttp_client, tmp_path):
    VoiceStore(tmp_path / "voices").add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")
    cfg = make_cfg(tmp_path)
    cfg.default_engine = "fake_slow"
    client = await aiohttp_client(make_app(cfg, specs=SPECS))
    resp = await client.post("/api/tts", json={"text": "w1 w2 w3 w4 w5 w6 w7 w8"}, headers=AUTH)
    assert resp.status == 200
    assert len(await resp.content.readexactly(WORD_BYTES)) == WORD_BYTES
    resp.close()  # phone hit stop
    t0 = asyncio.get_running_loop().time()
    resp, body = await tts(client, text="after cancel", cache=False)
    assert resp.status == 200 and len(body) == 2 * WORD_BYTES
    assert asyncio.get_running_loop().time() - t0 < 2.0  # did not wait for all 8 words


async def test_ranked_lock_serves_the_most_urgent_first():
    from elvenreader_server.engines.manager import RankedLock

    lock, order, ranks = RankedLock(), [], {"a2": 2, "b0": 0, "a1": 1, "late": 5}

    async def take(name):
        await lock.acquire(lambda: ranks[name])
        order.append(name)
        await asyncio.sleep(0)
        lock.release()

    await lock.acquire()  # someone is speaking
    tasks = [asyncio.ensure_future(take(n)) for n in ("a2", "late", "a1", "b0")]
    gone = asyncio.ensure_future(take("gone"))
    await asyncio.sleep(0.01)
    gone.cancel()  # a reader that left
    ranks["late"] = -1  # its reader caught up: now the most urgent
    lock.release()
    await asyncio.gather(*tasks)
    assert order == ["late", "b0", "a1", "a2"] and not lock.locked()


async def test_two_readers_get_their_next_passages_before_each_others_read_ahead(aiohttp_client, tmp_path):
    VoiceStore(tmp_path / "voices").add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")
    cfg = make_cfg(tmp_path)
    cfg.default_engine = "fake_slow"
    client = await aiohttp_client(make_app(cfg, specs=SPECS))
    done = []

    async def ask(reader, seq, playing=0):
        resp, _ = await tts(client, text=f"{reader}{seq}", reader=reader, seq=seq, playing=playing, cache=False)
        assert resp.status == 200
        done.append(f"{reader}{seq}")

    first = [asyncio.ensure_future(ask("a", 0))]  # Ana plays passage 0 (it is being made)...
    await asyncio.sleep(0.15)
    first += [asyncio.ensure_future(ask("a", i)) for i in range(1, 4)]  # ...and reads three ahead
    await asyncio.sleep(0.05)
    second = asyncio.ensure_future(ask("b", 0))  # then Ben starts reading
    await asyncio.gather(*first, second)
    assert done == ["a0", "b0", "a1", "a2", "a3"]  # Ben's first passage doesn't wait behind Ana's read-ahead


async def test_narration_can_be_read_by_a_second_model_next_to_the_speech_model(client):
    r = await client.put("/api/settings", json={"narration_engine": "fake_slow"}, headers=AUTH)
    assert r.status == 200 and (await r.json())["narration_engine"] == "fake_slow"
    resp, _ = await tts(client, text="The rain fell.", kind="narration", cache=False)
    assert resp.status == 200 and resp.headers["X-Engine"] == "fake_slow"
    resp, _ = await tts(client, text="Stop!", delivery="shouting", cache=False)  # a quotation: the speech model
    assert resp.headers["X-Engine"] == "fake"
    resp, _ = await tts(client, text="Hello.", cache=False)
    assert resp.headers["X-Engine"] == "fake"
    from elvenreader_server.app import ENGINES_KEY

    manager = client.server.app[ENGINES_KEY]
    assert manager.status()["narration"]["engine"] == "fake_slow" and manager.active == "fake"  # side by side
    # One model again: narration goes back to the speech model.
    await client.put("/api/settings", json={"narration_engine": ""}, headers=AUTH)
    resp, _ = await tts(client, text="The rain fell.", kind="narration", cache=False)
    assert resp.headers["X-Engine"] == "fake"
    assert (await client.put("/api/settings", json={"narration_engine": "nope"}, headers=AUTH)).status == 400


async def test_a_site_setting_picks_the_computer_that_analyses_books(aiohttp_client, tmp_path):
    client = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))
    s = await (await client.get("/api/settings", headers=AUTH)).json()
    assert s["llm_backend"] == "local" and s["llm_remote"] is None
    r = await client.put("/api/settings", json={"llm_backend": "remote"}, headers=AUTH)
    assert r.status == 400  # no server set up
    cfg = make_cfg(tmp_path / "b", llm_remote_url="http://10.8.0.4:9292/v1", llm_remote_model="qwen-x",
                   llm_remote_label="2x CMP 170HX")
    client = await aiohttp_client(make_app(cfg, specs=SPECS))
    s = await (await client.get("/api/settings", headers=AUTH)).json()
    assert s["llm_remote"] == {"label": "2x CMP 170HX", "model": "qwen-x", "host": "10.8.0.4:9292"}
    s = await (await client.put("/api/settings", json={"llm_backend": "remote"}, headers=AUTH)).json()
    assert s["llm_backend"] == "remote" and cfg.llm_backend == "remote"
    saved = json.loads((cfg.data_path / "config.json").read_text(encoding="utf-8"))
    assert saved["llm_backend"] == "remote"
    assert (await client.put("/api/settings", json={"llm_backend": "cloud"}, headers=AUTH)).status == 400


async def test_one_speech_model_for_the_site(client, tmp_path):
    await tts(client, text="one")
    health = await (await client.get("/api/health", headers=AUTH)).json()
    assert health["engine"]["active"] == "fake"
    # a request can't swap the model
    resp, body = await tts(client, text="one", engine="fake_slow")
    assert resp.status == 409 and "Settings" in json.loads(body)["error"]
    assert (await (await client.get("/api/health", headers=AUTH)).json())["engine"]["active"] == "fake"
    # the setting does, and is saved
    s = await (await client.get("/api/settings", headers=AUTH)).json()
    assert s["speech_engine"] == "fake" and {"fake", "fake_slow"} <= {e["id"] for e in s["engines"]}
    r = await client.put("/api/settings", json={"speech_engine": "fake_slow", "llm_mode": "cpu"}, headers=AUTH)
    assert r.status == 200
    for _ in range(100):
        if (await (await client.get("/api/health", headers=AUTH)).json())["engine"]["active"] == "fake_slow":
            break
        await asyncio.sleep(0.05)
    assert (await (await client.get("/api/health", headers=AUTH)).json())["engine"]["active"] == "fake_slow"
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["default_engine"] == "fake_slow" and saved["llm_mode"] == "cpu"
    resp, _ = await tts(client, text="two")
    assert resp.status == 200
    engines = await (await client.get("/api/engines", headers=AUTH)).json()
    stats = {e["id"]: e["stats"] for e in engines["engines"]}
    assert stats["fake"]["requests"] == 1 and stats["fake_slow"]["requests"] == 1
    assert (await client.put("/api/settings", json={"llm_mode": "turbo"}, headers=AUTH)).status == 400
    assert (await client.put("/api/settings", json={"speech_engine": "ghost"}, headers=AUTH)).status == 400


async def test_designing_with_another_model_keeps_the_speech_model_loaded(tmp_path):
    manager = EngineManager(SPECS, tmp_path, designer_idle_s=0.3)
    freed = []

    async def make_room():
        freed.append(1)

    manager.before_designer = make_room
    try:
        await manager.preload("fake")
        from elvenreader_server.engines.manager import AudioChunk

        chunks = [ev async for ev in manager.design("fake_slow", "deep", "a b", 1) if isinstance(ev, AudioChunk)]
        assert chunks and manager.active == "fake" and manager.designer == "fake_slow" and freed == [1]
        # designing with the speech model itself uses it
        [ev async for ev in manager.design("fake", "deep", "a", 1)]
        assert manager.designer == "fake_slow"
        # the LLM's turn on the GPU unloads the designer only
        async with manager.exclusive(evict=False):
            assert manager.active == "fake" and manager.designer is None
        [ev async for ev in manager.design("fake_slow", "deep", "a", 1)]
        await asyncio.sleep(0.8)  # idle: the designer goes away by itself
        assert manager.designer is None and manager.active == "fake"
    finally:
        await manager.close()


async def test_a_designer_load_cancelled_midway_is_not_left_loading(tmp_path):
    """Cancelling a design while the designer loads (the speech model switched under it) used to leave the slot
    "loading" with no engine: the banner showed "Loading the voice designer (null)" until a restart."""
    slow_load = dataclasses.replace(FAKE_ENGINE, id="slow_load", defaults={"load_delay": 2.0})
    manager = EngineManager({**SPECS, "slow_load": slow_load}, tmp_path, designer_idle_s=5)
    try:
        await manager.preload("fake")
        design = asyncio.ensure_future(_drain_events(manager.design("slow_load", "deep", "a b", 1)))
        for _ in range(50):
            if manager.status()["voice_designer"]["state"] == "loading":
                break
            await asyncio.sleep(0.02)
        assert manager.status()["voice_designer"]["engine"] == "slow_load"
        design.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await design
        status = manager.status()["voice_designer"]
        assert status["state"] == "idle" and status["engine"] is None and status["loading_s"] is None
    finally:
        await manager.close()


async def test_load_times_survive_a_restart_for_the_banner(tmp_path):
    """The banner's estimate for the first load after a reboot comes from earlier loads, not a 30 s guess."""
    path = tmp_path / "load-times.json"
    manager = EngineManager(SPECS, tmp_path, load_times_path=path)
    try:
        await manager.preload("fake")
    finally:
        await manager.close()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["main"]["fake"]) == 1
    saved["main"]["fake"] = [80.0, 240.0, 75.0]
    path.write_text(json.dumps(saved), encoding="utf-8")
    again = EngineManager(SPECS, tmp_path, load_times_path=path)
    again._main.loading = "fake"  # as while a load runs
    assert again.status()["speech"]["expected_s"] == 240.0  # the slowest recent load (a cold start)
    again._main.loading = None
    for _ in range(3):
        again._remember_load("main", "fake", 10.0)
    assert json.loads(path.read_text(encoding="utf-8"))["main"]["fake"] == [10.0, 10.0, 10.0]


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
    from elvenreader_server.engines.manager import AudioChunk

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


async def test_a_parkable_speech_model_steps_aside_without_reloading(tmp_path):
    from elvenreader_server.engines.manager import AudioChunk

    park = dataclasses.replace(FAKE_ENGINE, id="parky", defaults={"can_park": True, "load_delay": 0.5},
                               parked_vram_gb=2.0)
    broken = dataclasses.replace(FAKE_ENGINE, id="parky2", defaults={"can_park": True, "fail_unpark": True})
    manager = EngineManager({"parky": park, "parky2": broken, "fake": FAKE_ENGINE}, tmp_path)

    async def speak(engine):
        return sum([isinstance(ev, AudioChunk) async for ev in manager.synthesize(engine, {"text": "a b"})])

    try:
        await manager.preload("parky")
        pid = manager._main.worker.proc.pid
        async with manager.exclusive():
            assert manager.state == "parked" and manager.parked_vram_gb == 2.0
        assert await speak("parky") == 2  # back from RAM: the same process, no reload
        assert manager._main.worker.proc.pid == pid and manager.state == "ready" and manager.parked_vram_gb == 0
        assert manager.stats["parky"].load_s is not None
        # An engine that can't come back is loaded again instead.
        await manager.preload("parky2")
        pid = manager._main.worker.proc.pid
        async with manager.exclusive():
            pass
        assert await speak("parky2") == 2 and manager._main.worker.proc.pid != pid
        # One that can't park at all is stopped, as before.
        await manager.preload("fake")
        async with manager.exclusive():
            assert manager.active is None and manager.parked_vram_gb == 0
    finally:
        await manager.close()


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
    # bringing the original version back finds the original clips again
    original = (await (await client.get("/api/voices/nar", headers=AUTH)).json())["history"][-1]["stamp"]
    assert (await client.post(f"/api/voices/nar/history/{original}/restore", headers=AUTH)).status == 200
    assert (await tts(client, text="alpha beta", voice="nar"))[0].headers["X-Cache"] == "hit"


async def test_voice_patch_history_and_restore(client):
    r = await client.patch("/api/voices/nar", json={"name": "Old Tom", "description": "male, elderly", "seed": "42",
                                                    "language": "English"}, headers=AUTH)
    v = await r.json()
    assert (v["name"], v["description"], v["seed"], v["language"]) == ("Old Tom", "male, elderly", 42, "English")
    assert len(v["history"]) == 1 and v["history"][0]["name"] == "Narrator"
    assert (await client.patch("/api/voices/nar", json={"seed": "abc"}, headers=AUTH)).status == 400
    unchanged = await (await client.patch("/api/voices/nar", json={"name": "Old Tom"}, headers=AUTH)).json()
    assert len(unchanged["history"]) == 1  # no-op edits are not versions
    stamp = unchanged["history"][0]["stamp"]
    back = await (await client.post(f"/api/voices/nar/history/{stamp}/restore", headers=AUTH)).json()
    # the earlier version is back, and the one it replaced is kept (bringing it back is undone the same way)
    assert back["name"] == "Narrator" and back["seed"] is None and back["history"][0]["name"] == "Old Tom"
    assert (await client.post("/api/voices/nar/history/123/restore", headers=AUTH)).status == 404
    assert (await client.get("/api/voices/nope", headers=AUTH)).status == 404
    assert (await client.get("/api/voices/nar")).status == 401


async def test_replace_reference_trims_converts_and_transcribes(client):
    from elvenreader_server.voice_edit import TRANSCRIBE_KEY

    heard = []

    async def fake(wav, language):
        heard.append((_wav_seconds(wav), language))
        return "Transcribed words."

    client.server.app[TRANSCRIBE_KEY].impl = fake
    await client.patch("/api/voices/nar", json={"description": "A calm storyteller."}, headers=AUTH)
    clip = wav_bytes(b"\x10\x00" * 16000 * 4, 16000)  # 4 s
    r = await _put_audio(client, "nar", clip, start="0.5", end="3", language="English")
    v = await r.json()
    assert r.status == 200 and v["ref_text"] == "Transcribed words." and v["language"] == "English"
    # a recording now, not a design (the flag is kept with the voice); the design can be brought back
    assert v["recorded"] is True
    back = await (await client.post(f"/api/voices/nar/history/{v['history'][0]['stamp']}/restore", headers=AUTH)).json()
    assert back["description"] == "A calm storyteller."
    r = await _put_audio(client, "nar", clip, start="0.5", end="3", language="English")
    assert heard == [(2.5, "English")] * 2
    audio = await (await client.get("/api/voices/nar/audio", headers=AUTH)).read()
    assert _wav_seconds(audio) == 2.5
    # a typed transcript wins over Whisper
    r = await _put_audio(client, "nar", clip, ref_text="My own words.")
    assert (await r.json())["ref_text"] == "My own words." and len(heard) == 2
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
    assert v["recorded"] is False  # a design now, not the uploaded recording it was
    assert len(v["history"]) == 1
    # the other candidates are gone once one is kept
    assert (await client.get(f"/api/voice-candidates/{cands[0]['id']}/audio", headers=AUTH)).status == 404


async def test_used_by_and_duplicate_for_one_character(client):
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

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


def test_a_voice_is_a_recording_unless_it_says_the_design_passage(tmp_path):
    from elvenreader_server.voices import DESIGN_TEXT

    store = VoiceStore(tmp_path)
    wav = wav_bytes(b"\x10\x00" * 2400, 24000)
    designed = store.add("Myne", wav, DESIGN_TEXT, source="character", voice_id="design")
    assert designed.recorded is False and designed.is_recording is False
    # from before the flag: a clip saying anything else is a recording, its data untouched
    older = store.add("Old", wav, "I was really relieved.", source="character", voice_id="older")
    assert older.recorded is False and older.is_recording is True
    # uploads set the flag; a design put back clears it
    store.replace_audio("design", wav, "My own words.", recorded=True)
    assert store.get("design").recorded and store.get("design").details()["recorded"] is True
    store.replace_audio("design", wav, DESIGN_TEXT, instruct="A girl.")
    assert store.get("design").details()["recorded"] is False
    assert store.add("Mine", wav, "Hello.", voice_id="mine").recorded  # a voice made from your own clip


def test_default_voice_prefers_your_own_voices_over_designed_ones(tmp_path):
    store = VoiceStore(tmp_path)
    wav = wav_bytes(b"\x10\x00" * 2400, 24000)
    store.add("Alpha (designed for a character)", wav, "x", source="design")
    store.add("The Documentarian", wav, "x", source="voicestudio")
    store.add("Zed", wav, "x", source="upload")
    assert store.default().name == "The Documentarian"


async def test_design_younger_or_older_options(client):
    await client.patch("/api/voices/nar", json={"seed": 4242}, headers=AUTH)
    r = await client.post("/api/voices/nar/design", json={"description": "An adult woman with a warm, low voice.",
                                                           "engine": "fake", "count": 2, "age": -1}, headers=AUTH)
    body = await r.json()
    assert body["description"] == "A young woman with a warm, low voice."
    assert (body["age_from"], body["age_to"]) == ("adult", "young adult")
    assert body["candidates"][0]["seed"] == 4242  # the first option keeps the voice's seed
    assert all(c["description"] == body["description"] for c in body["candidates"])
    v = await (await client.post(f"/api/voices/nar/design/{body['candidates'][1]['id']}", headers=AUTH)).json()
    assert v["instruct"] == "A young woman with a warm, low voice."
    r = await client.post("/api/voices/nar/design", json={"description": "An elderly man.", "engine": "fake",
                                                           "count": 1, "age": 1}, headers=AUTH)
    assert r.status == 400 and "already elderly" in (await r.json())["error"]

def test_old_configs_move_the_llm_to_a_shared_gpu(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"token": "t", "llm_mode": "auto"}), encoding="utf-8")
    cfg = load_config(tmp_path)
    assert cfg.llm_mode == "adaptive" and cfg.settings_v == 3
    (tmp_path / "config.json").write_text(json.dumps({"token": "t", "llm_mode": "auto", "settings_v": 1}),
                                          encoding="utf-8")
    assert load_config(tmp_path).llm_mode == "auto"  # chosen after the migration: kept
    (tmp_path / "config.json").write_text(json.dumps({"token": "t", "llm_mode": "shared", "settings_v": 1}),
                                          encoding="utf-8")
    assert load_config(tmp_path).llm_mode == "adaptive"  # v2: the shared default becomes adaptive
    (tmp_path / "config.json").write_text(json.dumps({"token": "t", "voice_design_engine": "omnivoice", "settings_v": 2}),
                                          encoding="utf-8")
    assert load_config(tmp_path).voice_design_engine == "qwen3tts"  # v3: OmniVoice doesn't design voices

async def test_the_llm_waits_for_a_designer_in_use(tmp_path):
    """An LLM step queued while a voice is being designed waits until the designer has been idle."""
    loop = asyncio.get_running_loop()
    manager = EngineManager(SPECS, tmp_path, designer_idle_s=1.0)
    try:
        await manager.preload("fake")
        design = asyncio.ensure_future(_drain_events(manager.design("fake_slow", "deep", "a b", 1)))
        await asyncio.sleep(0.1)
        entered = []

        async def llm_step():
            async with manager.exclusive(evict=False):
                entered.append(loop.time())

        step = asyncio.ensure_future(llm_step())
        await design
        done = loop.time()
        assert manager.designer == "fake_slow"
        await asyncio.wait_for(step, 10)
        assert entered[0] - done >= 0.8 and manager.designer is None and manager.active == "fake"
    finally:
        await manager.close()


async def _drain_events(stream):
    return [ev async for ev in stream]

async def test_compressed_speech_for_the_website(client):
    import shutil

    resp, body = await tts(client, text="one two three four", format="pcm")
    assert resp.headers["Content-Type"].startswith("audio/L16") and len(body) == 4 * WORD_BYTES
    resp, body = await tts(client, text="one", format="flac")
    assert resp.status == 400
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    resp, body = await tts(client, text="alpha beta gamma delta", format="opus")
    assert resp.status == 200 and resp.headers["Content-Type"].startswith("audio/ogg")
    assert body[:4] == b"OggS" and resp.headers["X-Cache"] == "miss" and resp.headers["X-Encoding"] == "opus"
    resp, body = await tts(client, text="alpha beta gamma delta", format="mp3")
    assert resp.headers["Content-Type"] == "audio/mpeg" and resp.headers["X-Cache"] == "hit"
    assert body[:3] == b"ID3" or body[0] == 0xFF

async def test_series_pronunciations_apply_to_every_book_of_the_series(client):
    from elvenreader_server.dramatize import DRAMA_STORE_KEY

    store = client.server.app[DRAMA_STORE_KEY]
    for book, title in (("bw1", "Bookworm, Vol. 1"), ("bw2", "Bookworm, Vol. 2"), ("other", "Other Tale")):
        store.save_book(book, {"title": title, "chapters": []})
    r = await client.post("/api/pronunciations", json={"term": "Myne", "respelling": "my nuh",
                                                       "scope": "series", "book": "bw1"}, headers=AUTH)
    assert r.status == 201 and (await r.json())["series"] == "Bookworm"
    for book, words in (("bw1", 2), ("bw2", 2), ("other", 1)):
        _, body = await tts(client, text="Myne", book=book, cache=False)
        assert len(body) == words * WORD_BYTES, book
    listing = await (await client.get("/api/pronunciations?book=bw2", headers=AUTH)).json()
    assert listing["series_name"] == "Bookworm" and [e["term"] for e in listing["series"]] == ["Myne"]
    listing = await (await client.get("/api/pronunciations?series=Bookworm", headers=AUTH)).json()
    assert [e["term"] for e in listing["series"]] == ["Myne"]
    # renaming the series keeps them
    await client.post("/api/series/rename", json={"from": "Bookworm", "to": "Ascendance of a Bookworm"}, headers=AUTH)
    _, body = await tts(client, text="Myne", book="bw2", cache=False)
    assert len(body) == 2 * WORD_BYTES
    assert (await client.post("/api/pronunciations", json={"term": "X", "respelling": "y", "scope": "series"},
                              headers=AUTH)).status == 400
    # There are no site-wide pronunciations: each belongs to a series or a book.
    for body in ({"term": "X", "respelling": "y", "scope": "global"}, {"term": "X", "respelling": "y"}):
        assert (await client.post("/api/pronunciations", json=body, headers=AUTH)).status == 400


async def test_default_narrator_voice_setting(aiohttp_client, tmp_path):
    store = VoiceStore(tmp_path / "voices")
    store.add("Narrator", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="nar")
    store.add("Storyteller", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="story")
    store.add("Tom", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="ch-tom", source="character")
    client = await aiohttp_client(make_app(make_cfg(tmp_path), specs=SPECS))
    s = await (await client.get("/api/settings", headers=AUTH)).json()
    assert [v["id"] for v in s["voices"]] == ["nar", "story"]  # character voices aren't offered
    r = await client.put("/api/settings", json={"default_voice": "story"}, headers=AUTH)
    assert (await r.json())["default_voice"] == "story"
    assert (await (await client.get("/api/voices", headers=AUTH)).json())["default"] == "story"
    assert (await client.put("/api/settings", json={"default_voice": "ch-tom"}, headers=AUTH)).status == 400
    assert (await client.put("/api/settings", json={"default_voice": "ghost"}, headers=AUTH)).status == 400


async def test_two_readers_needing_new_speech_take_the_gpu_from_analysis(client, monkeypatch):
    from elvenreader_server import activity as act
    from elvenreader_server.activity import ACTIVITY_KEY

    a = client.server.app[ACTIVITY_KEY]
    crowds = []

    async def crowd():
        crowds.append(a.readers())

    a.on_crowd.append(crowd)
    await tts(client, text="One two.", reader="ana", seq=0, playing=0)
    assert a.readers() == 1 and not a.crowded
    await tts(client, text="One two.", reader="ben", seq=0, playing=0)  # made already: listening to stored speech
    assert a.readers() == 1 and not a.crowded
    await tts(client, text="Three four.", reader="ben", seq=1, playing=0)
    await asyncio.sleep(0)
    assert a.readers() == 2 and a.crowded and crowds == [2]
    assert (await (await client.get("/api/gpu", headers=AUTH)).json())["activity"]["crowded"] is True
    # Ben stops: analysis comes back only CROWD_HOLD_S later, not at once.
    a._making["ben"] -= act.READER_WINDOW_S + 1
    assert a.readers() == 1 and a.crowded
    a.last_crowded -= act.CROWD_HOLD_S + 1
    assert not a.crowded
