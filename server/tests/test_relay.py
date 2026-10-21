import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from vs_relay.app import make_app, pairing_link
from vs_relay.config import RelayConfig, load_config
from vs_relay.text import clamp_speed, clean_text, is_speakable

TOKEN = "testtoken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PCM = [b"\x01\x00" * 100, b"\x02\x00" * 50, b"\x03\x00" * 10]


class FakeVoiceStudio:
    """Mimics the /ws/tts, /model/status, /health and /v1/audio/voices endpoints."""

    def __init__(self):
        self.model_status = {"status": "ready", "loaded": True, "loading": False}
        self.requests: list[dict] = []
        self.closed = asyncio.Event()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws/tts", self.ws_tts)
        app.router.add_get("/model/status", self.status)
        app.router.add_get("/health", self.health)
        app.router.add_get("/engines/tts", self.engines)
        app.router.add_get("/v1/audio/voices", self.voices)
        return app

    async def status(self, request):
        return web.json_response(self.model_status)

    async def health(self, request):
        return web.json_response({"status": "ok", "version": "0.5.6", "device": "cuda"})

    async def engines(self, request):
        return web.json_response({"active": "omnivoice", "active_model": "k2-fsa/OmniVoice"})

    async def voices(self, request):
        return web.json_response(
            {
                "voices": [
                    {"voice_id": "alloy", "name": "Alloy", "type": "openai_alias"},
                    {"voice_id": "nova", "name": "Nova", "type": "openai_alias"},
                    {"voice_id": "demo0001", "name": "Demo", "type": "profile", "language": "English"},
                ]
            }
        )

    async def ws_tts(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for msg in ws:
                data = json.loads(msg.data)
                self.requests.append(data)
                text = data["text"]
                if "early error" in text:
                    await ws.send_json({"type": "error", "detail": "engine exploded"})
                    continue
                await ws.send_json({"type": "routing", "status": "accelerated"})
                await ws.send_json({"type": "start", "sample_rate": 24000, "channels": 1})
                if "late error" in text:
                    await ws.send_bytes(PCM[0])
                    await ws.send_json({"type": "error", "detail": "died mid-way"})
                    continue
                if "slow" in text:
                    await ws.send_bytes(PCM[0])
                    await asyncio.wait_for(ws.receive(), timeout=30)  # returns when relay closes
                    break
                for chunk in PCM:
                    await ws.send_bytes(chunk)
                await ws.send_json({"type": "done", "samples": 160, "sample_rate": 24000})
        finally:
            self.closed.set()
        return ws


@pytest.fixture
async def fake_vs(aiohttp_server):
    fake = FakeVoiceStudio()
    server = await aiohttp_server(fake.app())
    fake.url = str(server.make_url("")).rstrip("/")
    return fake


@pytest.fixture
async def relay(aiohttp_client, fake_vs):
    cfg = RelayConfig(token=TOKEN, voicestudio_url=fake_vs.url, first_audio_timeout_s=5, chunk_timeout_s=5)
    return await aiohttp_client(make_app(cfg))


# ---------------------------------------------------------------- text helpers


def test_clean_text_strips_invisible_and_control_chars():
    assert clean_text("Hy\u00adphen\u200bated\n\tline\x00 end  ") == "Hyphenated line end"


def test_is_speakable():
    assert is_speakable("Hello")
    assert is_speakable("42")
    assert is_speakable("Привет")
    assert not is_speakable("...")
    assert not is_speakable(" * \u2014 ")
    assert not is_speakable("")


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, 1.0), ("1.5", 1.5), (0.1, 0.5), (9, 2.5), (None, 1.0), ("x", 1.0), (float("nan"), 1.0)],
)
def test_clamp_speed(value, expected):
    assert clamp_speed(value) == expected


def test_pairing_link_is_url_encoded():
    assert pairing_link("http://192.168.1.5:3950", "ab c") == (
        "vstts://pair?url=http%3A%2F%2F192.168.1.5%3A3950&token=ab%20c"
    )


def test_config_creates_and_persists_token(tmp_path, monkeypatch):
    monkeypatch.delenv("VS_RELAY_TOKEN", raising=False)
    first = load_config(tmp_path)
    assert len(first.token) == 16
    assert load_config(tmp_path).token == first.token
    monkeypatch.setenv("VS_RELAY_TOKEN", "override")
    assert load_config(tmp_path).token == "override"
    assert json.loads((tmp_path / "config.json").read_text())["token"] == first.token


# ---------------------------------------------------------------- auth


async def test_ping_is_public(relay):
    resp = await relay.get("/api/ping")
    assert resp.status == 200
    assert (await resp.json())["service"] == "voicestudio-tts-relay"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
async def test_api_requires_token(relay, headers):
    resp = await relay.post("/api/tts", json={"text": "hi"}, headers=headers)
    assert resp.status == 401


async def test_pair_page_is_public_and_has_no_token(relay):
    resp = await relay.get("/pair")
    assert resp.status == 200
    assert TOKEN not in await resp.text()


async def test_index_served_on_loopback(relay):
    resp = await relay.get("/")
    assert resp.status == 200


# ---------------------------------------------------------------- endpoints


async def test_health_reports_voicestudio(relay):
    data = await (await relay.get("/api/health", headers=AUTH)).json()
    assert data["voicestudio"]["reachable"] is True
    assert data["voicestudio"]["engine"] == "omnivoice"
    assert data["voicestudio"]["model_status"] == "ready"


async def test_voices_collapses_aliases(relay):
    data = await (await relay.get("/api/voices", headers=AUTH)).json()
    assert [v["id"] for v in data["voices"]] == ["", "demo0001"]


async def test_tts_streams_pcm(relay, fake_vs):
    resp = await relay.post(
        "/api/tts",
        json={"text": "Hello world.", "voice": "demo0001", "language": "English", "speed": 9},
        headers=AUTH,
    )
    assert resp.status == 200
    assert resp.headers["X-Sample-Rate"] == "24000"
    assert await resp.read() == b"".join(PCM)
    assert fake_vs.requests == [
        {"text": "Hello world.", "speed": 2.5, "voice": "demo0001", "language": "English"}
    ]


async def test_tts_unspeakable_text_returns_silence_without_calling_voicestudio(relay, fake_vs):
    resp = await relay.post("/api/tts", json={"text": " ... "}, headers=AUTH)
    assert resp.status == 200
    assert await resp.read() == b""
    assert fake_vs.requests == []


async def test_tts_rejects_bad_body(relay):
    resp = await relay.post("/api/tts", data=b"not json", headers=AUTH)
    assert resp.status == 400


async def test_tts_error_before_audio_is_json(relay):
    resp = await relay.post("/api/tts", json={"text": "early error"}, headers=AUTH)
    assert resp.status == 502
    assert (await resp.json())["error"] == "engine exploded"


async def test_tts_fails_fast_while_model_loading(relay, fake_vs):
    fake_vs.model_status = {"status": "loading", "loading": True, "detail": "Loading TTS weights"}
    resp = await relay.post("/api/tts", json={"text": "Hello"}, headers=AUTH)
    assert resp.status == 503
    assert "Loading TTS weights" in (await resp.json())["error"]
    assert fake_vs.requests == []


async def test_tts_error_mid_stream_truncates_response(relay):
    resp = await relay.post("/api/tts", json={"text": "late error"}, headers=AUTH)
    assert resp.status == 200
    with pytest.raises(aiohttp.ClientPayloadError):
        await resp.read()


async def test_tts_voicestudio_unreachable(aiohttp_client, unused_tcp_port):
    cfg = RelayConfig(token=TOKEN, voicestudio_url=f"http://127.0.0.1:{unused_tcp_port}", connect_timeout_s=2)
    client = await aiohttp_client(make_app(cfg))
    resp = await client.post("/api/tts", json={"text": "Hello"}, headers=AUTH)
    assert resp.status == 503
    assert "not reachable" in (await resp.json())["error"]
    health = await (await client.get("/api/health", headers=AUTH)).json()
    assert health["voicestudio"]["reachable"] is False


async def test_tts_cancel_closes_voicestudio_socket(relay, fake_vs):
    resp = await relay.post("/api/tts", json={"text": "slow sentence"}, headers=AUTH)
    assert resp.status == 200
    assert await resp.content.readexactly(len(PCM[0])) == PCM[0]
    resp.close()  # phone hits stop
    await asyncio.wait_for(fake_vs.closed.wait(), timeout=5)
