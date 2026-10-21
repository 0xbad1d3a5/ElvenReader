"""aiohttp application: the phone-facing API of the relay."""
from __future__ import annotations

import asyncio
import contextlib
import html
import io
import logging
import secrets
import time
from urllib.parse import quote

import aiohttp
from aiohttp import web

from . import __version__
from .config import RelayConfig
from .netinfo import lan_addresses
from .text import MAX_TEXT_CHARS, clamp_speed, clean_text, is_speakable
from .voicestudio import AudioChunk, StreamStart, VoiceStudioClient, VoiceStudioError

logger = logging.getLogger("vs_relay")

ANDROID_PACKAGE = "com.androidvoicestudio.tts"
CONFIG_KEY = web.AppKey("config", RelayConfig)
VS_KEY = web.AppKey("voicestudio", VoiceStudioClient)

_PUBLIC_PATHS = {"/api/ping", "/pair"}
_LOOPBACK_ONLY_PATHS = {"/"}
_LOOPBACK = {"127.0.0.1", "::1"}


def _json_error(status: int, detail: str) -> web.Response:
    return web.json_response({"error": detail}, status=status)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    path = request.path
    if path in _PUBLIC_PATHS:
        return await handler(request)
    if path in _LOOPBACK_ONLY_PATHS:
        if request.remote not in _LOOPBACK:
            raise web.HTTPNotFound()
        return await handler(request)
    expected = request.app[CONFIG_KEY].token
    header = request.headers.get("Authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        return _json_error(401, "Missing or invalid relay token")
    return await handler(request)


async def ping(request: web.Request) -> web.Response:
    return web.json_response({"service": "voicestudio-tts-relay", "version": __version__})


async def health(request: web.Request) -> web.Response:
    vs = await request.app[VS_KEY].health()
    return web.json_response({"relay": "ok", "version": __version__, "voicestudio": vs})


async def voices(request: web.Request) -> web.Response:
    try:
        items = await request.app[VS_KEY].voices()
    except VoiceStudioError as exc:
        return _json_error(exc.status, exc.detail)
    return web.json_response({"voices": items})


def _audio_headers(sample_rate: int) -> dict:
    return {
        "Content-Type": f"audio/L16; rate={sample_rate}; channels=1",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "1",
        "X-Encoding": "pcm16le",
        "Cache-Control": "no-store",
    }


async def tts(request: web.Request) -> web.StreamResponse:
    """Synthesize one utterance and stream raw PCM16LE mono back.

    Errors that happen before any audio is produced are reported as JSON with a
    4xx/5xx status. An error after audio has started aborts the connection, so
    the client sees a truncated chunked body rather than a clean end.
    """
    try:
        body = await request.json()
    except ValueError:
        return _json_error(400, "Body must be JSON")
    if not isinstance(body, dict):
        return _json_error(400, "Body must be a JSON object")

    text = clean_text(str(body.get("text") or ""))
    if len(text) > MAX_TEXT_CHARS:
        return _json_error(413, f"Text longer than {MAX_TEXT_CHARS} characters")
    if not is_speakable(text):
        return web.Response(body=b"", headers=_audio_headers(24000))

    payload: dict = {"text": text, "speed": clamp_speed(body.get("speed"))}
    for key in ("voice", "language", "instruct"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            payload[key] = value.strip()

    vs = request.app[VS_KEY]
    try:
        await vs.ensure_ready()
    except VoiceStudioError as exc:
        return _json_error(exc.status, exc.detail)

    t0 = time.perf_counter()
    first_audio: float | None = None
    total = 0
    resp: web.StreamResponse | None = None
    stream = vs.stream_tts(payload)
    try:
        async with contextlib.aclosing(stream):
            async for event in stream:
                if isinstance(event, StreamStart):
                    resp = web.StreamResponse(headers=_audio_headers(event.sample_rate))
                    resp.enable_chunked_encoding()
                    await resp.prepare(request)
                elif isinstance(event, AudioChunk) and resp is not None:
                    if first_audio is None:
                        first_audio = time.perf_counter() - t0
                    total += len(event.pcm)
                    await resp.write(event.pcm)
    except VoiceStudioError as exc:
        if resp is None:
            logger.warning("tts failed: %s", exc.detail)
            return _json_error(exc.status, exc.detail)
        logger.warning("tts failed mid-stream, aborting: %s", exc.detail)
        if request.transport is not None:
            request.transport.close()
        return resp
    except (ConnectionResetError, aiohttp.ClientConnectionResetError):
        logger.info("phone cancelled after %d bytes", total)
        return resp or web.Response(status=499)
    except asyncio.CancelledError:
        logger.info("phone cancelled after %d bytes", total)
        raise

    if resp is None:
        return web.Response(body=b"", headers=_audio_headers(24000))
    await resp.write_eof()
    logger.info(
        "tts %d chars -> %.1fs audio, first audio %s, total %.2fs",
        len(text),
        total / 2 / 24000,
        f"{first_audio * 1000:.0f}ms" if first_audio is not None else "n/a",
        time.perf_counter() - t0,
    )
    return resp


def pairing_link(base_url: str, token: str) -> str:
    """Deep link understood by the Android app's settings screen."""
    return f"vstts://pair?url={quote(base_url, safe='')}&token={quote(token, safe='')}"


_PAIR_PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Pair VoiceStudio TTS</title><style>body{font-family:sans-serif;margin:2em;line-height:1.5}
a.btn{display:inline-block;padding:.8em 1.2em;background:#3557d6;color:#fff;border-radius:.5em;text-decoration:none}
code{background:#eee;padding:.1em .3em}</style></head><body>
<h2>VoiceStudio TTS</h2><p id=msg>Opening the app&hellip;</p>
<p><a class=btn id=open href="#">Open VoiceStudio TTS</a></p>
<p>Or enter manually in the app:<br>Server URL: <code id=url></code><br>Token: <code id=tok></code></p>
<script>
var tok = decodeURIComponent(location.hash.slice(1)), url = location.origin;
document.getElementById('url').textContent = url;
document.getElementById('tok').textContent = tok || '(missing - scan the QR code again)';
var q = 'url=' + encodeURIComponent(url) + '&token=' + encodeURIComponent(tok);
document.getElementById('open').href = 'intent://pair?' + q + '#Intent;scheme=vstts;package=__PKG__;end';
document.getElementById('msg').textContent = tok ? 'Tap the button to pair this phone.' : 'No token in link.';
</script></body></html>"""


async def pair_page(request: web.Request) -> web.Response:
    # Public on purpose: the token lives in the URL fragment, which browsers never send to the server.
    return web.Response(text=_PAIR_PAGE.replace("__PKG__", ANDROID_PACKAGE), content_type="text/html")


def _qr_svg(data: str) -> str:
    import qrcode
    import qrcode.image.svg

    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=7, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8")


async def index(request: web.Request) -> web.Response:
    """Loopback-only page showing pairing QR codes (it contains the token)."""
    cfg = request.app[CONFIG_KEY]
    parts = [
        "<!doctype html><html><head><title>VoiceStudio TTS relay</title>"
        "<style>body{font-family:sans-serif;margin:2em}.card{display:inline-block;margin:1em;"
        "padding:1em;border:1px solid #ccc;border-radius:.5em;vertical-align:top}</style></head><body>",
        "<h2>VoiceStudio TTS relay</h2>",
        "<p>Scan a code with the phone's camera, open the link, then tap <b>Open VoiceStudio TTS</b>.</p>",
    ]
    addresses = lan_addresses()
    if not addresses:
        parts.append("<p><b>No LAN address found.</b> Is this PC connected to a network?</p>")
    for addr in addresses:
        base = f"http://{addr['ip']}:{cfg.port}"
        url = f"{base}/pair#{cfg.token}"
        parts.append(
            f"<div class=card><h3>{html.escape(addr['ip'])} ({addr['kind']})</h3>{_qr_svg(url)}"
            f"<p>URL: <code>{html.escape(base)}</code><br>Token: <code>{html.escape(cfg.token)}</code></p></div>"
        )
    parts.append("</body></html>")
    return web.Response(text="".join(parts), content_type="text/html")


def make_app(cfg: RelayConfig, vs_client: VoiceStudioClient | None = None) -> web.Application:
    app = web.Application(middlewares=[auth_middleware], client_max_size=256 * 1024)
    app[CONFIG_KEY] = cfg

    async def _client_ctx(app: web.Application):
        session = None
        if vs_client is None:
            session = aiohttp.ClientSession()
            app[VS_KEY] = VoiceStudioClient(
                session,
                cfg.voicestudio_url,
                connect_timeout_s=cfg.connect_timeout_s,
                first_audio_timeout_s=cfg.first_audio_timeout_s,
                chunk_timeout_s=cfg.chunk_timeout_s,
            )
        else:
            app[VS_KEY] = vs_client
        yield
        if session is not None:
            await session.close()

    app.cleanup_ctx.append(_client_ctx)
    app.router.add_get("/", index)
    app.router.add_get("/pair", pair_page)
    app.router.add_get("/api/ping", ping)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/voices", voices)
    app.router.add_post("/api/tts", tts)
    return app
