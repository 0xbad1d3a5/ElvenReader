"""Thin async client for the parts of the VoiceStudio backend the relay uses."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("vs_relay.voicestudio")

DEFAULT_SAMPLE_RATE = 24000


class VoiceStudioError(Exception):
    """Synthesis failed; ``status`` is the HTTP status the relay should report."""

    def __init__(self, detail: str, status: int = 502):
        super().__init__(detail)
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class StreamStart:
    sample_rate: int
    engine: str | None = None


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes


class VoiceStudioClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        *,
        connect_timeout_s: float = 5.0,
        first_audio_timeout_s: float = 90.0,
        chunk_timeout_s: float = 60.0,
    ):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.connect_timeout_s = connect_timeout_s
        self.first_audio_timeout_s = first_audio_timeout_s
        self.chunk_timeout_s = chunk_timeout_s

    async def _get_json(self, path: str, timeout_s: float = 10.0):
        try:
            async with self.session.get(
                self.base_url + path, timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as resp:
                if resp.status >= 400:
                    raise VoiceStudioError(f"VoiceStudio {path} returned HTTP {resp.status}")
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
            raise VoiceStudioError(
                f"VoiceStudio is not reachable at {self.base_url} ({exc.__class__.__name__})",
                status=503,
            ) from exc

    async def model_status(self) -> dict:
        return await self._get_json("/model/status", timeout_s=3.0)

    async def health(self) -> dict:
        """Summarise VoiceStudio state for the phone's settings screen."""
        out: dict = {"reachable": False}
        try:
            info = await self._get_json("/health", timeout_s=3.0)
        except VoiceStudioError as exc:
            out["detail"] = exc.detail
            return out
        out.update(reachable=True, version=info.get("version"), device=info.get("device"))
        try:
            status = await self.model_status()
            out["model_status"] = status.get("status")
            out["model_detail"] = status.get("detail") or status.get("error")
        except VoiceStudioError:
            pass
        try:
            engines = await self._get_json("/engines/tts")
            out["engine"] = engines.get("active")
            out["engine_model"] = engines.get("active_model")
        except VoiceStudioError:
            pass
        return out

    async def voices(self) -> list[dict]:
        """Saved VoiceStudio voice profiles.

        The OpenAI alias voices (alloy, nova, ...) all map to the engine's default
        voice, so they are collapsed into one "default" entry.
        """
        data = await self._get_json("/v1/audio/voices")
        voices = [{"id": "", "name": "Engine default voice", "type": "default", "language": None}]
        for v in data.get("voices", []):
            if v.get("type") == "profile" and v.get("voice_id"):
                voices.append(
                    {
                        "id": v["voice_id"],
                        "name": v.get("name") or v["voice_id"],
                        "type": "profile",
                        "language": v.get("language"),
                    }
                )
        return voices

    async def ensure_ready(self) -> None:
        """Fail fast while VoiceStudio is still loading/downloading its model."""
        try:
            status = await self.model_status()
        except VoiceStudioError as exc:
            if exc.status == 503:
                raise
            return  # endpoint missing on this version: let synthesis decide
        state = (status.get("status") or "").lower()
        if status.get("loading") or state in {"loading", "downloading"}:
            detail = status.get("detail") or "loading model"
            raise VoiceStudioError(f"VoiceStudio is not ready yet: {detail}", status=503)
        if state == "error" and status.get("error"):
            raise VoiceStudioError(f"VoiceStudio model error: {status['error']}", status=503)

    async def stream_tts(self, payload: dict) -> AsyncIterator[StreamStart | AudioChunk]:
        """Stream one utterance from ``/ws/tts``.

        Yields exactly one :class:`StreamStart` before any :class:`AudioChunk`.
        Closing the generator (e.g. because the phone cancelled) closes the
        VoiceStudio socket, which stops synthesis after the current sentence.
        """
        url = self.base_url + "/ws/tts"
        try:
            ws = await asyncio.wait_for(
                self.session.ws_connect(
                    url,
                    timeout=aiohttp.ClientWSTimeout(ws_close=2.0),
                    max_msg_size=0,
                ),
                timeout=self.connect_timeout_s,
            )
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
            raise VoiceStudioError(
                f"VoiceStudio is not reachable at {self.base_url} ({exc.__class__.__name__})",
                status=503,
            ) from exc

        try:
            await ws.send_json(payload)
            started = False
            while True:
                wait = self.chunk_timeout_s if started else self.first_audio_timeout_s
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=wait)
                except asyncio.TimeoutError as exc:
                    raise VoiceStudioError(
                        f"VoiceStudio produced no audio for {wait:.0f}s", status=504
                    ) from exc

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not started:
                        started = True
                        yield StreamStart(DEFAULT_SAMPLE_RATE)
                    if msg.data:
                        yield AudioChunk(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        frame = json.loads(msg.data)
                    except ValueError:
                        continue
                    kind = frame.get("type")
                    if kind == "start" and not started:
                        started = True
                        yield StreamStart(
                            int(frame.get("sample_rate") or DEFAULT_SAMPLE_RATE),
                            frame.get("engine"),
                        )
                    elif kind == "done":
                        if not started:
                            yield StreamStart(
                                int(frame.get("sample_rate") or DEFAULT_SAMPLE_RATE),
                                frame.get("engine"),
                            )
                        return
                    elif kind == "error":
                        raise VoiceStudioError(frame.get("detail") or "synthesis failed")
                    else:
                        logger.debug("VoiceStudio frame: %s", frame)
                else:
                    raise VoiceStudioError(
                        f"VoiceStudio closed the stream unexpectedly ({msg.type.name})"
                    )
        finally:
            await ws.close()
