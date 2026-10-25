"""Framing used between the server and engine worker processes (stdlib only).

Each frame is a 4-byte big-endian length followed by the payload. The first
payload byte is the kind: ``J`` (UTF-8 JSON object) or ``A`` (audio: 2-byte
request-id length, request id, then PCM16LE mono samples).
"""
from __future__ import annotations

import json
import struct

MAX_FRAME = 64 * 1024 * 1024


def encode_json(obj: dict) -> bytes:
    body = b"J" + json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack("!I", len(body)) + body


def encode_audio(request_id: str, pcm: bytes) -> bytes:
    rid = request_id.encode("utf-8")
    body = b"A" + struct.pack("!H", len(rid)) + rid + pcm
    return struct.pack("!I", len(body)) + body


def decode(body: bytes) -> tuple[str, object]:
    """Returns ("json", dict) or ("audio", (request_id, pcm))."""
    kind = body[:1]
    if kind == b"J":
        return "json", json.loads(body[1:].decode("utf-8"))
    if kind == b"A":
        (n,) = struct.unpack("!H", body[1:3])
        return "audio", (body[3 : 3 + n].decode("utf-8"), body[3 + n :])
    raise ValueError(f"unknown frame kind {kind!r}")


def read_frame_sync(stream) -> bytes | None:
    header = _read_exact(stream, 4)
    if header is None:
        return None
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME:
        raise IOError(f"frame too large: {n}")
    body = _read_exact(stream, n)
    if body is None:
        raise IOError("short read")
    return body


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf) if len(buf) == n else None
        buf.extend(chunk)
    return bytes(buf)


async def read_frame_async(reader) -> bytes | None:
    import asyncio

    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME:
        raise IOError(f"frame too large: {n}")
    try:
        return await reader.readexactly(n)
    except asyncio.IncompleteReadError:
        return None
