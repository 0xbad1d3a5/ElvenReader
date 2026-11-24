"""Append-only zip files, compacted now and then.

A commit writes the new entries after the end of the file, then a fresh central directory listing only the live
entries, a zip64 end record and the end record. The bytes before never change: a replaced or deleted entry, and
the previous central directory, stay behind as dead bytes that no directory points at. Any zip reader that uses
the central directory (Python's zipfile, libzip, zip.js, Android's ZipFile) sees exactly the live entries;
readers that walk the file front to back (Java's ZipInputStream) are not supported.

The end record's comment is JSON: {"generation": n, "dead": bytes, ...}. Compacting copies the live entries
byte for byte into a new file, which replaces the old one, and bumps the generation, so a copy that only
fetches new bytes knows to start over.

A crash mid-commit leaves an incomplete tail after the last complete end record; opening the file finds that
record (searching backwards) and the next commit cuts the tail off.
"""
from __future__ import annotations

import json
import os
import struct
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

_LOCAL = struct.Struct("<IHHHHHIIIHH")  # 30 bytes
_CENTRAL = struct.Struct("<IHHHHHHIIIHHHHHII")  # 46 bytes
_EOCD = struct.Struct("<IHHHHIIH")  # 22 bytes
_EOCD64 = struct.Struct("<IQHHIIQQQQ")  # 56 bytes
_LOC64 = struct.Struct("<IIQI")  # 20 bytes
_SIG_LOCAL, _SIG_CENTRAL, _SIG_EOCD = 0x04034B50, 0x02014B50, 0x06054B50
_SIG_EOCD64, _SIG_LOC64 = 0x06064B50, 0x07064B50
_EOCD_MAGIC = b"PK\x05\x06"
STORED, DEFLATED = 0, 8
_UTF8 = 0x0800
_MAX16, _MAX32 = 0xFFFF, 0xFFFFFFFF
_MADE_BY = 45  # zip spec 4.5 (zip64), MS-DOS attributes


class ZipFormatError(Exception):
    pass


@dataclass(frozen=True)
class Member:
    name: str
    offset: int  # of the local header
    method: int
    crc: int
    csize: int
    size: int
    dostime: int
    dosdate: int
    header_size: int = 0  # local header + name + extra; 0 = not read yet

    @property
    def record_size(self) -> int:
        return (self.header_size or _LOCAL.size + len(self.name.encode("utf-8"))) + self.csize


def _dos_stamp(t: float) -> tuple[int, int]:
    lt = time.localtime(t)
    return (lt.tm_hour << 11) | (lt.tm_min << 5) | (lt.tm_sec // 2), \
        (max(lt.tm_year, 1980) - 1980) << 9 | (lt.tm_mon << 5) | lt.tm_mday


def _replace_file(src: Path, dst: Path, tries: int = 10) -> None:
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:  # Windows: someone has the file open
            if i == tries - 1:
                raise
            time.sleep(0.2)


class AppendZip:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.members: dict[str, Member] = {}
        self.meta: dict = {}
        self.size = 0  # end of the last complete commit
        self.cd_offset = 0

    @classmethod
    def open(cls, path: Path) -> "AppendZip":
        """The file's live entries; an empty archive when the file doesn't exist yet."""
        z = cls(path)
        if z.path.exists() and z.path.stat().st_size:
            z._load()
        return z

    # ------------------------------------------------------------ reading

    @property
    def generation(self) -> int:
        return int(self.meta.get("generation") or 1)

    @property
    def dead(self) -> int:
        return int(self.meta.get("dead") or 0)

    def __contains__(self, name: str) -> bool:
        return name in self.members

    def names(self) -> list[str]:
        return list(self.members)

    def member(self, name: str) -> Member | None:
        return self.members.get(name)

    def same(self, name: str, data: bytes) -> bool:
        """True when the entry exists with exactly these bytes (by CRC-32 and length)."""
        m = self.members.get(name)
        return m is not None and m.size == len(data) and m.crc == zlib.crc32(data)

    def read(self, name: str) -> bytes:
        m = self.members.get(name)
        if m is None:
            raise KeyError(name)
        with open(self.path, "rb") as f:
            return self._read(f, m)

    def read_many(self, names: list[str]) -> dict[str, bytes]:
        with open(self.path, "rb") as f:
            return {n: self._read(f, self.members[n]) for n in names if n in self.members}

    def _read(self, f, m: Member) -> bytes:
        f.seek(m.offset)
        head = f.read(_LOCAL.size)
        if len(head) < _LOCAL.size or _LOCAL.unpack(head)[0] != _SIG_LOCAL:
            raise ZipFormatError(f"{m.name}: bad local header")
        nlen, elen = _LOCAL.unpack(head)[9:11]
        f.seek(m.offset + _LOCAL.size + nlen + elen)
        raw = f.read(m.csize)
        if m.method == DEFLATED:
            data = zlib.decompress(raw, -15)
        elif m.method == STORED:
            data = raw
        else:
            raise ZipFormatError(f"{m.name}: compression method {m.method} is not supported")
        if zlib.crc32(data) != m.crc or len(data) != m.size:
            raise ZipFormatError(f"{m.name}: damaged (CRC or size mismatch)")
        return data

    def _load(self) -> None:
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            end = self._end_at_eof(f, n) or self._scan_back(f, n)
            if end is None:
                raise ZipFormatError(f"{self.path.name}: not a zip file, or no complete central directory")
            self._read_directory(f, end)

    def _parse_end(self, f, pos: int, file_size: int) -> dict | None:
        """The end record at `pos`, if it is one and the central directory it points at is where it says."""
        f.seek(pos)
        raw = f.read(_EOCD.size)
        if len(raw) < _EOCD.size:
            return None
        sig, disk, cd_disk, _, count, cd_size, cd_offset, clen = _EOCD.unpack(raw)
        if sig != _SIG_EOCD or disk or cd_disk or pos + _EOCD.size + clen > file_size:
            return None
        comment = f.read(clen)
        cd_end = pos
        if pos >= _LOC64.size:
            f.seek(pos - _LOC64.size)
            sig64, _, z64_pos, _ = _LOC64.unpack(f.read(_LOC64.size))
            if sig64 == _SIG_LOC64:
                if z64_pos + _EOCD64.size > pos - _LOC64.size:
                    return None
                f.seek(z64_pos)
                rec = _EOCD64.unpack(f.read(_EOCD64.size))
                if rec[0] != _SIG_EOCD64:
                    return None
                count, cd_size, cd_offset = rec[7], rec[8], rec[9]
                cd_end = z64_pos
        if cd_offset + cd_size != cd_end:
            return None
        if count:
            f.seek(cd_offset)
            if f.read(4) != struct.pack("<I", _SIG_CENTRAL):
                return None
        return {"count": count, "cd_size": cd_size, "cd_offset": cd_offset, "comment": comment,
                "end": pos + _EOCD.size + clen}

    def _end_at_eof(self, f, n: int) -> dict | None:
        """The usual case: an end record whose comment ends the file."""
        span = min(n, _EOCD.size + _MAX16)
        f.seek(n - span)
        tail = f.read(span)
        i = len(tail)
        while (i := tail.rfind(_EOCD_MAGIC, 0, i)) >= 0:
            found = self._parse_end(f, n - span + i, n)
            if found and found["end"] == n:
                return found
        return None

    def _scan_back(self, f, n: int, chunk: int = 1 << 20) -> dict | None:
        """After a crash mid-commit: the last complete end record anywhere in the file."""
        hi = n
        while hi > 0:
            lo = max(0, hi - chunk)
            f.seek(lo)
            buf = f.read(min(hi + 3, n) - lo)  # overlap so a signature across chunks is found
            i = len(buf)
            while (i := buf.rfind(_EOCD_MAGIC, 0, i)) >= 0:
                if lo + i < hi and (found := self._parse_end(f, lo + i, n)):
                    return found
            hi = lo
        return None

    def _read_directory(self, f, end: dict) -> None:
        f.seek(end["cd_offset"])
        cd = f.read(end["cd_size"])
        members: dict[str, Member] = {}
        p = 0
        for _ in range(end["count"]):
            (sig, _, _, flags, method, dostime, dosdate, crc, csize, size, nlen, elen, clen, _, _, _,
             offset) = _CENTRAL.unpack_from(cd, p)
            if sig != _SIG_CENTRAL:
                raise ZipFormatError("bad central directory")
            p += _CENTRAL.size
            raw_name = cd[p:p + nlen]
            name = raw_name.decode("utf-8" if flags & _UTF8 else "cp437")
            extra = cd[p + nlen:p + nlen + elen]
            p += nlen + elen + clen
            if _MAX32 in (size, csize, offset):
                size, csize, offset = self._zip64_fields(extra, size, csize, offset)
            members[name] = Member(name, offset, method, crc, csize, size, dostime, dosdate)
        self.members = members
        self.cd_offset = end["cd_offset"]
        self.size = end["end"]
        try:
            meta = json.loads(end["comment"].decode("utf-8")) if end["comment"] else {}
        except ValueError:
            meta = {}
        self.meta = meta if isinstance(meta, dict) else {}

    @staticmethod
    def _zip64_fields(extra: bytes, size: int, csize: int, offset: int) -> tuple[int, int, int]:
        p = 0
        while p + 4 <= len(extra):
            tag, ln = struct.unpack_from("<HH", extra, p)
            if tag == 1:
                q = p + 4
                if size == _MAX32:
                    size = struct.unpack_from("<Q", extra, q)[0]
                    q += 8
                if csize == _MAX32:
                    csize = struct.unpack_from("<Q", extra, q)[0]
                    q += 8
                if offset == _MAX32:
                    offset = struct.unpack_from("<Q", extra, q)[0]
                return size, csize, offset
            p += 4 + ln
        raise ZipFormatError("zip64 sizes missing")

    # ------------------------------------------------------------ writing

    def commit(self, writes: dict[str, tuple[bytes, bool]], delete=(), meta: dict | None = None) -> None:
        """Append `writes` ({name: (data, compress)}; an existing name is replaced) and drop `delete`, in one
        step: until the new end record is on disk, readers and a crash see the archive as it was."""
        now = time.time()
        dostime, dosdate = _dos_stamp(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        members = dict(self.members)
        dead = self.dead + (self.size - self.cd_offset if self.size else 0)
        for name in delete:
            if name not in writes and (old := members.pop(name, None)):
                dead += old.record_size
        order = sorted(writes, key=lambda n: n != "mimetype") if not self.members else list(writes)
        with open(self.path, "r+b" if self.path.exists() else "w+b") as f:
            f.seek(self.size)
            f.truncate()
            for name in order:
                data, compress = writes[name]
                raw, method = data, STORED
                if compress and data:
                    co = zlib.compressobj(6, zlib.DEFLATED, -15)
                    packed = co.compress(data) + co.flush()
                    if len(packed) < len(data):
                        raw, method = packed, DEFLATED
                if len(data) >= _MAX32 or len(raw) >= _MAX32:
                    raise ValueError(f"{name}: entries of 4 GB or more are not supported")
                nb = name.encode("utf-8")
                offset = f.tell()
                f.write(_LOCAL.pack(_SIG_LOCAL, 20, _UTF8, method, dostime, dosdate, zlib.crc32(data), len(raw),
                                    len(data), len(nb), 0) + nb)
                f.write(raw)
                if (old := members.pop(name, None)) is not None:
                    dead += old.record_size
                members[name] = Member(name, offset, method, zlib.crc32(data), len(raw), len(data), dostime,
                                       dosdate, _LOCAL.size + len(nb))
            f.flush()
            os.fsync(f.fileno())
            new_meta = {**self.meta, **(meta or {}), "generation": self.generation, "dead": dead,
                        "updated": round(now)}
            cd_offset = self._write_directory(f, members, new_meta)
            f.flush()
            os.fsync(f.fileno())
            size = f.tell()
        self.members, self.meta, self.cd_offset, self.size = members, new_meta, cd_offset, size

    def compact(self) -> bool:
        """Rewrite the file with only its live entries. False when the file couldn't be replaced (Windows: it
        is open elsewhere); the old file is then left as it was."""
        tmp = self.path.with_name(self.path.name + ".compact")
        order = sorted(self.members.values(), key=lambda m: m.name != "mimetype")
        members: dict[str, Member] = {}
        with open(self.path, "rb") as src, open(tmp, "wb") as dst:
            for m in order:
                src.seek(m.offset)
                head = src.read(_LOCAL.size)
                nlen, elen = _LOCAL.unpack(head)[9:11]
                header_size = _LOCAL.size + nlen + elen
                offset = dst.tell()
                dst.write(head)
                remaining = header_size - _LOCAL.size + m.csize
                while remaining:
                    block = src.read(min(remaining, 1 << 20))
                    if not block:
                        raise ZipFormatError(f"{m.name}: truncated")
                    dst.write(block)
                    remaining -= len(block)
                members[m.name] = replace(m, offset=offset, header_size=header_size)
            meta = {**self.meta, "generation": self.generation + 1, "dead": 0, "updated": round(time.time())}
            cd_offset = self._write_directory(dst, members, meta)
            dst.flush()
            os.fsync(dst.fileno())
            size = dst.tell()
        try:
            _replace_file(tmp, self.path)
        except PermissionError:
            tmp.unlink(missing_ok=True)
            return False
        self.members, self.meta, self.cd_offset, self.size = members, meta, cd_offset, size
        return True

    @staticmethod
    def _write_directory(f, members: dict[str, Member], meta: dict) -> int:
        cd = bytearray()
        for m in members.values():
            nb = m.name.encode("utf-8")
            extra, offset = b"", m.offset
            if offset >= _MAX32:
                extra, offset = struct.pack("<HHQ", 1, 8, m.offset), _MAX32
            cd += _CENTRAL.pack(_SIG_CENTRAL, _MADE_BY, 45 if extra else 20, _UTF8, m.method, m.dostime, m.dosdate,
                                m.crc, m.csize, m.size, len(nb), len(extra), 0, 0, 0, 0, offset) + nb + extra
        cd_offset = f.tell()
        f.write(cd)
        z64 = f.tell()
        count = len(members)
        f.write(_EOCD64.pack(_SIG_EOCD64, _EOCD64.size - 12, _MADE_BY, 45, 0, 0, count, count, len(cd), cd_offset))
        f.write(_LOC64.pack(_SIG_LOC64, 0, z64, 1))
        comment = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        if len(comment) > _MAX16:
            raise ValueError("zip comment too long")
        f.write(_EOCD.pack(_SIG_EOCD, 0, 0, min(count, _MAX16), min(count, _MAX16), min(len(cd), _MAX32),
                           min(cd_offset, _MAX32), len(comment)) + comment)
        return cd_offset
