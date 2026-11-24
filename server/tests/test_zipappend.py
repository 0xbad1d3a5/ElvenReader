import json
import zipfile

import pytest

from elvenreader_server import zipappend
from elvenreader_server.zipappend import AppendZip, ZipFormatError


def entries(path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return {n: z.read(n) for n in z.namelist()}


def test_commits_make_a_zip_any_reader_opens(tmp_path):
    p = tmp_path / "b.zip"
    z = AppendZip.open(p)
    z.commit({"a.json": (b'{"x": 1}' * 50, True), "mimetype": (b"application/x-test", False),
              "pic.jpg": (b"\xff\xd8 not really", False)}, meta={"format": "test"})
    with zipfile.ZipFile(p) as zf:
        first = zf.infolist()[0]
        assert first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED and first.header_offset == 0
        assert zf.getinfo("a.json").compress_type == zipfile.ZIP_DEFLATED
        assert json.loads(zf.comment)["format"] == "test"
    assert entries(p)["a.json"] == b'{"x": 1}' * 50
    again = AppendZip.open(p)
    assert again.read("pic.jpg") == b"\xff\xd8 not really" and again.meta["format"] == "test"
    assert again.same("a.json", b'{"x": 1}' * 50) and not again.same("a.json", b"other")


def test_appending_never_rewrites_earlier_bytes(tmp_path):
    p = tmp_path / "b.zip"
    z = AppendZip.open(p)
    z.commit({"keep": (b"k" * 1000, False), "old": (b"o" * 1000, False), "gone": (b"g" * 500, False)})
    before = p.read_bytes()
    z.commit({"old": (b"new", False), "added": (b"a", False)}, delete=["gone"])
    after = p.read_bytes()
    assert after[:z.cd_offset][:len(before)] == before  # the old file is a prefix of the new one
    assert entries(p) == {"keep": b"k" * 1000, "old": b"new", "added": b"a"}
    assert z.dead >= 1000 + 500  # the replaced and the deleted entry, plus the first directory
    assert AppendZip.open(p).dead == z.dead and AppendZip.open(p).generation == 1


def test_an_interrupted_commit_is_cut_off_and_the_archive_stays_readable(tmp_path):
    p = tmp_path / "b.zip"
    z = AppendZip.open(p)
    z.commit({"a": (b"first", False)})
    size = p.stat().st_size
    with open(p, "ab") as f:  # a crash after writing an entry, before the new directory
        f.write(b"PK\x03\x04" + b"\x00" * 26 + b"half an entry" * 20000)
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(p)
    z = AppendZip.open(p)
    assert z.names() == ["a"] and z.size == size and z.read("a") == b"first"
    z.commit({"b": (b"second", False)})
    assert entries(p) == {"a": b"first", "b": b"second"}


def test_compacting_keeps_the_live_entries_and_starts_a_new_generation(tmp_path):
    p = tmp_path / "b.zip"
    z = AppendZip.open(p)
    z.commit({"mimetype": (b"x/y", False), "a": (b"1" * 5000, True)})
    for i in range(5):
        z.commit({"a": (str(i).encode() * 5000, True), f"clip{i}": (bytes(range(256)) * 20, False)})
    z.commit({}, delete=["clip0"])
    big = p.stat().st_size
    live = entries(p)
    assert z.compact()
    assert p.stat().st_size < big and z.dead == 0 and z.generation == 2
    assert entries(p) == live
    with zipfile.ZipFile(p) as zf:
        assert zf.infolist()[0].filename == "mimetype" and zf.infolist()[0].header_offset == 0
    z.commit({"after": (b"!", False)})
    assert entries(p) == {**live, "after": b"!"} and AppendZip.open(p).generation == 2


def test_offsets_beyond_32_bits_use_zip64_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(zipappend, "_MAX32", 300)  # pretend 300 bytes is 4 GB
    p = tmp_path / "b.zip"
    z = AppendZip.open(p)
    for i in range(8):
        z.commit({f"e{i}": (bytes([i]) * 60, False)})
    again = AppendZip.open(p)
    assert again.member("e7").offset > 300
    assert all(again.read(f"e{i}") == bytes([i]) * 60 for i in range(8))


def test_not_a_zip(tmp_path):
    p = tmp_path / "b.zip"
    p.write_bytes(b"hello" * 100)
    with pytest.raises(ZipFormatError):
        AppendZip.open(p)


def test_reads_ordinary_zips(tmp_path):
    p = tmp_path / "b.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("x.txt", "hello " * 100)
        zf.writestr("y.bin", b"\x00\x01")
    z = AppendZip.open(p)
    assert z.read("x.txt") == b"hello " * 100 and z.read("y.bin") == b"\x00\x01"
    z.commit({"z": (b"added", False)})
    assert entries(p) == {"x.txt": b"hello " * 100, "y.bin": b"\x00\x01", "z": b"added"}
