import sqlite3
import struct
from pathlib import Path

from elvenreader_server.audio import wav_bytes
from elvenreader_server.text import split_for_synthesis, trailing_pause_ms
from elvenreader_server.voices import VoiceStore

WAV = wav_bytes(b"\x10\x00" * 2400, 24000)


def test_add_get_list_delete(tmp_path):
    store = VoiceStore(tmp_path)
    v = store.add("Narrator", WAV, "Reference words.", language="English", seed=7)
    assert store.get(v.id).name == "Narrator"
    assert store.get(v.id).wav_path.read_bytes() == WAV
    assert [x.id for x in store.list()] == [v.id]
    assert store.default().id == v.id
    assert store.get("../etc") is None
    assert store.delete(v.id) and store.get(v.id) is None


def test_fingerprint_changes_with_transcript_or_audio(tmp_path):
    store = VoiceStore(tmp_path)
    v = store.add("A", WAV, "one", voice_id="a")
    fp = v.fingerprint
    assert store.update("a", ref_text="two").fingerprint != fp
    store.delete("a")
    other = store.add("A", wav_bytes(b"\x20\x00" * 2400, 24000), "one", voice_id="a")
    assert other.fingerprint != fp


def test_rejects_non_wav(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        VoiceStore(tmp_path).add("x", b"not a wav", "t")


def test_import_voicestudio_profiles_read_only(tmp_path):
    vs_voices = tmp_path / "vs"
    vs_voices.mkdir()
    (vs_voices / "demo0001.wav").write_bytes(WAV)
    (vs_voices / "d1.wav").write_bytes(WAV)
    db = tmp_path / "omnivoice.db"
    con = sqlite3.connect(db)
    con.execute(
        "create table voice_profiles (id, name, ref_audio_path, ref_text, instruct, language, "
        "locked_audio_path, seed, is_locked, description)"
    )
    con.executemany(
        "insert into voice_profiles values (?,?,?,?,?,?,?,?,?,?)",
        [
            ("demo0001", "Demo", "demo0001.wav", "Hey there.", "", "English", "", None, 0, "warm"),
            ("d1", "Designed", "d1.wav", "The valley.", "male, low pitch", "English", "", 42, 0, ""),
            ("noref", "No ref", "", "", "female", "English", "", None, 0, ""),
        ],
    )
    con.commit()
    con.close()
    before = db.read_bytes()
    store = VoiceStore(tmp_path / "store")
    imported = store.import_omnivoice(db, vs_voices)
    assert sorted(v.id for v in imported) == ["d1", "demo0001"]
    d1 = store.get("d1")
    assert (d1.seed, d1.instruct, d1.ref_text, d1.source) == (42, "male, low pitch", "The valley.", "voicestudio")
    assert store.import_omnivoice(db, vs_voices) == []  # idempotent
    assert db.read_bytes() == before


def test_wav_header():
    data = wav_bytes(b"\x00\x00" * 10, 22050)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack("<I", data[24:28])[0] == 22050
    assert struct.unpack("<I", data[40:44])[0] == 20


def test_split_for_synthesis_keeps_short_text_and_splits_long():
    assert split_for_synthesis("Short one.") == ["Short one."]
    sentences = " ".join(f"Sentence number {i} is here." for i in range(30))
    pieces = split_for_synthesis(sentences, limit=100)
    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces) == sentences
    long_clause = "word " * 80
    pieces = split_for_synthesis(long_clause.strip(), limit=60)
    assert all(len(p) <= 60 for p in pieces)
    assert " ".join(pieces) == long_clause.strip()


def test_trailing_pause():
    assert trailing_pause_ms("Done.", 300) == 300
    assert trailing_pause_ms("He said \u201cstop!\u201d", 300) == 300
    assert trailing_pause_ms("A clause,", 300) == 100
    assert trailing_pause_ms("no end", 300) == 0


def test_an_edited_voice_reaches_the_engines_under_a_new_path(tmp_path):
    """Engines cache what they derive from a reference by its path: new audio must come with a new path."""
    store = VoiceStore(tmp_path)
    v = store.add("Bishop", wav_bytes(b"\x10\x00" * 2400, 24000), "Hello.", voice_id="bishop")
    first = v.worker_spec()["ref_audio"]
    assert first != str(v.wav_path) and Path(first).read_bytes() == v.wav_path.read_bytes()
    assert store.get("bishop").worker_spec()["ref_audio"] == first  # same audio, same path
    store.replace_audio("bishop", wav_bytes(b"\x20\x00" * 2400, 24000), "Hello.")  # same file name and text
    second = store.get("bishop").worker_spec()["ref_audio"]
    assert second != first and Path(second).read_bytes() == store.get("bishop").wav_path.read_bytes()
    assert [x.id for x in store.list()] == ["bishop"]  # the copies aren't voices

def test_clean_text_puts_a_space_after_tight_ellipses():
    from elvenreader_server.text import clean_text

    assert clean_text("I\u2026I didn\u2019t") == "I\u2026 I didn\u2019t"
    assert clean_text("Um...yes. H\u2025huh? Wait......what") == "Um... yes. H\u2025 huh? Wait...... what"
    assert clean_text("\u201c\u2026I see.\u201d") == "\u201c\u2026 I see.\u201d"
    # Left alone: already spaced, before punctuation or a closing quote, decimals.
    for s in ("Well\u2026 fine", "Um\u2026,", "\u201cBut\u2026\u201d", "So...?", "Pi is 3.14", "Wait\u2026 \u2026no"):
        assert clean_text(s) == s.replace("\u2026no", "\u2026 no")