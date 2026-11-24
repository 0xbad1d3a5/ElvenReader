import shutil
import sqlite3
import struct
from pathlib import Path

import pytest

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

def test_clean_text_spaces_an_ellipsis_before_quotes_and_brackets():
    from elvenreader_server.text import clean_text

    assert clean_text("\u201c\u2026\u2018Boss,\u2019 she says\u201d") == "\u201c\u2026 \u2018Boss,\u2019 she says\u201d"
    assert clean_text("...\u2019Cause, I mean") == "... \u2019Cause, I mean"
    assert clean_text("Just\u2026(whispers) don\u2019t") == "Just\u2026 (whispers) don\u2019t"
    assert clean_text("\u201cUrk\u2026\u201d Miss") == "\u201cUrk\u2026\u201d Miss"  # a closing quote stays put


def test_clean_text_collapses_doubled_quotation_marks():
    from elvenreader_server.text import clean_text

    assert clean_text("\u201c\u201cWh\u2026what?!\u201d\u201d") == "\u201cWh\u2026 what?!\u201d"
    assert clean_text("\u2018\u2018No!\u2019\u2019 and \"\"yes\"\"") == "\u2018No!\u2019 and \"yes\""
    assert clean_text("\u201cHe said \u2018hi.\u2019\u201d") == "\u201cHe said \u2018hi.\u2019\u201d"  # nested: different marks


def test_a_clause_left_open_is_closed_for_the_speech_model():
    from elvenreader_server.text import close_clause

    assert close_clause("When she was his page, Ronie always said,") == "When she was his page, Ronie always said:"
    assert close_clause("\u201cI\u2019ll do it,\u201d") == "\u201cI\u2019ll do it.\u201d"  # a quotation: a period
    assert close_clause("he said; ") == "he said:"
    for s in ("It was late.", "\u201cWait\u2026\u201d", "Then\u2014", "Who?", "She said"):
        assert close_clause(s) == s


def test_stutters_are_spelled_as_they_sound():
    from elvenreader_server.text import engine_text, spell_stutters
    from elvenreader_server.engines.registry import ENGINES

    assert spell_stutters("\u201cR-right, I\u2019ll go.\u201d") == "\u201cRuh-right, I\u2019ll go.\u201d"
    assert spell_stutters("Wh-what? S-s-sorry! Y-yeah. Th-that\u2019s it") == \
        "Whuh-what? Suh-suh-sorry! Yuh-yeah. Thuh-that\u2019s it"
    # Left alone: fragments with a vowel, fragments that aren't the next word's start, ordinary hyphens.
    for s in ("I-I don\u2019t know", "Wha-what?", "a T-shirt", "an X-ray", "well-known", "B-52"):
        assert spell_stutters(s) == s
    assert engine_text(ENGINES["voxcpm"], "B-but he said,") == "Buh-but he said:"
    assert engine_text(ENGINES["omnivoice"], "B-but he said,") == "B-but he said,"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_phone_recording_with_its_index_at_the_end_is_read(tmp_path):
    """Phones' .m4a recordings keep their index (moov) after the audio: ffmpeg only finds it by seeking."""
    import subprocess

    from elvenreader_server.voice_edit import to_wav

    clip = tmp_path / "recording.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=60", "-ac", "2",
                    "-ar", "48000", "-c:a", "aac", "-b:a", "256k", str(clip)], check=True)
    wav = to_wav(clip.read_bytes(), 1.0, 11.0)
    assert wav[:4] == b"RIFF" and abs((len(wav) - 44) / 2 / 24000 - 10.0) < 0.1

