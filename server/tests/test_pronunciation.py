import pytest

from androidreader_server.cache import AudioCache, cache_key
from androidreader_server.pronunciation import (
    PHONEME_FORMATS,
    Entry,
    PronunciationStore,
    apply,
    effective_entries,
    entries_version,
    normalize_phonemes,
    speakable_respelling,
)


def E(term, respelling, **kw):
    return Entry(term=term, respelling=respelling, **kw)


HERMIONE = E("Hermione", "her-MY-oh-nee")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hermione smiled.", "her-my-oh-nee smiled."),
        ("hermione smiled.", "her-my-oh-nee smiled."),
        ("It was Hermione's wand.", "It was her-my-oh-nee's wand."),
        ("It was Hermione\u2019s wand.", "It was her-my-oh-nee\u2019s wand."),
        ("Two Hermiones met.", "Two her-my-oh-nees met."),
        ("\u201cHermione!\u201d", "\u201cher-my-oh-nee!\u201d"),
        ("Hermioneville is a town.", "Hermioneville is a town."),  # not a whole word
        ("SuperHermione", "SuperHermione"),
    ],
)
def test_whole_word_possessive_and_plural(text, expected):
    assert apply(text, [HERMIONE]) == expected


def test_case_sensitive_entry_only_matches_exact_case():
    entry = E("Reading", "REDD-ing", case_sensitive=True)  # the town, not the verb
    assert apply("We drove to Reading.", [entry]) == "We drove to redd-ing."
    assert apply("She was reading.", [entry]) == "She was reading."


def test_multi_word_names_beat_single_words_and_span_whitespace():
    entries = [E("Sirius", "SEER-ee-us"), E("Sirius Black", "SEER-ee-us BLACK-k")]
    assert apply("Sirius  Black laughed. Sirius left.", entries) == "seer-ee-us black-k laughed. seer-ee-us left."


def test_phonemes_used_when_engine_supports_them_else_respelling():
    entry = E("Hermione", "her-MY-oh-nee", phonemes="hh er0 m ay1 ah0 n iy0")
    assert entry.phonemes == "hh er0 m ay1 ah0 n iy0"  # raw until stored
    entry.phonemes = normalize_phonemes(entry.phonemes)
    assert apply("Hermione ran.", [entry], PHONEME_FORMATS["cmu_brackets"]) == "[HH ER0 M AY1 AH0 N IY0] ran."
    assert apply("Hermione ran.", [entry], PHONEME_FORMATS["cmu_annotated"]) == "<Hermione|HH ER0 M AY1 AH0 N IY0> ran."
    assert apply("Hermione ran.", [entry]) == "her-my-oh-nee ran."
    # A possessive can't be expressed in phonemes, so the respelling is used.
    assert apply("Hermione's", [entry], PHONEME_FORMATS["cmu_brackets"]) == "her-my-oh-nee's"


def test_normalize_phonemes_rejects_non_arpabet():
    assert normalize_phonemes("[b ey1 s]") == "B EY1 S"
    assert normalize_phonemes("hɜːˈmaɪəni") == ""


def test_speakable_respelling_lowercases_stress_caps_only():
    assert speakable_respelling("her-MY-oh-nee") == "her-my-oh-nee"
    assert speakable_respelling("Shiv-awn") == "shiv-awn"
    assert speakable_respelling("Siobhan Shivawn") == "Siobhan Shivawn"


def test_book_entries_override_global_for_same_term():
    entries = [
        E("Hermione", "her-MY-oh-nee"),
        E("hermione", "HER-mee-own", book="b1"),
        E("Gandalf", "GAN-dalf"),
    ]
    for_b1 = {e.term.casefold(): e.respelling for e in effective_entries(entries, "b1")}
    assert for_b1 == {"hermione": "HER-mee-own", "gandalf": "GAN-dalf"}
    for_other = {e.term.casefold(): e.respelling for e in effective_entries(entries, "b2")}
    assert for_other["hermione"] == "her-MY-oh-nee"
    assert len(effective_entries(entries, None)) == 2  # no book: global entries only


def test_store_crud_and_listing(tmp_path):
    store = PronunciationStore(tmp_path / "p.json")
    g = store.add("Hermione", "her-MY-oh-nee")
    b = store.add("Hermione", "HER-mee-own", book="b1")
    store.add("Hermione", "her-MY-oh-nee-2")  # same scope + term replaces
    assert len(store.all()) == 2
    listing = store.listing("b1")
    assert [e["respelling"] for e in listing["global"]] == ["her-MY-oh-nee-2"]
    assert [e["scope"] for e in listing["book"]] == ["book"]
    assert store.update(b.id, respelling="her-MEE-own").respelling == "her-MEE-own"
    assert store.delete(b.id) and not store.delete(b.id)
    reloaded = PronunciationStore(tmp_path / "p.json")
    assert [e.respelling for e in reloaded.all()] == ["her-MY-oh-nee-2"]
    assert g.id not in [e.id for e in reloaded.all()]  # replaced entry is gone
    with pytest.raises(ValueError):
        store.add("...", "x")
    with pytest.raises(ValueError):
        store.add("Name", "")


def test_editing_an_entry_changes_text_version_and_cache_key():
    entries = [E("Hermione", "her-MY-oh-nee")]
    text1 = apply("Hermione ran.", entries)
    v1 = entries_version(entries)
    key1 = cache_key(engine="omnivoice", voice="v", text=text1)
    entries[0].respelling = "her-MEE-oh-nee"
    text2 = apply("Hermione ran.", entries)
    assert text1 != text2
    assert entries_version(entries) != v1
    assert cache_key(engine="omnivoice", voice="v", text=text2) != key1
    # Unrelated sentences keep their key, so their cached audio stays valid.
    assert apply("Ron ran.", entries) == "Ron ran."


def test_audio_cache_roundtrip_and_prune(tmp_path):
    cache = AudioCache(tmp_path, max_bytes=3000)
    k1, k2 = cache_key(a=1), cache_key(a=2)
    assert cache.get(k1) is None
    cache.put(k1, 24000, b"\x01\x00" * 1000)
    assert cache.get(k1) == (24000, b"\x01\x00" * 1000)
    cache.put(k2, 16000, b"\x02\x00" * 1000)  # over the limit: oldest goes
    assert cache.get(k2) == (16000, b"\x02\x00" * 1000)
    assert cache.stats()["bytes"] <= 3000
