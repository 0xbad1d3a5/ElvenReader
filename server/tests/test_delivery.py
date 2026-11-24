from elvenreader_server import delivery as dl
from elvenreader_server import dramatize as dz
from elvenreader_server import sounds
from elvenreader_server.engines import voxcpm_styles as vs
from elvenreader_server.engines.registry import ENGINES as BUILTIN_ENGINES
from elvenreader_server.text import clean_text, engine_text


def test_approved_deliveries_keep_one_per_slot_in_order():
    assert dl.normalize(["shout", "very angry", "loud", "laugh", "sigh", "bogus", "angry"]) == "very angry, shout"
    assert dl.normalize(["afraid", "loud", "fast", "clear", "sigh"]) == "afraid, loud"  # two at most
    assert dl.normalize(["sigh", "fast", "thought"]) == "sigh, thought"  # situation, then sound, then manner
    assert dl.normalize("slow, fast, clear") == "slow"  # one pace/manner
    assert dl.normalize([]) is None and dl.normalize(None) is None
    assert "very angry" in dl.all_tags() and "very shout" not in dl.all_tags()


def test_free_text_deliveries_map_to_approved_tags():
    assert dl.from_text("shouting, urgent") == "firm, shout"
    assert dl.from_text("extremely furious, whispering") == "very angry, whisper"
    assert dl.from_text("a bit nervous") == "slightly afraid"
    assert dl.from_text("neutral") is None and dl.from_text("casual, informative") is None
    assert dl.from_text("concerned", known={"concerned": ["tender"]}) == "tender"
    assert dl.match("casual") is None and dl.match("calm") == []


def test_voxcpm_renders_approved_tags_with_the_words_it_follows():
    assert vs.render("very angry, shout, laugh") == ("very angry, speaking loudly", "[laugh-haha]")
    assert vs.render("slightly sad") == ("sad", "")  # 'slightly' measured as nothing
    assert vs.render("firm, hesitant") == ("", "")  # nothing it acts out: carries on from the voice's clip
    assert vs.render("husky, firm") == ("husky", "")  # a reader's own words pass through
    assert vs.render(None) == ("", "")


def test_sound_marks_are_made_from_the_named_words():
    text = "\u201cAhaha, that\u2019s fine. Gyaaah! Gyaaah!\u201d"
    marked = sounds.mark(text, [{"written": "Ahaha", "sound": "laugh"}, {"written": "Gyaaah!", "sound": "scream"},
                                {"written": "Hmph", "sound": "hmph"}, {"written": "fine", "sound": "bogus"}])
    assert marked == "\u201c{laugh|Ahaha}, that\u2019s fine. {scream|Gyaaah}! {scream|Gyaaah}!\u201d"
    assert sounds.plain(marked) == text and sounds.kinds(marked) == ["laugh", "scream", "scream"]
    assert sounds.mark(marked, [{"written": "Ahaha", "sound": "sigh"}]) == marked  # never inside a mark
    assert sounds.mark("\u201cHaha-ha\u201d", [{"written": "Haha", "sound": "laugh"}]) == "\u201cHaha-ha\u201d"


def test_marks_render_as_an_engines_tags_or_as_written():
    tags = {"laugh": "[laugh-haha]", "sigh": "[sigh-hah]"}
    assert sounds.render("{laugh|Ahaha}, that's fine.", tags) == "[laugh-haha] That's fine."
    assert sounds.render("\u201c{sigh|Haaah}... really?\u201d", tags) == "\u201c[sigh-hah] Really?\u201d"
    assert sounds.render("Well, {laugh|heh}, maybe.", tags) == "Well, [laugh-haha] maybe."  # mid-sentence
    assert sounds.render("{scream|Kyaaah}! What?!", tags) == "Kyaaah! What?!"  # no tag: as written
    assert sounds.render("{laugh|Ahaha}!", tags) == "Ahaha!"  # nothing but sounds: as written
    assert sounds.render("{laugh|Ahaha}, fine.", None) == "Ahaha, fine."


def test_stray_braces_are_not_marks():
    assert clean_text("a {weird} {laugh|Haha} b") == "a (weird) {laugh|Haha} b"
    assert clean_text("{bogus|x} y") == "(bogus|x) y"


def test_each_engine_gets_marks_in_its_own_form():
    vox = BUILTIN_ENGINES["voxcpm"]
    other = next(s for k, s in BUILTIN_ENGINES.items() if k != "voxcpm")
    text = "{laugh|Ahaha}, that's fine."
    assert engine_text(other, text) == "Ahaha, that's fine."
    for kind, tag in vox.sound_tags.items():
        assert engine_text(vox, "{%s|Hm}, fine." % kind) == f"{tag} Fine."


def test_spoken_form_combines_respelling_and_sounds():
    text = "\u201cAhaha, whaaaat?\u201d"
    say = dz.spoken_form(text, [{"written": "whaaaat", "said": "what"}], [{"written": "Ahaha", "sound": "laugh"}])
    assert say == "\u201c{laugh|Ahaha}, what?\u201d"
    assert dz.spoken_form(text, [], [{"written": "Ahaha", "sound": "laugh"}], said="\u201cAhaha, what?\u201d") == \
        "\u201c{laugh|Ahaha}, what?\u201d"  # builds on an earlier spoken form
    assert dz.spoken_form(text, [], []) == ""
    props = dz.chunk_schema(["1"])["properties"]["quotes"]["items"]
    assert "sounds" in props["required"] and props["properties"]["delivery"]["type"] == "array"
    assert props["properties"]["delivery"]["maxItems"] == 2


def test_a_sentence_ending_between_two_quotations_makes_two_turns():
    assert dz.interrupts(" he said, ")
    assert dz.interrupts(" said Mr. Akutsu, ")  # a title, not a sentence end
    assert not dz.interrupts(" Asuna whispered. Right behind her ear, Kirito said, ")
    assert not dz.interrupts(" he said. ") and not dz.interrupts(None)


def test_narration_naming_another_speaker_makes_two_turns():
    # Alice ch. 7: one number for both made every model give the Dormouse's line to the Hatter or March Hare.
    assert not dz.interrupts(" and the Dormouse sulkily remarked, ")
    assert not dz.interrupts(" and Leafa commented, ")
    assert not dz.interrupts(" and a Canary called out in a trembling voice to its children, ")
    assert not dz.interrupts(" then the Rabbit\u2019s voice along\u2014")
    # the same speaker carrying on
    assert dz.interrupts(" Then she asked astutely, ")
    assert dz.interrupts(" And then she put a hand to her chest and continued, ")
    assert dz.interrupts(" and I said, ")
    assert dz.interrupts(" Before Miyuki\u2019s unease could rear its head again, he added, ")
    assert dz.interrupts(" Then, he added in his own tongue, ")
    p = ("\u201cThere\u2019s no such thing!\u201d Alice was beginning very angrily, but the Hatter and the March Hare "
         "went \u201cSh! sh!\u201d and the Dormouse sulkily remarked, \u201cIf you can\u2019t be civil, you\u2019d "
         "better finish the story for yourself.\u201d")
    book = {"chapters": [{"paragraphs": [p]}]}
    book["chapters"][0]["quotes"] = [[f"c0p0q{s}", 0, s, e] for s, e in dz.quote_regions(p)]
    _, local = dz.render_chunk(book, dz.make_chunks(book)[0])
    assert sorted(len(v) for v in local.values()) == [1, 1, 1]


def test_turns_and_an_interrupted_quotation_are_numbered():
    book = {"chapters": [{"paragraphs": ["\u201cShall we go to bed now?\u201d Asuna whispered. Right behind her ear, "
                                         "Kirito said, \u201cSounds good.\u201d",
                                         "\u201cYes,\u201d he said, \u201cI will.\u201d"]}]}
    book["chapters"][0]["quotes"] = [
        [f"c0p{pi}q{s}", pi, s, e] for pi, p in enumerate(book["chapters"][0]["paragraphs"])
        for s, e in dz.quote_regions(p)]
    chunk = dz.make_chunks(book)[0]
    _, local = dz.render_chunk(book, chunk)
    assert sorted(len(v) for v in local.values()) == [1, 1, 2]  # two turns, then one interrupted quotation


def _tokens(*pieces):
    """An answer's tokens: (text, probability) pairs."""
    import math

    return [{"token": t, "logprob": math.log(p)} for t, p in pieces]


def test_delivery_confidence_is_read_from_the_answers_tokens():
    toks = _tokens(('{"quotes":[{"id":"1","speaker":"Asuna","delivery":', 1.0), ('["', 0.9), ("af", 0.8),
                   ("raid", 1.0), ('","', 0.5), ("shout", 0.6), ('"]', 1.0), ('},{"id":"2","speaker":"Kirito",', 1.0),
                   ('"delivery":', 1.0), ("[]", 0.7), ("}]}", 1.0))
    (first, second) = dz.delivery_odds(toks)
    assert [t for t, _ in first] == ["afraid", "shout"] and second == []
    assert abs(first[0][1] - 0.72) < 1e-6  # adding a tag at all (0.9) x the tag (0.8)
    assert abs(first[1][1] - 0.3) < 1e-6  # another tag rather than closing the list (0.5) x the tag (0.6)


def test_only_tags_the_model_is_sure_of_are_kept():
    assert dl.MIN_CONF == 0.7
    delivery, odds = dz.sure_delivery(["afraid", "shout"], [("afraid", 0.72), ("shout", 0.3)])
    assert delivery == "afraid" and odds == [["afraid", 0.72], ["shout", 0.3]]
    assert dz.sure_delivery([], []) == (None, [])
    assert dz.sure_delivery(["angry"], None) == ("angry", None)  # no probabilities (a reasoning model): kept
    script = {"characters": {}, "lines": {}}
    dz.apply_chunk_result(script, {"speakers": [], "quotes": [
        {"id": "1", "speaker": "Tom", "delivery": ["tired"], "_dodds": [("tired", 0.09)]},
        {"id": "2", "speaker": "Tom", "delivery": ["angry", "shout"], "_dodds": [("angry", 0.88), ("shout", 0.95)]},
    ]}, {"1": ["a"], "2": ["b"]})
    assert script["lines"]["a"]["delivery"] is None and script["lines"]["a"]["dodds"] == [["tired", 0.09]]
    assert script["lines"]["b"]["delivery"] == "angry, shout"
    pub = dz.ScriptStore.public(script)
    assert pub["lines"]["a"]["dodds"] == [["tired", 0.09]]


async def test_lines_labelled_before_get_their_deliveries_scored(monkeypatch):
    book = {"title": "B", "chapters": [{"title": "One", "paragraphs": [
        "\u201cHow dare you!\u201d she said.", "\u201cGreat, I feel better.\u201d"]}]}
    book["chapters"][0]["quotes"] = [[f"c0p{pi}q{s}", pi, s, e] for pi, p in enumerate(book["chapters"][0]["paragraphs"])
                                     for s, e in dz.quote_regions(p)]
    script = {"characters": {"ann": {"name": "Ann"}}, "lines": {
        "c0p0q0": {"speaker": "ann", "delivery": "angry, shout, laugh"},
        "c0p1q0": {"speaker": "ann", "delivery": "tired"}}}
    chunk = dz.make_chunks(book)[0]
    assert dz.needs_scoring(script, chunk)
    seen = {}

    async def fake(self, session, system, user, schema, max_tokens=0, **kw):
        seen.update(system=system, user=user, kw=kw)
        return {"quotes": [{"id": "1", "delivery": ["angry", "shout"]}, {"id": "2", "delivery": ["tired"]}],
                "_tokens": _tokens(('{"quotes":[{"id":"1","delivery":', 1.0), ('["angry', 0.9), ('","shout', 0.8),
                                   ('"]},{"id":"2","delivery":', 1.0), ('["tired', 0.1), ('"]}]}', 1.0))}

    monkeypatch.setattr(dz.LocalLlm, "complete", fake)
    llm = dz.LocalLlm.__new__(dz.LocalLlm)
    out = await dz.score_deliveries(llm, None, book, script, chunk)
    assert out["c0p0q0"][0] == "angry, shout" and out["c0p1q0"] == (None, [["tired", 0.1]])
    assert "[1] Ann" in seen["user"] and seen["kw"]["logprobs"] and "delivery:" in seen["system"]
