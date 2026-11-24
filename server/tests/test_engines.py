"""Engine adapters' request handling, with stand-in models (no GPU, no model packages)."""
from elvenreader_server.config import ServerConfig
from elvenreader_server.dramatize import speech_model_extra_layers
from elvenreader_server.engines.voxcpm_engine import Engine as VoxCPM
from elvenreader_server.text import spell_stutters


class FakeVox:
    def __init__(self):
        self.calls = []

    def generate_streaming(self, text, **kw):
        self.calls.append((text, kw))
        yield [0.2, -0.2] * 5  # some sound: a silent take is made again


def vox(**options):
    e = VoxCPM(options)
    e.model = FakeVox()
    return e


VOICE = {"ref_audio": "ref.wav", "ref_text": "Hello there.", "seed": 7}


def run(e, text, delivery=None, voice=VOICE):
    list(e.synthesize({"text": text, "voice": voice, "delivery": delivery}, lambda: False))
    return e.model.calls[-1]


def test_voxcpm_continues_the_reference_without_a_delivery_and_steers_with_one():
    e = vox(mode="auto")
    text, kw = run(e, "It was late.")
    assert text == "It was late." and kw["prompt_wav_path"] == "ref.wav" and kw["prompt_text"] == "Hello there."
    assert kw["seed"] == 7
    text, kw = run(e, "Stop!", "shouting, urgent")
    assert text == "(shouting, urgent)Stop!" and "prompt_wav_path" not in kw and kw["reference_wav_path"] == "ref.wav"


def test_voxcpm_takes_a_lines_own_guidance():
    e = vox(cfg_value=2.0)
    assert run(e, "Stop!", "shouting")[1]["cfg_value"] == 2.0
    list(e.synthesize({"text": "Stop!", "voice": VOICE, "delivery": "shouting", "guidance": 1.5}, lambda: False))
    assert e.model.calls[-1][1]["cfg_value"] == 1.5


def test_voxcpm_designs_voices_at_its_own_settings_not_the_reading_tuning():
    # the reading tuning (1.75 / 0.95) gave designs a hum in their pauses
    e = vox(cfg_value=1.75, temperature=0.95)
    e.seed_everything = lambda seed: None
    list(e.design_voice("An adult woman.", "It was late.", 7))
    assert e.model.calls[-1][1]["cfg_value"] == 2.0 and e.temperature == 1.0
    run(e, "It was late.")
    assert e.model.calls[-1][1]["cfg_value"] == 1.75 and e.temperature == 0.95


def test_voxcpm_never_speaks_an_instruction_while_continuing():
    text, kw = run(vox(mode="continuation"), "Stop!", "shouting, urgent")
    assert text == "Stop!" and kw["prompt_wav_path"] == "ref.wav"  # the model would read "(shouting...)" aloud
    text, _ = run(vox(mode="reference"), "It was late.")
    assert text == "It was late."


def test_voxcpm_reads_brackets_as_words():
    assert run(vox(), "(The Neighbor's POV) Hello.")[0] == "The Neighbor's POV, Hello."
    assert run(vox(), "Like Volumes 1 and 2 (Aincrad) and 3 and 4 (Fairy Dance), the title spans both.")[0] == \
        "Like Volumes 1 and 2, Aincrad, and 3 and 4, Fairy Dance, the title spans both."
    assert run(vox(), "the Abyssal Horror (the spacebeast from Volume 18), so")[0] == \
        "the Abyssal Horror, the spacebeast from Volume 18, so"
    assert run(vox(), "He left (for good).")[0] == "He left, for good."
    # The delivery's own brackets stay the only instruction.
    assert run(vox(), "Stop (now)!", "shouting")[0] == "(shouting)Stop, now!"


class TrailingVox(FakeVox):
    """Speaks, pauses 0.48 s, speaks, then misses its stop: near-silence for 6.4 s and a click."""
    rate = 1000

    def generate_streaming(self, text, **kw):
        self.calls.append((text, kw))
        self.made = 0
        talk, hush = [0.1] * 160, [0.0001] * 160
        for chunk in [talk] * 10 + [hush] * 3 + [talk] * 10 + [hush] * 40 + [[0.2] * 20]:
            self.made += 1
            yield chunk


def test_voxcpm_stops_when_the_model_misses_its_stop():
    e = vox()
    e.model, e.sample_rate = TrailingVox(), 1000
    chunks = list(e.synthesize({"text": "Hi there, how are you? Fine.", "voice": VOICE}, lambda: False))
    seconds = sum(len(c) for c in chunks) / 1000
    assert 3.6 <= seconds <= 4.4  # both phrases and the 0.48 s pause, then a 0.3 s tail
    assert e.model.made < 40  # generation stopped about 1.2 s into the silence, not 6.4 s


def test_voxcpm_keeps_a_long_pause_before_the_text_is_spoken():
    e = vox()
    e.model, e.sample_rate = TrailingVox(), 1000
    long_text = "x" * 2000  # far more text than 3.2 s of speech: the silence can't be the end yet
    chunks = list(e.synthesize({"text": long_text, "voice": VOICE}, lambda: False))
    assert e.model.made == 64  # nothing cut short


class RunawayVox(FakeVox):
    """The first generation never ends: a steady quiet whistle. Later ones say the line in 1 s."""
    rate = 1000

    def generate_streaming(self, text, **kw):
        self.calls.append((text, kw))
        self.made = 0
        whistle = len(self.calls) == 1
        try:
            for _ in range(1000 if whistle else 6):
                self.made += 1
                yield [0.01 if whistle else 0.2] * 160
        finally:
            self.first_made = getattr(self, "first_made", self.made)


def test_voxcpm_makes_a_runaway_short_line_again_with_another_seed():
    e = vox()
    e.model, e.sample_rate = RunawayVox(), 1000
    chunks = list(e.synthesize({"text": "Wh… what?!", "voice": VOICE, "delivery": "shouting"}, lambda: False))
    assert len(e.model.calls) == 2 and e.model.made == 6
    assert all(c[0] == 0.2 for c in chunks)  # only the second take: no whistle
    assert sum(len(c) for c in chunks) == 960
    assert e.model.first_made <= 26  # the whistle was stopped at the 4 s cap, not left to run 160 s


def test_voxcpm_cuts_a_runaway_long_line_at_the_length_cap():
    from elvenreader_server.engines.voxcpm_engine import length_cap

    e = vox()
    e.model, e.sample_rate = RunawayVox(), 1000
    text = "word " * 20  # 7 s expected: streamed, so it can't be made again
    chunks = list(e.synthesize({"text": text.strip(), "voice": VOICE}, lambda: False))
    seconds = sum(len(c) for c in chunks) / 1000
    assert len(e.model.calls) == 1 and seconds <= length_cap(len(text.strip()) / 14) < 30


def test_only_lines_an_engine_makes_differently_get_new_cache_keys():
    from elvenreader_server.app import synthesis_key
    from elvenreader_server.engines.registry import ENGINES

    class V:
        fingerprint = "f"

    cfg = ServerConfig(token="t")
    key = lambda eng, text: synthesis_key(cfg, eng, ENGINES[eng], V(), text, "English", None, 1)
    short, long_, bracketed = "Hello there.", "word " * 80, "He left (for good)."
    vox = ENGINES["voxcpm"]
    plain = vox.__class__(**{**vox.__dict__, "max_chunk_chars": 0, "brackets_as_asides": False})
    base = lambda text: synthesis_key(cfg, "voxcpm", plain, V(), text, "English", None, 1)
    assert key("voxcpm", short) == base(short)  # nothing changes for most lines
    assert key("voxcpm", long_.strip()) != base(long_.strip())  # one piece instead of two
    assert key("voxcpm", bracketed) != base(bracketed)  # brackets read as asides
    assert key("omnivoice", long_.strip()) == synthesis_key(
        cfg, "omnivoice", ENGINES["omnivoice"], V(), long_.strip(), "English", None, 1)


def test_shared_llm_keeps_more_experts_in_ram_next_to_a_bigger_speech_model():
    assert speech_model_extra_layers(ServerConfig(token="t", default_engine="omnivoice")) == 0
    assert speech_model_extra_layers(ServerConfig(token="t", default_engine="voxcpm")) == 6


def test_made_up_speech_after_the_last_word_is_heard():
    from elvenreader_server.engines.voxcpm_engine import ran_on

    assert ran_on("Kirito murmured.", "kiitto murmured baley biot anto") == 13
    # A name at the end heard as two words, or misheard, is not extra.
    assert ran_on("\u201cFive plates, please, Kirito.\u201d", "five plates please cury toe") < 6
    assert ran_on("not allowed to fight with the pontifex.", "not allowed to fight with the pont effects") < 6
    assert ran_on("\u201cRonie!!\u201d", "ronney") < 6
    assert ran_on("Hmm.", "") == 0  # nothing recognised: no telling


class TalkativeVox(FakeVox):
    """Every take is half a second of speech; the recogniser hears made-up words after the first one."""

    def generate_streaming(self, text, **kw):
        self.calls.append((text, kw))
        yield [0.2] * 500


class FakeAsr:
    def __init__(self, heard):
        self.heard, self.n = heard, 0

    def generate(self, input, **kw):
        self.n += 1
        return [{"text": "<|en|><|NEUTRAL|>" + self.heard[min(self.n, len(self.heard)) - 1]}]


def test_a_short_line_that_runs_on_is_made_again():
    e = vox()
    e.model, e.sample_rate = TalkativeVox(), 1000
    e.asr = FakeAsr(["kirito murmured baley biot anto", "kirito murmured"])
    list(e.synthesize({"text": "Kirito murmured.", "voice": VOICE}, lambda: False))
    assert len(e.model.calls) == 2 and e.asr.n == 2  # the second take ended where the text does
    e = vox()
    e.model, e.sample_rate = TalkativeVox(), 1000
    e.asr = FakeAsr(["kirito murmured"])
    list(e.synthesize({"text": "Kirito murmured.", "voice": VOICE}, lambda: False))
    assert len(e.model.calls) == 1


def test_words_left_out_are_noticed_but_mishearings_are_not():
    from elvenreader_server.engines.voxcpm_engine import MISSED_LETTERS, MISSED_SHARE, missed

    text = ("Bemused, Ronie extended her left hand to hold Kirito\u2019s and her right hand toward Tiese. Opposite her, "
            "Asuna did the same, looking a bit resigned. The four of them were now forming a circle.")
    skipped = "Opposite her, Asuna did the same, looking a bit resigned. The four of them were now forming a circle."
    misheard = text.replace("Ronie", "Roni").replace("Tiese", "teasy").replace("Kirito\u2019s", "cury toe's")
    m, n = missed(text, skipped)
    assert m >= MISSED_LETTERS and m >= MISSED_SHARE * n
    m, n = missed(text, misheard)
    assert m < MISSED_LETTERS
    m, n = missed("\u201cLikewise, Tiese Schtrinen!\u201d", "likewise tease a strain in")
    assert m < MISSED_LETTERS  # a short line whose name is misheard


def test_a_line_that_stopped_before_its_last_sentence_is_made_again():
    e = vox()
    e.model, e.sample_rate = SkippingVox(), 1000
    e.asr = FakeAsr(["huh that's not as easy as it looks but i guess it's good to be ambitious",
                     "huh that's not as easy as it looks but i guess it's good to be ambitious "
                     "anyway let's hurry and get to that inn"])
    text = ("\u201cHuh?! Th-that\u2019s not as easy as it looks\u2026 B-but I guess it\u2019s good to be ambitious. "
            "Anyway, let\u2019s hurry and get to that inn,\u201d")
    list(e.synthesize({"text": spell_stutters(text), "voice": VOICE, "whole": True}, lambda: False))
    assert len(e.model.calls) == 2 and e.asr.n == 2
    # The spelled stutters ("Wuh-wuh-wait") are heard as the word, not as words left out.
    e = vox()
    e.model, e.sample_rate = SkippingVox(), 1000
    e.asr = FakeAsr(["wait that's mine"])
    list(e.synthesize({"text": spell_stutters("\u201cW-w-wait! Th-th-that\u2019s m-mine!\u201d"), "voice": VOICE,
                       "whole": True}, lambda: False))
    assert len(e.model.calls) == 1


class SkippingVox(FakeVox):
    """A long line: every take is 8 s; the recogniser says the first take left out a sentence."""

    def generate_streaming(self, text, **kw):
        self.calls.append((text, kw))
        for _ in range(16):
            yield [0.2] * 500


LONG = ("Bemused, Ronie extended her left hand to hold Kirito\u2019s and her right hand toward Tiese. Opposite her, "
        "Asuna did the same, looking a bit resigned. The four of them were now forming a circle.")


def test_a_line_taken_whole_that_left_words_out_is_made_again():
    e = vox()
    e.model, e.sample_rate = SkippingVox(), 1000
    e.asr = FakeAsr(["Opposite her, Asuna did the same, looking a bit resigned. The four of them were now forming a circle.",
                     LONG])
    list(e.synthesize({"text": LONG, "voice": VOICE, "whole": True}, lambda: False))
    assert len(e.model.calls) == 2 and e.asr.n == 2
    # Streamed (the client plays it as it comes): sent as made, not heard back.
    e = vox()
    e.model, e.sample_rate = SkippingVox(), 1000
    e.asr = FakeAsr(["nothing"])
    list(e.synthesize({"text": LONG, "voice": VOICE}, lambda: False))
    assert len(e.model.calls) == 1 and e.asr.n == 0


def test_voxcpm_passes_a_temperature_to_its_decoder():
    from types import SimpleNamespace

    seen = []

    class Decoder:
        def forward(self, mu, temperature=1.0, **kw):
            seen.append(temperature)

    e = vox(temperature=0.8)
    e.model = FakeVox()
    e.model.tts_model = SimpleNamespace(feat_decoder=Decoder())
    e.temperature = 0.8
    e._wrap_decoder()
    e.model.tts_model.feat_decoder.forward("mu")
    list(e.synthesize({"text": "Hi.", "voice": VOICE, "temperature": 0.6}, lambda: False))
    e.model.tts_model.feat_decoder.forward("mu")
    assert seen == [0.8, 0.6]  # the engine's, then the line's own


def test_numbers_the_normaliser_reads_wrong_are_put_right_first():
    import importlib.util

    f = VoxCPM._for_normalizer
    assert f("the 2nd-year students and 3rd-floor rooms") == "the 2nd year students and 3rd floor rooms"
    if importlib.util.find_spec("inflect"):  # the engine's venv has it
        assert f("about 1,342 of them") == "about one thousand, three hundred and forty-two of them"
    assert f("1.5 and 3,14 and 2026") == "1.5 and 3,14 and 2026"


def test_voxcpm_style_slots_build_its_instruction():
    from elvenreader_server.engines import voxcpm_styles as vs

    assert vs.instruction() == ""  # all neutral: no instruction, the voice's own clip carries on
    assert vs.instruction("angry", "strong", "loud") == "very angry, speaking loudly"
    assert vs.instruction("sad", volume="whisper", pace="slow") == "sad, whispering, slowly"
    assert vs.instruction(intensity="strong", volume="soft") == "speaking softly"  # no emotion: no "very"
    assert vs.instruction("excited") == ""  # dropped: no measurable effect
    assert vs.sound_tag("sigh") == "[sigh-hah]" and vs.sound_tag("cough") == "" and vs.sound_tag(None) == ""
    assert vs.instruction("sad", manner="breathless", pace="slow") == "sad, breathless, slowly"
    for table in (vs.EMOTIONS, vs.INTENSITY, vs.MANNER, vs.VOLUME, vs.PACE, vs.SOUNDS):
        assert all(status in ("works", "acoustic", "weak") for _, status, _ in table.values())
    for table, dropped in ((vs.EMOTIONS, vs.DROPPED_EMOTIONS), (vs.MANNER, vs.DROPPED_MANNER)):
        assert not set(table) & set(dropped)


def test_voxcpm_is_given_no_quotation_marks():
    # '"paper"' and '"Yes."' came out as nothing at all; without the marks they are said.
    assert run(vox(), "\u201cpaper\u201d")[0] == "paper"
    assert run(vox(), "\u201cYes.\u201d")[0] == "Yes."
    assert run(vox(), "\u2018Hello,\u2019")[0] == "Hello,"
    assert run(vox(), "\u201cI didn\u2019t see the girls\u2019 room, did I?\u201d")[0] == \
        "I didn\u2019t see the girls\u2019 room, did I?"  # apostrophes stay
    assert run(vox(), "how to say \u201cpaper\u201d in this place\u2019s language.")[0] == \
        "how to say paper in this place\u2019s language."


class SilentFirst(FakeVox):
    """Says nothing the first time, then speaks."""

    def generate_streaming(self, text, **kw):
        self.calls.append((text, kw))
        yield [0.0] * 2400 if len(self.calls) == 1 else [0.2, -0.2] * 1200


def test_a_take_with_no_speech_is_made_again():
    e = vox()
    e.model = SilentFirst()
    out = list(e.synthesize({"text": "\u201cpaper\u201d", "voice": VOICE}, lambda: False))
    assert len(e.model.calls) == 2 and any(max(abs(x) for x in a) > 0.1 for a in out)


def test_short_lines_are_heard_back_too():
    from elvenreader_server.engines.voxcpm_engine import missed

    assert missed("No.", "no") == (0, 2) and missed("Ah!", "") == (2, 2)
    assert missed("paper", "pay")[0] >= 3 and missed("paper", "pay per")[0] == 2
    e = vox()
    e.asr = object()
    take = [[0.2, -0.2] * 1200]
    e._heard = lambda chunks: "pay"
    assert e._fault("paper", take) > 0  # under the 15-letter limit, but half of it not heard
    e._heard = lambda chunks: "paper"
    assert e._fault("paper", take) == 0
    e._heard = lambda chunks: "teas"
    assert e._fault("Thank you, Tiese, for all of it.", take) > 0  # long: the usual limits


def test_no_the_word_is_not_read_as_number():
    fix = VoxCPM._for_normalizer
    assert fix("No. I won't.") == "no. I won't." and fix("No...") == "no..." and fix("NO.") == "no."
    assert fix("Room No. 5") == "Room No. 5"  # the abbreviation: "Number five"


def test_lines_read_differently_now_get_a_new_cache_key(tmp_path):
    from elvenreader_server.app import synthesis_key
    from elvenreader_server.engines.registry import ENGINES

    cfg = ServerConfig(token="t")
    spec = ENGINES["voxcpm"]

    class V:
        fingerprint = "v1"

    key = lambda text: synthesis_key(cfg, "voxcpm", spec, V(), text, None, None, 1)
    old = __import__("dataclasses").replace(spec, reread={})
    assert key("It was late.") == synthesis_key(cfg, "voxcpm", old, V(), "It was late.", None, None, 1)
    assert key("\u201cNo. It looks terrible.\u201d") != synthesis_key(
        cfg, "voxcpm", old, V(), "\u201cNo. It looks terrible.\u201d", None, None, 1)


def test_a_short_line_with_made_up_syllables_is_made_again():
    e = vox()
    e.asr = object()
    take = [[0.2, -0.2] * 1200]
    e._heard = lambda chunks: "tap heaper"
    assert e._fault("paper", take) > 0
    e._heard = lambda chunks: "number"
    assert e._fault("No.", take) > 0
    e._heard = lambda chunks: "yes"
    assert e._fault("Yes!", take) == 0


def test_analysis_shares_the_gpu_only_when_it_fits_next_to_the_speech_models(monkeypatch):
    from types import SimpleNamespace

    from elvenreader_server import dramatize

    monkeypatch.setattr(dramatize, "_GPU_TOTAL_GB", [15.9])  # RTX 5080
    fits = lambda speech, narration="": dramatize.shared_fits(SimpleNamespace(default_engine=speech, narration_engine=narration))
    assert fits("voxcpm", "omnivoice") and fits("breeze") and fits("voxcpm")
    assert not fits("breeze", "omnivoice")  # 14 GB measured with Windows: the LLM's 3 GB would spill


def test_breeze_reads_deliveries_as_instructions_and_asides_as_dashes():
    from elvenreader_server.engines.breeze_engine import instruction_for, spoken

    assert instruction_for("very angry, shout") == "Very angry, speaking loudly."
    assert instruction_for(None) == "" and instruction_for("") == ""
    assert spoken('He said (quietly) \u201cno\u201d.') == "He said \u2014 quietly \u2014 no."

