"""VoxCPM2 (OpenBMB) adapter: zero-shot cloning that also follows a free-text delivery ("(shouting, urgent)").

VoxCPM2 clones in two ways. Continuation ("ultimate cloning": the reference clip and its transcript, the model
carries on in that voice) is the closest likeness but copies the clip's own manner. Isolated reference (the clip
gives the timbre only) can be steered by a style instruction in parentheses at the start of the text. `mode`:
"auto" (default) uses continuation for lines without a delivery and the reference with the delivery for lines
that have one; "continuation" or "reference" use one way for every line. 48 kHz output.
"""
from __future__ import annotations

import re

from .base import BaseEngine
from .voxcpm_styles import render as render_delivery

# A parenthesis opening the text would be read as a style instruction.
_LEADING_PAREN = re.compile(r"^\s*\(([^()]*)\)\s*")
# VoxCPM2's non-verbal sound tags ("[laugh-haha]", "[Uhm]"): from sound marks or a delivery's sound.
_TAGS = re.compile(r"\[[A-Za-z][A-Za-z\-]*\]")
# Quotation marks are for the reader's eye, not the voice. Given to VoxCPM2 they did harm: '"paper"' (a one-word
# quotation) came out as nothing at all, every seed, while 'paper' was said. Double marks always go; single ones
# only where they surely open (after a space) or close (after punctuation: ",' ?'") a quotation, never as
# apostrophes ("didn't", "'em", "girls' room" keep theirs).
_QUOTE_MARKS = re.compile(r"[\"\u201c\u201d\u201e\u201f\u00ab\u00bb\u300c\u300d\u300e\u300f]"
                          r"|(?<!\w)\u2018|(?<![^\s])'(?=[A-Z])|(?<=[.,!?;:\u2026\u2014-])[\u2019']")


def _unquoted(text: str) -> str:
    """The text without its quotation marks (see _QUOTE_MARKS), spaces tidied."""
    return re.sub(r"\s{2,}", " ", _QUOTE_MARKS.sub("", text)).strip()


def _has_speech(take: list) -> bool:
    """Whether any of a take's chunks is louder than silence (QUIET_RMS)."""
    import numpy as np

    for a in take:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size and float(np.sqrt(np.mean(a * a))) >= QUIET_RMS:
            return True
    return False
QUIET_RMS = 120 / 32768  # a chunk quieter than this is silence
STOP_AFTER_S = 1.2  # silence after the speech that ends a generation (natural pauses measured up to 0.74 s)
KEEP_TAIL_S = 0.3  # of the silence at the end
CHARS_PER_S = 14.0  # about how fast English is read aloud; for "has the text been spoken yet"
REF_MIN_GAP_DB = 25.0  # references whose pauses sit less far under the speech are cleaned before cloning
# A generation longer than this has run away (a 13.6 s whistle for "Wh…whaaaaaaat?!"): it is cut there.
MAX_LENGTH_FACTOR, MAX_LENGTH_PAD_S, MIN_CAP_S = 3.0, 2.0, 4.0
WHOLE_UP_TO_S = 4.0  # lines expected to take up to this are made whole first, so a runaway can be made again
RETRIES = 2  # with other seeds


class _Runaway(Exception):
    pass


def length_cap(expected_s: float) -> float:
    return max(MIN_CAP_S, MAX_LENGTH_FACTOR * expected_s + MAX_LENGTH_PAD_S)


RAN_ON_LETTERS = 6  # letters heard after a short line's last word that mean the model made speech up
# Letters of the text not heard: a take that dropped words. The narrator's seed dropped a whole sentence of a
# 189-character passage at guidance 1.5 (72 of 149 letters, 48%); one of Kirito's lines stopped before its last
# sentence ("Anyway, let's hurry and get to that inn": 30 of 85 letters, 35%). Over Vol. 20's 522 clips no other
# reached a quarter with more than 13 (short lines with a name misheard: "Tiese Schtrinen" -> "tease a strain
# in"); the most any long one missed was 31 (11%: numbers heard as digits, names).
MISSED_LETTERS, MISSED_SHARE = 15, 0.25
# Stutters spelled for VoxCPM2 ("Thuh-that", text.spell_stutters) are heard as the word ("that"): the added
# syllables are left out of the comparison.
_SPELLED_STUTTER = re.compile(r"\b(?:[b-df-hj-np-tv-z]{1,3}uh-)+(?=\w)", re.IGNORECASE)


def missed(text: str, heard: str) -> tuple[int, int]:
    """(letters of the text not heard, letters in the text); runs of 3+ letters count as heard (the whole word
    for a shorter text: "No", "Ah")."""
    import difflib

    a, b = re.sub(r"[^a-z]", "", text.lower()), re.sub(r"[^a-z]", "", heard.lower())
    least = min(3, len(a)) or 1
    got = sum(m.size for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()
              if m.size >= least)
    return len(a) - got, len(a)


def ran_on(text: str, heard: str) -> int:
    """How many letters a recogniser heard after the text's last word: made-up speech when VoxCPM2's stop
    head misses ("Kirito murmured." + "baley biot anto": 13). Letters, not words: a name at the end heard as
    two words ("Kirito" -> "cury toe") is not extra. Over Vol. 20's 319 clips that one scored 13, no other
    more than 3. 0 when nothing of the text was recognised (no telling where it ends)."""
    import difflib

    a, b = re.sub(r"[^a-z]", "", text.lower()), re.sub(r"[^a-z]", "", heard.lower())
    blocks = [m for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks() if m.size >= 3]
    if not blocks:
        return 0
    last = blocks[-1]
    return max(0, len(b) - (last.b + last.size) - max(0, len(a) - (last.a + last.size)))


def _gap_db(path) -> float | None:
    """noise_gap_db of a WAV file (the worker has no server package, so it is measured here the same way)."""
    import numpy as np
    import soundfile as sf

    x, rate = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    win = max(1, int(rate * 0.04))
    n = len(x) // win
    if n < 5:
        return None
    frames = x[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = rms > 0.02
    if not loud.any():
        return None
    near = np.convolve(loud.astype(int), np.ones(5, dtype=int), "same") > 0
    quiet = frames[(~near) & (rms > 1e-6)]
    if len(quiet) < 2:
        return None
    return float(20 * np.log10(np.sqrt((frames[loud] ** 2).mean()) / (np.sqrt((quiet ** 2).mean()) + 1e-9)))


class Engine(BaseEngine):
    sample_rate = 48000

    def load(self) -> None:
        optimize = bool(self.options.get("optimize", True))
        if optimize:
            # torch.compile (Triton) makes it about 4.5x faster (RTF 2.0 -> 0.44 on an RTX 5080). PyTorch 2.8's static
            # CUDA launcher passes pointers as a C long, which is 32 bits on Windows: "Python int too large".
            import torch._inductor.config as inductor

            inductor.use_static_cuda_launcher = False
        from voxcpm import VoxCPM

        # ZipEnhancer (ModelScope) cleans noisy references; see _reference.
        self.denoise = bool(self.options.get("denoise_refs", True))
        kwargs = {"load_denoiser": self.denoise, "optimize": optimize}
        try:
            self.model = VoxCPM.from_pretrained(self.options.get("repo", "openbmb/VoxCPM2"), **kwargs)
        except TypeError:
            kwargs.pop("optimize", None)
            self.model = VoxCPM.from_pretrained(self.options.get("repo", "openbmb/VoxCPM2"), **kwargs)
        except ImportError:  # no modelscope: clone the references as they are
            self.denoise = kwargs["load_denoiser"] = False
            self.model = VoxCPM.from_pretrained(self.options.get("repo", "openbmb/VoxCPM2"), **kwargs)
        self.sample_rate = int(getattr(self.model.tts_model, "sample_rate", 48000))
        self._refs: dict[str, str] = {}
        # VoxCPM2's own text normalisation, run here on the words alone (not the style instruction), so the checks
        # below compare what was heard with what was said ("Vol. 20" -> "Volume twenty").
        self.tn = None
        if self.options.get("normalize", False):
            from voxcpm.utils.text_normalize import TextNormalizer

            self.tn = TextNormalizer()
        # The noise each patch of speech starts from is scaled by a temperature its diffusion decoder takes but is
        # never given (always 1.0); it is passed in here. Only the decoder's estimator is compiled, so this holds.
        self.temperature = float(self.options.get("temperature", 1.0))
        self._wrap_decoder()
        # SenseVoice-Small (FunASR, ~1 GB of VRAM, ~0.1 s a line) hears short lines back: see _ran_on.
        self.asr = None
        if self.options.get("check_endings", True):
            try:
                import torch
                from funasr import AutoModel

                # FunASR sets torch's thread count (4 unless told), which is part of what torch.compile checks: the
                # compiled VoxCPM2 was compiled again on the first line after every load (8 s). Given the current
                # count, it changes nothing.
                self.asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda", disable_update=True,
                                     disable_pbar=True, log_level="ERROR", ncpu=torch.get_num_threads())
            except Exception as exc:  # noqa: BLE001 - lines are still made, just not heard back
                import sys

                print(f"VoxCPM2: no SenseVoice to check line endings ({exc})", file=sys.stderr)

    def _wrap_decoder(self) -> None:
        decoder = self.model.tts_model.feat_decoder
        forward = decoder.forward
        decoder.forward = lambda *a, **k: forward(*a, **{**k, "temperature": self.temperature})

    # -- stepping aside for the LLM -----------------------------------------------------------------------------
    can_park = True

    def _gpu_modules(self) -> list:
        """Everything this adapter keeps on the GPU: VoxCPM2, its reference denoiser and SenseVoice."""
        import torch

        found = [self.model.tts_model]
        pipeline = getattr(getattr(self.model, "denoiser", None), "_pipeline", None)
        found.append(getattr(pipeline, "model", None))
        found.append(getattr(self.asr, "model", None))
        return [m for m in found if isinstance(m, torch.nn.Module)]

    def park(self) -> None:
        """The loaded model to system RAM, its VRAM freed, without losing what loading built: the compiled code
        stays (it is the same model, on the same device, when it comes back). The CUDA graphs torch.compile
        recorded point at the old weights; torch records them again when it sees the weights moved."""
        import gc

        import torch

        for m in self._gpu_modules():
            m.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def unpark(self) -> None:
        for m in self._gpu_modules():
            m.to("cuda")

    def _heard(self, chunks: list) -> str:
        import numpy as np

        a = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        try:
            from scipy.signal import resample_poly

            a = resample_poly(a, 16000, self.sample_rate).astype(np.float32)
        except ImportError:
            a = a[:: max(1, self.sample_rate // 16000)]
        text = self.asr.generate(input=a, language="auto", use_itn=False, disable_pbar=True)[0]["text"]
        return re.sub(r"<\|[^|]*\|>", "", text)

    def _reference(self, ref: str) -> str:
        """The reference clip to clone from. VoxCPM2 copies everything in it, background included: a voice whose
        pauses hum or hiss (less than REF_MIN_GAP_DB under the speech; one it designed itself measured 16 dB) gave
        that buzz to every line. Such a reference is cleaned once with ZipEnhancer and kept next to it in .clean/
        (one of Ronie's: lines went from 16 to 51 dB). Clean ones are left exactly as they are."""
        if not getattr(self, "denoise", False) or getattr(self.model, "denoiser", None) is None:
            return ref
        if ref not in self._refs:
            from pathlib import Path

            src = Path(ref)
            clean = src.parent / ".clean" / f"{src.stem}.wav"
            if clean.exists():
                self._refs[ref] = str(clean)
            elif (gap := _gap_db(src)) is not None and gap < REF_MIN_GAP_DB:
                clean.parent.mkdir(parents=True, exist_ok=True)
                tmp = clean.with_suffix(".tmp.wav")
                self.model.denoiser.enhance(str(src), output_path=str(tmp))
                tmp.replace(clean)
                self._refs[ref] = str(clean)
            else:
                self._refs[ref] = ref
        return self._refs[ref]

    def _kwargs(self, seed, guidance=None) -> dict:
        kw = {"cfg_value": float(guidance if guidance is not None else self.options.get("cfg_value", 2.0)),
              "inference_timesteps": int(self.options.get("inference_timesteps", 10))}
        if seed is not None:
            kw["seed"] = int(seed)
        return kw

    @staticmethod
    def _for_normalizer(text: str) -> str:
        """What VoxCPM2's normaliser gets wrong, put so it gets it right: "2nd-year" came out "two nd-year"
        ("2nd year": "second year"), "1,342" / "1,000" lost their "one" ("thousand three hundred..."), and
        "No." (the word, 171 times in the library: "No. It looks terrible.") was read "number" (it is left
        alone written "no."; "No. 5" still becomes "Number five")."""
        text = re.sub(r"(\d+(?:st|nd|rd|th))-(?=[A-Za-z])", r"\1 ", text)
        text = re.sub(r"\bN[Oo]\.(?!\s*\d)", "no.", text)
        try:
            import inflect
        except ImportError:
            return text
        p = inflect.engine()
        return re.sub(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,]*\.\d)(?!\d)",
                      lambda m: p.number_to_words(int(m.group(0).replace(",", ""))), text)

    def _normalized(self, text: str) -> str:
        if getattr(self, "tn", None) is None:
            return text
        # Sound tags ("[laugh-haha]") pass the normaliser untouched.
        return "".join(p if _TAGS.fullmatch(p) else self.tn.normalize(self._for_normalizer(p)) if p.strip() else p
                       for p in re.split(f"({_TAGS.pattern})", text))

    @staticmethod
    def _plain(text: str) -> str:
        """Brackets are style instructions to VoxCPM2 anywhere in the text: "Volumes 1 and 2 (Aincrad)" lost the word
        and "(the spacebeast from the ending of Volume 18)" came out garbled. They are read as asides, between
        commas."""
        if "(" not in text and ")" not in text:
            return text
        text = _LEADING_PAREN.sub(lambda m: f"{m.group(1)}, ", text, count=1)
        text = re.sub(r"\s*\(\s*", ", ", text)
        text = re.sub(r"\s*\)\s*(?=[,.;:!?\u2026\u201d\u2019\"']|$)", "", text)
        text = re.sub(r"\s*\)\s*", ", ", text)
        return re.sub(r",\s*,", ",", text).lstrip(", ")

    def _fault(self, words: str, take: list) -> int:
        """0 for a take that says the text, or how badly it went wrong: letters made up after the text's last
        word (ran_on) plus letters of the text not said (missed). A take with no speech at all is the worst
        fault, recogniser or not ('"paper"' came out as nothing, and that silence was served). A short text
        (under MISSED_LETTERS letters: one-word quotations) is judged by its own length: half its letters not
        heard is a fault. 0 without a recogniser otherwise."""
        letters = len(re.sub(r"[^a-z]", "", words.lower()))
        if not _has_speech(take):
            return 1000 + letters
        if getattr(self, "asr", None) is None:
            return 0
        words = _SPELLED_STUTTER.sub("", words)
        heard = self._heard(take)
        extra = ran_on(words, heard)
        miss, total = missed(words, heard)
        if total < MISSED_LETTERS:
            # A short line is heard as a whole: half of it missing, or syllables that aren't in it ("paper"
            # came out "tap heaper"; that slipped under the ran_on limit).
            longer = len(re.sub(r"[^a-z]", "", heard.lower())) - total
            dropped = total > 0 and (2 * miss >= total or longer >= max(3, total // 2))
            if dropped:
                miss = max(miss, longer, 1)
        else:
            dropped = miss >= MISSED_LETTERS and miss >= MISSED_SHARE * total
        return (extra if extra >= RAN_ON_LETTERS else 0) + (miss if dropped else 0)

    def _generate(self, text: str, words: str, cancelled, whole: bool = False, **kwargs):
        """`text` as the model gets it (a style instruction in front); `words` is what is to be said. Lines the
        client takes whole (`whole`: the website asks for compressed audio) and short ones are made whole
        first and heard back: a take that ran away, ran on into made-up speech or dropped words is made again
        with another seed (the best take is kept). Longer streamed lines are sent as they are made."""
        stream = getattr(self.model, "generate_streaming", None)
        if stream is None:
            yield self._call(self.model.generate, text, kwargs)
            return
        expected_s = len(words) / CHARS_PER_S
        if expected_s > WHOLE_UP_TO_S and not whole:
            try:
                yield from self._until_silence(self._call(stream, text, kwargs), expected_s, cancelled)
            except _Runaway:
                pass  # streamed already: cut where it went wrong
            return
        seed = kwargs.get("seed")  # _call drops it from kwargs when the model takes none
        first = None
        best = None  # (fault, take): the least wrong take, if every one went wrong
        for attempt in range(1 + RETRIES):
            if attempt:
                seed = int(seed or 0) + 1009
                self.seed_everything(seed)
                if "seed" in kwargs:
                    kwargs["seed"] = seed
            out = []
            try:
                for a in self._until_silence(self._call(stream, text, kwargs), expected_s, cancelled):
                    out.append(a)
            except _Runaway:
                first = out if first is None else first
                continue
            if cancelled():
                return
            fault = self._fault(words, out)
            if fault:
                if best is None or fault < best[0]:
                    best = (fault, out)
                continue
            yield from out
            return
        if not cancelled():
            yield from (best[1] if best else first) or []

    def _until_silence(self, chunks, expected_s: float, cancelled):
        """Passes the model's chunks on, and stops it once the speech is over. VoxCPM2 ends when its stop head
        says so; after a trailing "..." it can miss and keep making near-silence (6 s once, ending on a click) up
        to its length cap, and its retry for over-long output only works without streaming (and only above six
        times the text's length). Silence after the speech is held back; STOP_AFTER_S of it, once at least half
        the text's expected time has been spoken, ends the generation. Past length_cap (it went wrong without
        going quiet: a steady whistle) it raises _Runaway."""
        import numpy as np

        cap = length_cap(expected_s)
        total, spoken, quiet, held = 0.0, 0.0, 0.0, []
        try:
            for chunk in chunks:
                if cancelled():
                    return
                a = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if a.size == 0:
                    continue
                seconds = a.size / self.sample_rate
                total += seconds
                if total > cap:
                    raise _Runaway
                if float(np.sqrt(np.mean(a * a))) < QUIET_RMS:
                    if not spoken:
                        yield a  # leading silence: the trimmer cuts it
                        continue
                    held.append(a)
                    quiet += seconds
                    if quiet >= STOP_AFTER_S and spoken >= 0.5 * expected_s:
                        break  # it missed its stop
                    continue
                yield from held  # a pause inside the line: kept
                held, quiet = [], 0.0
                spoken += seconds
                yield a
            if held:
                yield np.concatenate(held)[: int(self.sample_rate * KEEP_TAIL_S)]
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()

    @staticmethod
    def _call(fn, text, kwargs):
        try:
            return fn(text=text, **kwargs)
        except TypeError:  # an older voxcpm without `seed`
            kwargs.pop("seed", None)
            return fn(text=text, **kwargs)

    def synthesize(self, req: dict, cancelled):
        voice = req.get("voice") or {}
        seed = req.get("seed") if req.get("seed") is not None else voice.get("seed")
        self.seed_everything(seed)
        text = self._normalized(_unquoted(self._plain(req["text"])))
        words = _TAGS.sub("", text).strip()  # sound tags (from sound marks) are not words to hear back
        delivery = " ".join((req.get("delivery") or "").split())
        # Approved tags -> the words and sound tag VoxCPM2 measurably follows (voxcpm_styles.render).
        delivery, sound = render_delivery(delivery)
        if sound and not _TAGS.search(text):  # the text's own sounds come first
            text = f"{sound}{text}"  # `words` (what the checks hear for) stays without it
        self.temperature = float(req["temperature"] if req.get("temperature") is not None
                                 else self.options.get("temperature", 1.0))
        kw = self._kwargs(seed, req.get("guidance"))
        ref, ref_text = voice.get("ref_audio"), (voice.get("ref_text") or "").strip()
        mode = self.options.get("mode", "auto")
        if not ref:
            # No reference: voice design from the voice's description (and the delivery).
            style = ", ".join(x for x in (voice.get("instruct"), delivery) if x)
            yield from self._generate(f"({style}){text}" if style else text, words, cancelled,
                                      whole=bool(req.get("whole")), **kw)
            return
        ref = self._reference(ref)
        continuation = ref_text and (mode == "continuation" or (mode == "auto" and not delivery))
        if continuation:
            # A style instruction here is spoken aloud ("Shouting, furious, I told you..."): no delivery.
            kw.update(prompt_wav_path=ref, prompt_text=ref_text, reference_wav_path=ref)
        else:
            kw["reference_wav_path"] = ref
            if delivery:
                text = f"({delivery}){text}"
        yield from self._generate(text, words, cancelled, whole=bool(req.get("whole")), **kw)

    # Voice design runs at VoxCPM2's own settings, not the reading tuning: at guidance 1.75 / temperature 0.95
    # (tuned for reading) designs came out with a hum in their pauses, 10-13 dB under the speech, against 25 dB
    # for the same description and seed at 2.0 / 1.0 (scripts/voxcpm_park_noise_probe.py --server).
    DESIGN_GUIDANCE, DESIGN_TEMPERATURE = 2.0, 1.0

    def design_voice(self, description: str, text: str, seed):
        self.seed_everything(seed)
        self.temperature = self.DESIGN_TEMPERATURE
        words = self._normalized(self._plain(text))
        yield from self._generate(f"({description}){words}", words, lambda: False, whole=True,
                                  **self._kwargs(seed, guidance=self.DESIGN_GUIDANCE))
