# AndroidReader

Listen to your EPUBs read by high-quality voice models running on **your own PC**, at home or away.
A standalone Android app plus this PC as the server; no cloud service and no other reading app.

* **The PC server** (`server/`) loads open-weight text-to-speech models on the GPU itself (no
  VoiceStudio or cloud service), clones voices from a short reference clip, applies your
  pronunciation fixes, caches every sentence, and streams audio to the phone.
* **The Android app** (`android/`) is an EPUB reader that reads aloud with highlighting, background
  playback and lock-screen/Bluetooth controls.
* **Dramatized reading (optional):** a local LLM works out who says each line and how
  (*"Stop!" she shouted* → the character's voice, shouting), and each character gets a voice.

```mermaid
flowchart LR
    subgraph Phone
      R[AndroidReader<br/>EPUB reader] --> C[ServerClient<br/>tries each server URL]
    end
    C -- "HTTP + token<br/>home Wi-Fi or your VPN" --> S
    subgraph PC
      S[androidreader_server<br/>port 3950] --> W[engine worker<br/>one model on the GPU]
      S --> V[(voices, pronunciations,<br/>audio cache)]
      S --> L[llama.cpp LLM<br/>book analysis]
    end
```

| Folder | What it is |
|--------|------------|
| `server/` | `androidreader_server`: aiohttp API, engine manager and workers, voice store, pronunciation dictionary, audio cache, dramatization. |
| `android/` | Kotlin/Compose app: library, reader, Media3 playback service, settings. |

## Voice models

All four run on an RTX 5080 (16 GB) with PyTorch 2.8 + CUDA 12.8. Only one is on the GPU at a
time; asking for another swaps it. Measured on this PC with `scripts\bench_engines.py` (five
sentences, two with a delivery, imported voice *The Documentarian*, after one warm-up sentence).
*RTF* is generation time ÷ audio length, so below 1.0 is faster than real time. *WER* is
Whisper-turbo's word error rate on the output (intelligibility).

| Engine (id) | Why it's here | Delivery control | Phonemes | RTF | First audio | Load | VRAM | WER | Licence |
|---|---|---|---|---|---|---|---|---|---|
| **OmniVoice** (`omnivoice`, default) | VoiceStudio's model, so imported voices sound the same; 600+ languages; 32-step quality preset | non-verbal tags only (`[laughter]`, `[sigh]`) | CMU `[B EY1 S]` | 0.27 | 0.9 s | 9 s | 4.0 GB | 0% | Apache-2.0 |
| **Fish Audio S2 Pro** (`fish_s2`) | Best-rated open model for clarity and expressiveness; free-form inline tags | `[shouting, urgent]` tags | – | 0.60 | 2.0 s | ~3 min | 13.8 GB | 0% | Fish Audio Research (non-commercial) |
| **IndexTTS 2.5** (`indextts`) | Expressive cloning; emotion from a delivery text | emotion text → emotion vector | CMU `<word\|PH>` | 0.54 plain, ~1.5–2.7 on short lines with a delivery | 2.8 s | 1 min | 6.8 GB | 0% | bilibili (non-commercial) |
| **Qwen3-TTS 1.7B** (`qwen3tts`) | Base model for cloning plus **VoiceDesign** to create character voices from a description | – (VoiceDesign only) | – | 1.76 | 6.1 s | 15 s | 4.4 GB | 0% | Apache-2.0 |

Whisper transcribes *"Stop!" she shouted* with an exclamation mark only for the two engines that
received the delivery (Fish S2 Pro, IndexTTS); OmniVoice and Qwen3-TTS read it as *"Stop,"*.
Speaker similarity to the reference wasn't measured objectively; listen to
`server\data\bench\<engine>-<voice>.wav` to compare.

Notes:
* Fish S2 Pro and IndexTTS use `torch.compile` (via `triton-windows`), which roughly doubles
  their speed (Fish: RTF 1.8 → 0.6) but makes loading slow; keep the reader's default engine on
  OmniVoice or set `default_engine` and let it preload at start-up.
* Fish S2 Pro's context is capped at 4k tokens and its codec runs in bf16; otherwise it needs
  ~21 GB and spills out of a 16 GB card at a third of the speed.
* IndexTTS spends ~1.5 s per line analysing a delivery text, so short dramatized lines are slower
  than real time; the reader's look-ahead absorbs it.
* Qwen3-TTS is slower than real time on Windows (Python-bound, no FlashAttention) and slower still
  while the LLM runs on the CPU. It is the **voice-design** engine; for continuous listening
  prefer the other three.
* Excluded on purpose: speed/lightweight models (Kokoro, Piper, Chatterbox Turbo), VibeVoice 7B
  (~19 GB of weights, doesn't fit), Higgs Audio v2 (needs 24 GB).

## 1. Set up the PC server

Requirements: Windows, an NVIDIA GPU with recent drivers, ~60 GB free disk for models and
environments. [`uv`](https://docs.astral.sh/uv/) is installed automatically with winget if missing.

```powershell
cd D:\Dev\AndroidReader\server
# Where models and per-engine Python environments go (outside git). Default: server\data\models
'{"models_dir": "D:\\Models\\AndroidReader"}' | Set-Content -Encoding ascii data\config.json

.\setup-engines.ps1                 # one venv per engine (torch cu128); or e.g. .\setup-engines.ps1 omnivoice fish_s2
.\download-models.ps1               # weights into <models_dir>\hf and <models_dir>\checkpoints
.\run-server.ps1 --import-voicestudio   # optional: copy VoiceStudio's voice profiles (read-only)
.\run-server.ps1                    # start; prints the token, URLs and a pairing QR code
```

`run-server.ps1` creates the server's own small environment (`server\.venv`) on first run.
Options: `--port 3960`, `--engine fish_s2` (default engine), `--no-preload`, `--rotate-token`,
`--show-pairing`. Settings live in `server\data\config.json` (git-ignored, contains the token):

| Key | Default | Meaning |
|---|---|---|
| `models_dir` | `data\models` | Engine venvs, model weights, compile caches, LLM files. |
| `default_engine` / `default_voice` | `omnivoice` / first voice | Used when the phone doesn't choose. |
| `extra_urls` | `[]` | More URLs for the pairing codes, e.g. `["http://10.8.0.2:3950"]` for the PC on your VPN. |
| `preload` | `true` | Load the default engine at start-up. |
| `engine_options` | `{}` | Per engine, e.g. `{"omnivoice": {"num_step": 16}, "indextts": {"compile": true}}`. |
| `sentence_pause_ms`, `trim_silence` | `300`, `true` | Pacing between sentences. |
| `cache_max_mb` | `4096` | Sentence audio cache (`data\cache`). |
| `llm_model`, `llm_mode`, `llm_cpu_moe`, `llm_threads` | Qwen3-30B-A3B, `auto`, `20`, `12` | Dramatization LLM (see below). |
| `voice_design_engine` | `qwen3tts` | Creates character voices. |

Environment overrides: `ANDROIDREADER_PORT`, `ANDROIDREADER_HOST`, `ANDROIDREADER_TOKEN`,
`ANDROIDREADER_DATA`, `ANDROIDREADER_MODELS`. Every request is logged with its engine, audio
length, time to first audio and RTF in `server\data\server.log`.

### Voices

Voices live in `server\data\voices` as a WAV plus a JSON file (name, transcript, language,
seed). Every voice has reference audio and its transcript, so every engine can clone it and the
speaker never changes between sentences. Add voices by importing from VoiceStudio
(`--import-voicestudio`, read-only), by `POST /api/voices` (WAV + transcript), or by designing
one from a description (`POST /api/voices/design`).

### Windows firewall (once, elevated)

```powershell
cd D:\Dev\AndroidReader\server
.\allow-firewall.ps1        # TCP 3950 on Domain and Private networks
```

Your home Wi-Fi must be a **Private** network (and so must your VPN's adapter, if the phone
reaches the PC through it). Without this rule the phone can't connect (the app reports *Could
not reach the PC … failed to connect*).

### Start automatically (optional)

```powershell
$a = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\Dev\AndroidReader\server\run-server.ps1"
Register-ScheduledTask -TaskName "AndroidReader server" -Action $a -Trigger (New-ScheduledTaskTrigger -AtLogOn)
```

## 2. Away from home: your own VPN

Use your own VPN to reach the PC from outside, and add the PC's address on that VPN as an extra
server URL: either in the app (*Settings → Connection*, one URL per line, after the home address)
or in `extra_urls` in `server\data\config.json`, so the pairing QR codes include it. Don't
port-forward 3950 on your router: the server speaks plain HTTP, protected only by its token.

The app tries the URL that worked last; if it can't connect, it tries the others, and it starts
again from the first (home) address whenever the phone's network changes. At home you get the
LAN; on mobile data you get the VPN address, without touching anything.

## 3. Build and install the app

Requirements: JDK 17 and the Android SDK (platform 35).

```powershell
cd D:\Dev\AndroidReader\android
$env:JAVA_HOME = "D:\Dev\.toolchains\jdk-17.0.20.1+1"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

The app id is `com.androidreader`.

## 4. Pair the phone

1. On the PC, open **http://127.0.0.1:3950/** (served only to the PC itself; it contains the token).
2. Scan any QR code with the phone camera, open the link, tap **Open AndroidReader**. Each code
   carries *all* the PC's addresses (plus `extra_urls`) and the token.

Or enter them by hand in *Settings → Connection to your PC*: one URL per line (home address
first, e.g. `192.168.86.10`, then the PC's address on your VPN) and the token, then **Save &
test connection**. *Settings* also holds the default model, voice and sentence pause for books.

## 5. Reading

* **Import:** *Import EPUB* opens the system file picker, and file managers can *Open with
  AndroidReader*. Books are copied into the app, with covers.
  Long-press a book to remove it.
* **Reader:** tap any sentence to start reading from it; the sentence being read is highlighted
  and the page follows it. The list icon opens the table of contents.
* **Controls:** play/pause, previous/next sentence and chapter, also from the lock screen,
  notification and Bluetooth headsets. Unplugging headphones pauses; calls pause reading.
* **Reading options** (sliders icon): text size, theme (light/sepia/dark), speed (pitch is kept),
  sleep timer (15/30/60 min or end of chapter), the **voice model and voice for this book**,
  dramatized reading, and how many sentences to prepare ahead (default 6; raise it for slower
  models). The status line shows how many sentences are ready and when the PC is still working.
* Your position is saved per book. Sentences already synthesized are cached on the PC, so
  re-listening is instant.

## 6. Pronunciations

Long-press a word while reading → **Fix pronunciation**:

* **Say it like:** a respelling such as `her-MY-oh-nee`. Works with every model. (Stress capitals
  are lower-cased before synthesis because some models spell out capitalised syllables.)
* **Phonemes (optional):** CMU/ARPAbet such as `HH ER0 M AY1 AH0 N IY0`. Used by OmniVoice and
  IndexTTS; other models use the respelling.
* **Preview** synthesizes the sentence with the current voice and the unsaved fix.
* **Only in this book** or **in all books.** Book entries win over global ones.
* Matching is whole-word, case-insensitive unless *Match case exactly* is ticked, and covers
  possessives and plurals (*Hermione's*, *Hermiones*) and multi-word names (*Sirius Black*).

Fixes are applied on the PC before synthesis and are part of the audio cache key, so a changed
fix never reuses stale audio; sentences already buffered on the phone are re-synthesized. The
list (spell-check icon) shows, edits and deletes entries.

## 7. Dramatized reading (multi-voice, expressive)

Turn on **Dramatized reading** in a book's reading options. The phone uploads the book's text and
the quotations it found (with stable IDs) to the PC, which analyses it in the background:

1. **Deterministic segmentation** on the phone: sentences are split at quotation marks, so
   *"Stop!" she shouted.* becomes the quotation (character voice) and *she shouted.* (narrator).
   The text is never rewritten, so highlighting and positions always match the EPUB.
2. **LLM pass per ~600-word chunk** (llama.cpp, JSON-schema constrained): for each numbered
   speaker turn, the speaker and a short delivery (*shouting, urgent*), using a rolling character
   roster and the last few attributions.
3. **Roster pass:** gender, age, accent, temperament and a voice description per main character.
4. **Cast:** *Cast → Create voices* designs a voice for each main character (Qwen3-TTS
   VoiceDesign) and locks it as a cloned reference so it stays consistent; minor characters share
   a small pool of voices by gender and age. Any character can be renamed, merged with an alias
   (*Lizzy* = *Elizabeth*) or given any voice.
5. Long-press a quotation → **Change speaker or delivery**. Corrections always win over the model.

Lines are voiced as soon as their chunk is analysed. Delivery is applied by engines that support
it (Fish S2 Pro: inline tag; IndexTTS: emotion; OmniVoice: laughter/sigh only); others read the
line plainly in the character's voice. Voice and delivery are part of the cache key.

**LLM setup:** `download-llm.ps1` fetches llama.cpp and **Qwen3-30B-A3B-Instruct-2507** (Q4_K_M,
18.6 GB, Apache-2.0); `-Alternative` adds gpt-oss-20b (`"llm_model": "gpt-oss-20b-MXFP4.gguf"`,
Apache-2.0). The server starts and stops llama.cpp itself, and frees its memory when idle.

Measured on hand-labelled chapters (*Pride and Prejudice* ch. 1, *Alice* ch. 7; 182 quotations,
`scripts\eval_attribution.py`). The whole of *Alice* (27k words) was analysed on the GPU in
12.5 minutes (1,117 lines, 42 characters), and casting created 20 voices.

| LLM | Where | Speaker accuracy | Time per 100k-word book |
|---|---|---|---|
| **Qwen3-30B-A3B-Instruct-2507** Q4_K_M (default) | GPU (experts of 20 layers in RAM) | **179/182 = 98.4%** | ~65 min |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M | CPU only (16 threads) | 179/182 = 98.4% | ~4 h |
| gpt-oss-20b MXFP4 | GPU (fully on the card) | 180/182 = 98.9% | ~19 min |
| gpt-oss-20b MXFP4 | CPU only | 174/182 = 95.6% | ~4 h |

The remaining misses are an illustration caption, a chorus line ("Sh! sh!") and, for gpt-oss on
the CPU, Alice's thoughts and an untagged Bennet exchange. Qwen is the default because it is
equally accurate on CPU and GPU and writes more actable deliveries (*"dry, amused"*, *"gossipy,
animated"*, *"exasperated, reproachful"* vs gpt-oss's *"curious"*, *"annoyed"*). For reference,
BookNLP scores 56.7% on the PDNC benchmark and the 2026 state of the art (ModernBookNLP) 94.5%;
our two chapters are a small sample, not a benchmark. Two prompt details mattered: numbering
*speaker turns* (both halves of *"Yes," he said, "I will."* share a number) and having the model
echo each quote's first words. Without them labels drifted by one after an interrupted quote
(91.8%).

`"llm_mode": "auto"` (default) runs the LLM on the GPU only after two minutes without speech
requests and on the CPU otherwise, so analysis never evicts the voice model while you listen.
`"gpu"` and `"cpu"` force one or the other.

**First-person books.** Many novels (and most light novels) are told by "I". The model reports
each passage's narrator, the next passage is told who "I" was, and a point-of-view heading such as
*(The Neighbor's POV)* starts a new passage with an explicit "the narrator changes here". The
model also reports nicknames ("Makeup", "Robot Girl"), which are merged into the character they
belong to. On a hand-labelled excerpt of *Sasaki and Peeps* vol. 10 (108 quotations: Sasaki's
first-person prison scene, then the neighbor's POV with her nicknames for everyone) this raised
accuracy from 36% to 88%. The remaining misses are group banter in the neighbor's chapter; fix
them in the viewer or the app. That excerpt is copyrighted, so it lives in the git-ignored
`server\tests\gold\private\`.

**Two narrators in different tenses.** Some books switch narrator without marking the way back.
In *Sasaki and Peeps* the neighbor's sections start with *(The Neighbor's POV)*, are told in the
present tense and end at an unmarked scene break. Sasaki narrates in the past tense, and the table
of contents gives the heading as the title of chapters that actually open with Sasaki. When the
narration alternates between tenses, the server:

* finds the switches from the narration itself (dialogue removed);
* splits passages there and ignores POV chapter titles;
* tells the model who narrated earlier passages in the same tense;
* afterwards moves any "…," I said line to the narrator of its tense.

The same switches decide whose voice reads the narration. Single-narrator and third-person books
are unaffected.

On the full *Sasaki and Peeps* vol. 10, the hand-labelled excerpt spans one of the neighbor's
sections. There, wrong speakers fell from 15 to 7 (93.5%). The other 12 misses are Nick's lines
before he gives his name, which are labelled "a prisoner".

## 8. Process a book on the PC

Dramatize a book without touching the phone, then optionally put it on the phone:

```powershell
cd D:\Dev\AndroidReader\server
.\run-server.ps1                               # in another window
.\.venv\Scripts\python.exe scripts\process_book.py "E:\Light Novels\Series\Book v10.epub" --phone
```

1. The EPUB is parsed with **the app's own Kotlin parser** on the JVM (`ExportBookTool`, run through
   Gradle, so JDK 17 and the Android SDK are needed), so the book ID (SHA-1 of the file) and every
   quotation ID match what the phone produces when it imports the same file.
2. The text is uploaded and analysed (progress is printed; about an hour per 100k words on the
   GPU), then voices are created for the cast.
3. With `--phone` (phone connected over adb, app installed and paired) the EPUB is copied into the
   app and imported with dramatized reading switched on. Without it, import the same file in the
   app later: it finds the finished analysis because the IDs match.

Options: `--no-cast`, `--engine qwen3tts` (voice-design engine), `--reanalyze` (start over; your
corrections, names and voices are kept). Running it again on a finished book only does the
missing steps.

## 9. PC viewer

Open **http://127.0.0.1:3950/books** on the PC. On other devices on your network (or your VPN),
open the viewer link or scan its QR code on the PC's server page (http://127.0.0.1:3950/). That
link is `http://<PC address>:3950/books#token=…`. The browser keeps the `#` part to itself, stores
the token and removes it from the address bar. Without the link, the viewer asks for the token,
which is shown on the server page. The page is served to any address, but holds no token itself;
every book, script and audio request needs the token, like the app's.

The viewer has:

* **Library:** every analysed book with its cover, word count, analysis progress and cast status.
* **Book:** chapters in reading order; quotations coloured by speaker with the speaker's name and
  delivery (*shouting, urgent*); ✎ marks your corrections.
* **Cast panel:** characters with aliases, line counts and voices; ▶ plays the voice sample. Click a
  name to list all of that character's lines across the book (or use the speaker filter, which
  also offers *quotes given to the narrator*); search finds text.
* **Playback:** click any line or narration to play from there through the chapter with each
  speaker's voice and delivery, using the chosen narrator voice and model. Audio comes from the
  same server cache as the phone.
* **Editing:** click a speaker tag to change the speaker or delivery; ✎ on a character renames it,
  merges it into another (aliases) or changes its voice; *Create missing voices* runs the cast.
  Changes go through the same API as the app; the phone picks them up the next time you open
  the book (sentences it had already prepared are re-synthesized).

## Troubleshooting

* **Could not reach the PC:** server running? Firewall rule added (`allow-firewall.ps1`, elevated)?
  Network *Private*? Away from home: is your VPN connected, and is the PC's VPN address in the
  app's server URLs? From the phone's browser, `http://<pc>:3950/api/ping` should return JSON.
* **Phone on Wi-Fi but routing over mobile data:** for LAN addresses the app binds to the Wi-Fi
  network that owns the subnet (Samsung *Switch to mobile data*); behind a VPN it leaves routing
  to the VPN and falls back to unbound sockets if Android refuses the binding.
* **First sentence slow:** the model is loading (OmniVoice ~10 s, Qwen3-TTS ~15 s, IndexTTS ~1 min,
  Fish S2 Pro ~3 min because of `torch.compile`). Switching models also reloads.
* **Gaps while reading:** the model is slower than real time for this text. Raise *Sentences
  prepared ahead* or pick a faster model for the book.
* **Wrong voice for a character / wrong speaker:** fix it in *Cast* or long-press the line.

## API (for reference)

All `/api/*` routes except `/api/ping` need `Authorization: Bearer <token>`.

| Route | Description |
|-------|-------------|
| `GET /api/ping` | Liveness; no auth. |
| `GET /api/health` | Version, default engine, loaded engine/state, voice count, cache stats. |
| `GET /api/engines` · `POST /api/engines/{id}/load` · `POST /api/engines/unload` | Engines with capabilities, licence and measured RTF; preload; free the GPU. |
| `GET/POST /api/voices` · `PATCH/DELETE /api/voices/{id}` · `GET /api/voices/{id}/audio` · `POST /api/voices/design` | Voice store. |
| `GET/POST /api/pronunciations` · `PATCH/DELETE /api/pronunciations/{id}` | Dictionary (`?book=` for a book's view). |
| `POST /api/tts` | Body `{text, engine?, voice?, language?, speed?, delivery?, seed?, book?, sentence_pause_ms?, extra_pronunciations?, cache?}`. Chunked **PCM16LE mono**, rate in `X-Sample-Rate`; `X-Cache: hit/miss`. Errors before audio are JSON; closing the connection cancels synthesis. |
| `GET /api/books` · `GET /api/books/{id}` · `GET /api/books/{id}/cover` | Uploaded books with analysis/cast status · a book's paragraphs and quotations · its cover. |
| `POST /api/books/{id}` · `GET /api/books/{id}/script` · `POST /api/books/{id}/analyze` | Upload book text (+`analyze`, `reanalyze`, optional base64 `cover`), dramatization status and lines. |
| `PATCH /api/books/{id}/lines/{quote}` · `PATCH /api/books/{id}/characters/{id}` · `POST /api/books/{id}/cast` | Corrections, rename/merge/voice, create character voices. |
| `GET /pair` · `GET /` · `GET /books[/{id}]` | Pairing page (token in the URL fragment) · QR codes for pairing and the viewer (PC only) · book viewer (any address; asks for the token unless opened on the PC). |

## Development

```powershell
# server tests (fake engine in a real worker process; no GPU needed)
cd server; .\.venv\Scripts\python.exe -m pytest

# engine benchmark (real models) and attribution accuracy (real LLM)
.\.venv\Scripts\python.exe scripts\bench_engines.py omnivoice fish_s2 indextts qwen3tts
.\.venv\Scripts\python.exe scripts\eval_attribution.py tests\gold\pride-ch1.json tests\gold\alice-ch7.json

# Android unit tests (EPUB parsing, sentence/quote segmentation, URLs, language names)
cd android; .\gradlew.bat testDebugUnitTest

# On-device tests (phone paired, server running). Installed manually so the app isn't
# uninstalled afterwards. Without the firewall rule, `adb reverse tcp:3950 tcp:3950` and the URL
# http://127.0.0.1:3950 work over USB.
.\gradlew.bat assembleDebugAndroidTest
adb install -r -t app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk
adb shell am instrument -w -e class com.androidreader.ReaderPlaybackTest com.androidreader.test/androidx.test.runner.AndroidJUnitRunner
# slow: analyses the whole of Alice, casts voices, then plays the Mad Tea-Party dramatized
adb shell am instrument -w -e class com.androidreader.DramaPlaybackTest#analysesCastsAndReadsWithCharacterVoices com.androidreader.test/androidx.test.runner.AndroidJUnitRunner
# a book processed with process_book.py --phone: narrator voices per section, character voices in a scene
adb shell am instrument -w -e class com.androidreader.DramaPlaybackTest#readsAProcessedBook -e bookId 2f145f7d38c07f00 -e mainAt 3:1 -e povAt 12:170 -e dialogueAt 10:12 com.androidreader.test/androidx.test.runner.AndroidJUnitRunner
```
