# Cat Video Interpreter

Use MiniCPM-V 4.5 (an 8B open-source multimodal LLM) running locally on your Mac to describe cat videos, then feed those descriptions to a text LLM to generate funny Animal-Planet-style voiceovers for short-form content.

Hardware target: Mac Mini M4, 24 GB unified memory.

## Architecture

Stage 1: `describe_video.py` — extracts frames from a video with ffmpeg and sends them to MiniCPM-V 4.5 via Ollama. Returns a detailed description of what is happening in the clip. Reads the `cats` list in `local_settings.json` (gitignored) to identify your cats by name in the output.

Stage 2: `generate_script.py` — takes a description and writes a funny voiceover in a chosen style (attenborough / sports / noir / reality_tv) using Qwen2.5-14B.

Stage 3: `generate_voice.py` — synthesizes the script to a `.wav`. Supports two engines:
  - **Kokoro** (default, 82M params, Apache 2.0) — voice presets, no cloning, British male documentary voice out of the box. Fast and simple.
  - **Chatterbox** (500M params, MIT license) — voice cloning from a 5–10 second reference clip (`.wav`, `.mp3`, `.m4a`, …), plus explicit emotion/exaggeration control.

Stage 4: `generate_music.py` — scores the clip with a generated classical string quartet, written to `<name>.music.wav`. A preset keyed to the voiceover style fixes the musical frame (tempo band, mode, technique vocabulary); Qwen2.5 then adjusts mood, texture and key *within* that frame and Python assembles the caption. Rendered by [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) in `MUSIC GEN/`, which runs as a separate local server. See `MUSIC_PLAN.md` for the design.

Batch: `batch.py` — runs `pipeline.py` over every video in `video_samples/`, numbering them 1, 2, 3… across runs and filing the outputs into `voice_samples/` and `music_samples/`.

Pipeline: `pipeline.py` — chains the stages in one call. Auto-detects video length via ffprobe so the target word count matches the clip. Stage 3 runs when you pass `--narrate`, stage 4 when you pass `--score`.

The music stem is delivered pre-leveled to sit under the narration (9 LU below it by default) but is **not** mixed or muxed for you — drop both `.wav`s into your editor over the video.

## One-time setup

Run the install script. It will install Homebrew dependencies, set up a Python virtualenv, and pull MiniCPM-V 4.5 (~5 GB download).

```bash
git clone https://github.com/heavymetalani/catplanet.git
cd catplanet
bash install.sh
```

Then create your personal settings file. It holds the voice to clone and your cats' names, and it is gitignored, so it never gets committed:

```bash
cp local_settings.example.json local_settings.json
```

Edit it with your own cats and the path to a 5–10 second voice clip you put in `voices/`. Every field is optional — with no file at all, the pipeline uses Chatterbox's default voice and describes cats without names.

For music (stage 4), ACE-Step is downloaded separately; see `MUSIC GEN/README.md`.

None of the input or output folders (`voices/`, `video_samples/`, `voice_samples/`, `music_samples/`, `samples/`) are in git. The pipeline creates the output folders on first use; create `voices/` and `video_samples/` yourself.

The script is safe to re-run — every step checks whether it's already done. If you prefer to run each step manually, here's what it does:

1. Install Homebrew if missing, then `brew install ollama ffmpeg python@3.12 espeak-ng`.
2. Start the Ollama background service (`brew services start ollama`) unless it's already running.
3. Wait up to 15 seconds for the daemon to answer at `http://localhost:11434/api/tags`, and bail out with a hint to run `ollama serve` if it never does.
4. Pull the vision model: `ollama pull openbmb/minicpm-v4.5` (~5 GB).
5. Pull the text model: `ollama pull qwen2.5:14b` (~9 GB).
6. Create a virtualenv in `.venv` using Python 3.12 specifically — Kokoro requires >=3.10,<3.13, so an existing `.venv` built against any other version is deleted and rebuilt. Then install dependencies from `requirements.txt`. Kokoro model weights (~300 MB) are pulled automatically on first TTS run.
7. Create the `samples/` and `descriptions/` directories if they don't exist.

## Running

Activate the venv once per shell session:

```bash
source .venv/bin/activate
```

### Batch processing (recommended for more than one clip)

Drop videos into `video_samples/` and run:

```bash
python batch.py
```

Each video is numbered, processed, and filed away. Numbering continues across runs — if the last batch ended at 6, the next starts at 7.

```
video_samples/              <- drop new videos here (this is the queue)
video_samples/processed/    <- 7.mov, plus 7.script.txt
voice_samples/              <- 7.wav
music_samples/              <- 7.1.wav, 7.2.wav, 7.3.wav
batch_state.json            <- version counter + a log of every run
```

Videos move to `processed/` only after they succeed, so `video_samples/` always shows exactly what's still outstanding. A clip that fails keeps its number and stays in the queue — re-running retries it with the *same* number rather than burning a new one, because a file already named as a bare number is recognized as versioned.

That last rule has a guard. If you drop in a video that merely *happens* to be called `1.mov`, reusing number 1 would overwrite the real 1's voiceover and music. So a bare-numbered file only reuses its number when that number has **no finished outputs**; otherwise it's treated as new and warns you. Retries keep their number, unrelated files can't clobber your work.

Two other guards run before anything is renamed:

- Every argument is validated against `pipeline.py`'s real parser up front. A typo like `--musictakes` fails immediately instead of being discovered once per clip, after the whole queue has been renamed and failed.
- `--audio-out` and `--music-out` are rejected — `batch.py` owns those, and passing them would silently do nothing. Use `--voice-dir` / `--music-dir`, or run `pipeline.py` directly for a one-off.

**Don't delete `batch_state.json`.** It holds the counter; losing it restarts numbering at 1 and overwrites your existing outputs. It is gitignored, because it lists your original video filenames, so it exists only on your machine — back it up along with your output folders.

The per-clip settings live in `DEFAULT_PIPELINE_ARGS` at the top of `batch.py`. The one exception is the voice Chatterbox clones: it is personal, so it comes from `reference_audio` in `local_settings.json` instead. Without it, Chatterbox narrates in its built-in default voice. Anything `batch.py` doesn't recognize is forwarded to `pipeline.py` and appended after those defaults, so it wins:

```bash
python batch.py --dry-run              # show the plan, touch nothing (safe, uses no numbers)
python batch.py --music-takes 1        # override a default for this run
python batch.py --limit 2              # just the first two in the queue
python batch.py --style noir
```

Stage 4 needs the ACE-Step server running (see below). `batch.py` checks for it before renaming anything, so a forgotten server doesn't leave half your queue in a half-versioned state.

### Full pipeline, one clip at a time

Video in, narrated `.wav` out:

```bash
# Script only (prints to stdout)
python pipeline.py samples/zoomies.mov

# Script + narrated voiceover audio
python pipeline.py samples/zoomies.mov --narrate

# Different style, different voice, save everything for later iteration
python pipeline.py samples/zoomies.mov --style noir --narrate \
    --save-description --save-script
```

Clip length is auto-detected via ffprobe and drives the target word count (~2.5 words per second of video, so a 28-second clip gets ~70 words of narration). Pass `--length N` to override.

Output files (all optional, written next to your video):

- `--save-description` → `<name>.description.txt` (stage 1 output, reuse for iterating stage 2 fast)
- `--save-script` → `<name>.script.txt` (stage 2 output, reuse for iterating stage 3 fast)
- `--narrate` → `<name>.voiceover.wav` (stage 3 output, import into your video editor)
  - Pauses between sentences are capped at 0.35s by default. The attenborough prompt asks for ellipses and paragraph breaks as dramatic beats, and the TTS engines honor every one — which reads as dead air in a short clip. Tighten with `--max-pause 0.25`, or pass `--max-pause 0` to keep the engine's own timing.
  - Normalized to −16 LUFS (`--target-lufs`), the usual podcast/short-form level. Neither engine normalizes on its own — Kokoro writes at about −24 LUFS. This matters beyond the voice: the music bed is leveled *relative* to this file, so a quiet narration drags the music down with it. Speech is peaky, so the true-peak guard often lands a little short of the target rather than clip; that's expected. `--target-lufs 0` skips it.
- `--score` → `<name>.music.wav` (stage 4 output, the string quartet bed)

Use `--audio-out path.wav` to override the audio destination, `--music-out path.wav` for the music.

### Adding music (`--score`)

Stage 4 needs the ACE-Step server running. **Two terminals.**

Terminal 1 — start the server and leave it open:

```bash
"MUSIC GEN/start_api_server.command"
```

First launch loads ~5 GB of weights. Wait for port 8001 to answer (`curl http://127.0.0.1:8001/health`).

Terminal 2 — run the pipeline as usual, plus `--score`:

```bash
# Voiceover + quartet score
python pipeline.py samples/zoomies.mov --narrate --score

# Three takes to audition, and keep the caption for reference
python pipeline.py samples/zoomies.mov --narrate --score \
    --music-takes 3 --save-music-prompt

# Music only, no narration
python pipeline.py samples/zoomies.mov --score
```

If the server isn't up, the pipeline says so and exits immediately rather than burning three minutes on stages 1–3 first.

| Flag | Default | Notes |
| --- | --- | --- |
| `--music-takes N` | 1 | 1–8. Each take gets its own accent, tempo and key so they're actually distinguishable — one generation per take. |
| `--music-bpm N` | preset range | Override the tempo outright, ignoring both the preset and the LLM. |
| `--no-music-vary` | off | One batched request instead of one per take. Faster, but the takes come back nearly identical. |
| `--music-style` | follows `--style` | Pick a musical frame that differs from the narration style. |
| `--music-lufs` | voiceover − 9 LU | Explicit loudness target for the bed. Falls back to −25 LUFS with no voiceover. A bed is meant to sit well under the narration; if you want it more present, `-24` puts it ~5 LU under, `-22` about 3 LU under and it will start competing with the voice. |
| `--music-bit-depth` | 24 | 16, 24 or 32. A bed sits ~20 dB below full scale, so 16-bit leaves only ~12 bits in use and adds audible grit to sustained strings. |
| `--music-seed N` | random | Fix it when tuning one variable so you hear that variable's real effect. |
| `--music-thinking` | off | Let ACE-Step's 5Hz LM plan the arrangement. Better musicality, more memory. |
| `--no-music-llm` | off | Skip stage 2b and score from the bare preset. Fast and deterministic. |
| `--music-timeout` | 300 | Seconds to wait before giving up. |

Stage 4 can also run standalone against a saved description, which is the fast way to iterate on music without re-running the vision model:

```bash
python generate_music.py --description samples/five.description.txt \
    --video samples/five.mov --style noir --takes 3 \
    --voiceover samples/five.voiceover.wav --out samples/five.music.wav

# See the caption without generating anything
python generate_music.py --description d.txt --length 18 --print-caption
```

**Why two terminals.** ACE-Step has its own venv with its own pinned torch/MLX that would fight kokoro and chatterbox if merged into this project's venv, so the two talk over HTTP instead. Keeping the server resident also means the model loads once rather than once per clip.

**Memory.** The pipeline releases Ollama's models before stage 4 so ACE-Step gets the room — that's also why the music caption (stage 2b) is generated right after stage 2, while Qwen is already loaded, rather than next to stage 4.

### Identifying your cats (`local_settings.json`)

Stage 1 reads the `cats` list in `local_settings.json` to know who's who. That file is gitignored, so your cats' names stay on your machine. The committed template, `local_settings.example.json`, shows the format with placeholder cats:

```json
{
  "cats": [
    {
      "name": "Whiskers",
      "pronouns": "she/her",
      "appearance": "orange tabby with a white chest and white front paws..."
    },
    {
      "name": "Shadow",
      "pronouns": "he/him",
      "appearance": "solid grey cat with no markings and green eyes..."
    }
  ]
}
```

The vision model is instructed to use these names based on coat pattern, distinguish them when both appear in the same frame, and fall back to "one of the cats" when it can't tell (partial view, bad angle). Since names flow through into the description text, stages 2 and 3 pick them up automatically — your voiceover will address your cats by name.

To edit: open `local_settings.json`, change the names / pronouns / appearance, save. No code changes needed. To add another cat, add another object to the `cats` array. To disable identification entirely, remove the `cats` list — stage 1 falls back to a generic prompt.

**Tips for good identification:**

- Describe the coat as concretely as possible (colors, markings, tail length, size). The more unique the description, the more reliably the model picks the right name.
- If the two cats have very similar builds, focus on the most distinctive differentiator first (e.g., "white chest and paws" vs "no white anywhere").
- Preview the prompt the model will actually see: `python describe_video.py --show-prompt`.
- Point at an alternate file with `--cats-file path/to/other.json` — useful if you ever shoot a clip with a friend's cat.

### Stage 1 only — describe a video

```bash
python describe_video.py samples/cat_clip.mov
python describe_video.py samples/cat_clip.mov > descriptions/cat_clip.txt
```

Useful flags:

- `--fps 2` — frames per second to sample (default 2). Bump to 4 for fast action (zoomies, pouncing), drop to 1 for slow footage (sleeping, grooming).
- `--max-frames 24` — cap on frames sent to the model. 16–24 is the sweet spot for quality vs. latency.
- `--prompt "..."` — override the default description prompt.

### Stage 2 only — write a script from an existing description

```bash
# From a file
python generate_script.py --description descriptions/zoomies.txt --length 28

# From stdin (chain the stages manually)
python describe_video.py samples/zoomies.mov \
  | python generate_script.py --length 28 --style attenborough
```

Styles: `attenborough` (default, dry documentary), `sports` (excitable play-by-play), `noir` (hardboiled detective), `reality_tv` (dramatic confessional). Iterate on these by editing the prompts at the top of `generate_script.py`.

Knobs: `--temperature 0.9` (raise to 1.1 for wilder output, drop to 0.7 for safer), `--wps 2.5` (target speaking pace in words/sec), `--model qwen2.5:14b` (swap the text model).

### Stage 3 only — synthesize audio from an existing script

```bash
# Kokoro (default)
python generate_voice.py --script samples/zoomies.script.txt --out audio/zoomies.wav

# From stdin
python generate_script.py --description d.txt --length 28 \
    | python generate_voice.py --out audio/zoomies.wav

# Kokoro with a different voice and slower pace for gravitas
python generate_voice.py --script s.txt --out out.wav \
    --engine kokoro --voice bm_lewis --speed 0.95

# Chatterbox with its default voice
python generate_voice.py --script s.txt --out out.wav --engine chatterbox

# Chatterbox cloning a specific voice from a reference recording
python generate_voice.py --script s.txt --out out.wav \
    --engine chatterbox --reference-audio voices/my_narrator.wav \
    --exaggeration 0.6
```

#### Kokoro voices (engine=kokoro)

Best fit for documentary narration:

- `bm_george` — warm, mature (default — closest to the documentary archetype)
- `bm_lewis` — slightly younger, more conversational
- `bm_daniel` — crisper, faster cadence
- `bm_fable` — storyteller-ish, good for longer narration

Full voice catalog: [huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Kokoro also supports American English (`am_*` / `af_*`), French, Hindi, Japanese, and others — voice prefix determines the language automatically.

Kokoro knobs: `--speed 0.95` slows to ~95% for more gravitas, `--speed 1.1` speeds up for short-form energy. Kokoro outputs 24 kHz mono WAV.

#### Chatterbox (engine=chatterbox)

Chatterbox clones voices from short audio samples. It ships with a default voice, so you don't *need* reference audio — but the whole point of picking Chatterbox over Kokoro is usually voice cloning.

To clone a voice, record or supply a **5–10 second clean audio clip** of the target voice saying anything at natural pace (ideally no music or noise). Put it somewhere like `voices/my_narrator.wav` and pass `--reference-audio voices/my_narrator.wav`.

The path is handed straight to Chatterbox, which decodes it with torchaudio — so `.wav`, `.mp3`, `.m4a`, and other common formats all work, and the existing files in `voices/` are a mix of them. `.wav` is still the safest bet if a clip fails to load.

Reasonable sources of reference audio:

- Record yourself putting on a documentary-narrator voice. 10 seconds of you reading a passage is enough.
- Ask a friend with a suitable voice to record a clip.
- Grab a royalty-free sample from [freesound.org](https://freesound.org/) (check license).
- Public-domain audiobook recordings (LibriVox, archive.org) — Dickens read by a British narrator works well.

**Avoid cloning real celebrities / public figures without permission.** UK/US voice-likeness law is a grey area, and creators like David Attenborough have publicly objected to unauthorized clones of their voice. Capture the *style* via your chosen source audio; don't impersonate someone specific.

Chatterbox knobs:

- `--exaggeration 0.3–0.7` — default 0.5. Lower = flatter, more deadpan (good for dry Attenborough-style). Higher = more emotive (good for sports or reality_tv styles).
- `--cfg-weight 0.3–0.7` — default 0.5. Lower = more natural pacing and breathing. Higher = tighter adherence to the text, crisper delivery.
- `--device mps|cpu|cuda` — auto-detects. MPS is fastest on your M4; use `--device cpu` if you hit a Metal op that isn't supported.
- `--max-chars 250` — default 250. Long scripts are split into chunks at paragraph, then sentence boundaries and synthesized one at a time, then concatenated with natural silence between them (0.4s between paragraphs, 0.15s between sentences). See **Chunking** below. Set to `0` to disable chunking entirely.

**Chunking (Chatterbox only).** Chatterbox generates garbled, phoneme-salad output when the input exceeds its reliable context length — sentences sound fine in isolation but a full 30-second script synthesized in one call often ends in unintelligible noise. The pipeline mitigates this by splitting any script longer than `--max-chars` (default 250) along paragraph breaks first, then sentence boundaries if a single paragraph is still too long, and finally word boundaries as a last resort. Each chunk is synthesized separately and stitched together with a short silence that matches the break kind (paragraph vs. sentence), which conveniently reinforces the Attenborough cadence. Kokoro doesn't need this — it streams internally and handles long inputs cleanly.

If a chunk still comes out garbled, lower `--max-chars` (try 180 or 150). If chunk boundaries sound abrupt, raise it (try 350). Set `--max-chars 0` to send the whole script as one call (useful for debugging — this is how you'll reproduce the original gibberish if it returns).

First run will download ~1 GB of model weights from Hugging Face. Synthesis of a 70-word clip on M4 with MPS takes roughly 10–20 seconds (Chatterbox is ~10× slower than Kokoro in exchange for voice cloning and better emotional range). Chunked long scripts scale linearly with chunk count.

#### Attenborough-style tuning recipe (Chatterbox + cloned voice)

Three layers compound to produce the classic Attenborough cadence. In order of impact:

**1. Reference audio is the biggest lever.** Chatterbox inherits prosody — pacing, pause length, intonation — directly from your reference sample. Your reference clip should itself exhibit the measured, pausy, soft-spoken delivery you want in the output. 8–15 seconds is the sweet spot. A reference that's fast or flat will never clone as slow or dramatic, no matter the parameters.

**2. Punctuation in the script does 60% of the work.** Chatterbox honors punctuation literally:

- `,` short pause
- `.` sentence-final pause + breath
- `...` (ellipsis) dramatic pause of about one beat — use these before punchlines and revelations
- ` — ` (em dash) mid-thought pause, shorter than ellipsis
- paragraph break = longer breath

The `attenborough` style prompt already tells the LLM to produce text shaped this way. If you ever write your own scripts by hand for this engine, follow the same conventions.

**3. Parameter recipe for Chatterbox:**

```bash
python pipeline.py samples/your_clip.mov --narrate \
  --engine chatterbox \
  --reference-audio voices/your_sample.wav \
  --exaggeration 0.3 \
  --cfg-weight 0.35 \
  --style attenborough
```

If the output is too flat, nudge `--exaggeration` up toward 0.45. If it rushes through the text and ignores punctuation, drop `--cfg-weight` to 0.25 (counter-intuitive — lower cfg lets the model use more of its own pacing judgment rather than cramming the text tightly).

**Iteration workflow.** Stage 3 is the slow step (10–20s per take with Chatterbox). To tune without re-running the LLMs:

```bash
# Once: produce and cache the script
python pipeline.py samples/clip.mov --save-script

# Many times: iterate on the voice only (fast)
python generate_voice.py --script samples/clip.script.txt \
  --out takes/take_01.wav --engine chatterbox \
  --reference-audio voices/narrator.wav --exaggeration 0.30 --cfg-weight 0.35

python generate_voice.py --script samples/clip.script.txt \
  --out takes/take_02.wav --engine chatterbox \
  --reference-audio voices/narrator.wav --exaggeration 0.40 --cfg-weight 0.25
```

Line them up in your editor, pick the best take.

#### Kokoro vs Chatterbox — which to pick

Pick **Kokoro** when you want fast iteration, don't need a specific voice, and are happy with a well-tuned British documentary preset. Great for testing lots of scripts quickly.

Pick **Chatterbox** when you want a *specific* voice (yours, a friend's, or a royalty-free sample) that you'll use consistently across a series of videos — voice consistency is half the magic of a recurring format.

## First-time sanity check

Put any short cat clip in the `samples/` folder and run:

```bash
python describe_video.py samples/<your_clip>.mp4
```

Expected: a paragraph describing what the cat is doing, posture, environment, and any notable moments. First run will be slower (~15–30 s) as the model loads into memory. Subsequent runs take 3–8 seconds for a 10-second clip at fps=2.

## Troubleshooting

**"connection refused" / Ollama not running** — start it with `brew services start ollama` or just run `ollama serve` in another terminal.

**Model download is slow** — it's 5 GB. Let it finish; it's a one-time thing.

**Out of memory** — close other heavy apps (Chrome, Xcode). With MiniCPM-V 4.5 quantized at int4 plus ~16 frames, you should stay well under 10 GB of memory used by the model itself.

**Descriptions are too generic** — edit the prompt in `describe_video.py`. The more specific you are about what to look for (tail position, eye dilation, interactions with objects), the more useful the output becomes as input for script-writing.

**`espeak-ng` not found** — Kokoro needs it for phoneme generation. Install with `brew install espeak-ng` and retry.

**Kokoro first run is slow / "Downloading…"** — the 82M model weights (~300 MB) are pulled from Hugging Face the first time. Subsequent runs are cached.

**Voice sounds robotic or clipped** — try `--speed 0.95` and/or a different voice preset. Kokoro occasionally trips on long run-on sentences; shorter, punchier scripts render more naturally.

**Chatterbox MPS error / Metal op not implemented** — known issue with certain torch ops on Apple Silicon. Fall back with `--device cpu`. Slower but reliable.

**Chatterbox cloned voice doesn't sound like the reference** — make sure the reference is clean (no music, no overlapping speech), 5–10 seconds long, and records natural pacing. Very short clips (<3s) or noisy clips produce poor clones. Try bumping `--cfg-weight` to 0.6 for tighter voice adherence. If the clip fails to load at all, re-encode it to `.wav` (`ffmpeg -i in.m4a out.wav`).

**Chatterbox output is gibberish / phoneme salad on longer scripts** — this is a context-length issue. Short sentences synthesize fine but longer scripts produce unintelligible noise. The pipeline auto-chunks Chatterbox input at paragraph/sentence boundaries (default `--max-chars 250`). If chunks still garble, lower `--max-chars` (e.g. 180). Kokoro is immune to this — switch engines with `--engine kokoro` if you need a fast sanity check.

## What's next

Possible upgrades once the basic pipeline feels good:

- **Mix and mux**: a stage 5 that combines video + voiceover + music into a finished `.mp4`, with sidechain ducking so the music dips under the narration instead of sitting at a fixed level. Deliberately left out for now — see `MUSIC_PLAN.md`.
- **A quartet LoKr**: if ACE-Step keeps sneaking non-string instruments into cues, training a style adapter on real quartet recordings is the durable fix. `MUSIC GEN/instructions.md` §3 has the procedure.
- **Timestamp-aware descriptions**: tweak the stage-1 prompt to emit timestamps, so stage 2 can produce a script with suggested cut points.
- **Native video path**: swap the Ollama frame-sampling path for the Python `transformers` library, which exposes MiniCPM-V 4.5's 3D resampler for true video encoding. Slower setup, but can perceive motion more fluidly.

## Example commands

```bash
# Chatterbox, cloning a voice clip you put in voices/
python pipeline.py samples/your_clip.mov --narrate \
  --engine chatterbox \
  --reference-audio voices/narrator.wav \
  --exaggeration 0.4

# Kokoro, no reference clip needed
python pipeline.py samples/your_clip.mov --narrate --engine kokoro

# The settings batch.py uses, for a single clip
python pipeline.py samples/your_clip.mov --narrate \
  --engine chatterbox \
  --reference-audio voices/narrator.wav \
  --exaggeration 0.3 \
  --cfg-weight 0.35 \
  --style attenborough \
  --save-script

# Re-voice an existing script on CPU
python generate_voice.py \
  --script path/to/script.txt \
  --out path/to/out.wav \
  --engine chatterbox \
  --reference-audio voices/narrator.wav \
  --exaggeration 0.3 \
  --cfg-weight 0.35 \
  --device cpu
```