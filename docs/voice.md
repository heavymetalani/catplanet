# The voiceover (stage 3)

Stage 3 reads the script aloud and writes a `.wav`. It supports two engines.

- **Kokoro** (the default; 82M parameters; Apache 2.0) uses voice presets and cannot clone a voice. It has a British male documentary voice out of the box and is fast and simple.
- **Chatterbox** (500M parameters; MIT license) clones a voice from a 5 to 10 second reference clip (`.wav`, `.mp3`, `.m4a` and so on) and gives explicit control over emotion.

`batch.py` uses Chatterbox and clones the clip named by `reference_audio` in `config/local_settings.json`. Keep your reference clips in `data/voices/`.

## Kokoro vs Chatterbox: which to pick

Pick **Kokoro** when you want fast iteration, don't need a specific voice, and are happy with a well-tuned British documentary preset. It is great for testing many scripts quickly.

Pick **Chatterbox** when you want a specific voice (yours, a friend's, or a royalty-free sample) that you will use consistently across a series. Voice consistency is half the magic of a recurring format.

## Kokoro voices

These presets suit documentary narration best:

- `bm_george`: warm and mature. This is the default and the closest to the documentary archetype.
- `bm_lewis`: slightly younger and more conversational.
- `bm_daniel`: crisper, with a faster cadence.
- `bm_fable`: storyteller-like, good for longer narration.

The full catalog is at [huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Kokoro also supports American English (`am_*` and `af_*`), French, Hindi, Japanese and other languages; the voice prefix picks the language automatically.

`--speed 0.95` slows the voice to about 95% for more gravitas, and `--speed 1.1` speeds it up for short-form energy. Kokoro outputs 24 kHz mono WAV.

```bash
python -m catplanet.generate_voice --script s.txt --out out.wav \
    --engine kokoro --voice bm_lewis --speed 0.95
```

## Chatterbox and voice cloning

Chatterbox ships with a default voice, so you don't need a reference clip, but cloning is usually the reason to choose it.

To clone a voice, supply a clean 5 to 10 second clip of the target voice speaking at a natural pace, ideally with no music or background noise. Put it in `data/voices/` and pass it with `--reference-audio data/voices/my_narrator.wav` (or set it as `reference_audio` in your settings for batch runs).

The path is handed straight to Chatterbox, which decodes it with torchaudio, so `.wav`, `.mp3`, `.m4a` and other common formats all work. `.wav` is the safest choice if a clip fails to load.

Good sources of reference audio:

- Record yourself doing a documentary-narrator voice. Ten seconds of reading a passage is enough.
- Ask a friend with a suitable voice to record a clip.
- Use a royalty-free sample from [freesound.org](https://freesound.org/) (check the license).
- Use a public-domain audiobook recording from LibriVox or archive.org. Dickens read by a British narrator works well.

**Do not clone real celebrities or public figures without permission.** Voice-likeness law in the UK and US is a grey area, and David Attenborough has publicly objected to unauthorized clones of his voice. Capture the style through your own source audio rather than impersonating a specific person.

### Chatterbox options

- `--exaggeration 0.3–0.7` (default 0.5). Lower is flatter and more deadpan, which suits the dry Attenborough style. Higher is more emotive, which suits sports or reality TV.
- `--cfg-weight 0.3–0.7` (default 0.5). Lower gives more natural pacing and breathing. Higher sticks more tightly to the text and sounds crisper.
- `--device mps|cpu|cuda` is detected automatically. MPS is fastest on the M4; use `--device cpu` if you hit a Metal operation that isn't supported.
- `--max-chars 250` (default 250) controls chunking, explained below. Set it to `0` to turn chunking off.

### Why long scripts are split into chunks

Chatterbox produces garbled, phoneme-salad output when the input is longer than it can reliably handle. Single sentences sound fine, but a full 30-second script sent in one call often ends in unintelligible noise.

To prevent this, any script longer than `--max-chars` is split at paragraph breaks first, then at sentence boundaries if a paragraph is still too long, and at word boundaries only as a last resort. Each chunk is synthesized separately and joined with a short silence that matches the break (0.4 seconds between paragraphs, 0.15 seconds between sentences), which also reinforces the Attenborough cadence. Kokoro does not need this, because it handles long inputs internally.

If a chunk still comes out garbled, lower `--max-chars` to 180 or 150. If the joins sound abrupt, raise it to 350. `--max-chars 0` sends the whole script in one call, which is useful for reproducing the original problem when debugging.

The first run downloads about 1 GB of model weights from Hugging Face. A 70-word clip takes roughly 10 to 20 seconds on the M4 with MPS, which is about ten times slower than Kokoro. Time grows in proportion to the number of chunks.

## The Attenborough tuning recipe (Chatterbox with a cloned voice)

Three things combine to produce the classic Attenborough cadence. In order of impact:

**1. The reference clip is the biggest lever.** Chatterbox copies pacing, pause length and intonation directly from the reference. The clip itself should have the measured, pausing, soft-spoken delivery you want. 8 to 15 seconds is ideal. A fast or flat reference will never come out slow or dramatic, whatever the settings.

**2. Punctuation in the script does most of the work.** Chatterbox reads punctuation literally:

- `,` is a short pause.
- `.` is a sentence-ending pause with a breath.
- `...` is a dramatic pause of about one beat. Use it before punchlines and reveals.
- ` — ` is a mid-thought pause, shorter than an ellipsis.
- A paragraph break is a longer breath.

The `attenborough` style prompt already asks the text model for text shaped like this. If you write a script by hand, follow the same conventions.

**3. The settings.** These are the values `batch.py` uses:

```bash
python pipeline.py data/scratch/your_clip.mov --narrate \
  --engine chatterbox \
  --reference-audio data/voices/your_sample.wav \
  --exaggeration 0.3 \
  --cfg-weight 0.35 \
  --style attenborough
```

If the result is too flat, nudge `--exaggeration` up toward 0.45. If it rushes through the text and ignores punctuation, drop `--cfg-weight` to 0.25. That sounds backwards, but a lower value lets the model use more of its own pacing judgment instead of cramming the text.

## Iterating on the voice quickly

Stage 3 is the slow step, so tune it without re-running the language models:

```bash
# Once: produce and save the script
python pipeline.py data/scratch/clip.mov --save-script

# Many times: re-voice only that script
python -m catplanet.generate_voice --script data/scratch/clip.script.txt \
  --out data/scratch/takes/take_01.wav --engine chatterbox \
  --reference-audio data/voices/narrator.wav --exaggeration 0.30 --cfg-weight 0.35

python -m catplanet.generate_voice --script data/scratch/clip.script.txt \
  --out data/scratch/takes/take_02.wav --engine chatterbox \
  --reference-audio data/voices/narrator.wav --exaggeration 0.40 --cfg-weight 0.25
```

Line the takes up in your editor and pick the best one.

To re-voice a script on the CPU instead of the GPU:

```bash
python -m catplanet.generate_voice \
  --script path/to/script.txt \
  --out path/to/out.wav \
  --engine chatterbox \
  --reference-audio data/voices/narrator.wav \
  --exaggeration 0.3 \
  --cfg-weight 0.35 \
  --device cpu
```
