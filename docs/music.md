# The music score (stage 4)

Stage 4 scores the clip with a generated classical string quartet. A preset tied to the narration style fixes the musical frame (tempo range, major or minor, playing techniques). Qwen2.5 then adjusts the mood, texture and key within that frame, and Python assembles the final caption. [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), which lives in `MUSIC GEN/`, renders the audio as a separate local server. The design is explained in [MUSIC_PLAN.md](MUSIC_PLAN.md).

The music comes out pre-leveled to sit under the narration (9 LU below it by default), but it is not mixed into the video for you. Drop the voiceover and the music into your editor together.

## Running it: two terminal windows

**Terminal 1:** start the server and leave it open.

```bash
"MUSIC GEN/start_api_server.command"
```

The first launch loads about 5 GB of weights. Wait until port 8001 answers (`curl http://127.0.0.1:8001/health`).

**Terminal 2:** run the pipeline or batch as usual. `batch.py` scores by default; with `pipeline.py` you add `--score`.

```bash
# Voiceover and quartet score
python pipeline.py data/scratch/zoomies.mov --narrate --score

# Three takes to audition, and keep the caption for reference
python pipeline.py data/scratch/zoomies.mov --narrate --score \
    --music-takes 3 --save-music-prompt

# Music only, no narration
python pipeline.py data/scratch/zoomies.mov --score
```

If the server is not up, the pipeline says so and stops immediately, instead of spending three minutes on stages 1 to 3 first.

To install ACE-Step the first time, follow [MUSIC GEN/README.md](../MUSIC%20GEN/README.md).

## Options

| Option | Default | Notes |
| --- | --- | --- |
| `--music-takes N` | 1 (3 in `batch.py`) | 1 to 8. Each take gets its own accent, tempo and key so they really sound different, with one generation per take. With more than one take the files are numbered: `<name>.music.1.wav`, `<name>.music.2.wav` and so on. |
| `--music-bpm N` | preset range | Sets the tempo outright, ignoring both the preset and the language model. |
| `--no-music-vary` | off | Sends one batched request instead of one per take. It is faster, but the takes come back nearly identical. |
| `--music-style` | follows `--style` | Picks a musical frame that differs from the narration style. |
| `--music-lufs` | voiceover − 9 LU | An explicit loudness target for the music. It falls back to −25 LUFS when there is no voiceover. The music is meant to sit well under the narration; `-24` puts it about 5 LU under, and `-22` about 3 LU under, where it starts competing with the voice. |
| `--music-bit-depth` | 24 | 16, 24 or 32. The music sits about 20 dB below full scale, so 16-bit leaves only about 12 bits in use and adds audible grit to sustained strings. |
| `--music-seed N` | random | Fix it while tuning one setting so you hear only that setting's effect. |
| `--music-thinking` | off | Lets ACE-Step's 5Hz language model plan the arrangement. Better musicality, more memory. |
| `--no-music-llm` | off | Skips stage 2b and scores from the bare preset. Fast and predictable. |
| `--save-music-prompt` | off | Saves the assembled caption and settings as a `.json` next to the music. |
| `--music-server URL` | `http://127.0.0.1:8001` | Where the ACE-Step server is listening. |
| `--music-timeout` | 300 | Seconds to wait before giving up. |

## Why the server runs separately

ACE-Step has its own virtualenv with its own pinned versions of torch and MLX. Merged into this project's virtualenv, those would clash with Kokoro and Chatterbox, so the two talk over HTTP instead. Keeping the server running also means the model loads once instead of once per clip.

## Memory

The pipeline releases Ollama's models before stage 4 so ACE-Step has room. That is also why the music caption (stage 2b) is written right after stage 2, while Qwen is still loaded, rather than just before stage 4.

## Experiments

The early hand-written caption test, `phase1_quartet_test.sh`, and its recordings are in `review/`.
