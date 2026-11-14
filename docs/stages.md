# Running the stages one at a time

`pipeline.py` chains every stage for you, but each stage can also run on its own. This is the fast way to iterate: save the output of a slow stage once, then re-run only the step you are tuning.

Run these from the project root with the virtualenv active. The stage modules live in the `catplanet/` package, so you call them with `python -m catplanet.<stage>`.

## The full pipeline for one clip

```bash
# Script only (printed to the terminal)
python pipeline.py data/scratch/zoomies.mov

# Script and narrated voiceover
python pipeline.py data/scratch/zoomies.mov --narrate

# Different style, and save the in-between files for later iteration
python pipeline.py data/scratch/zoomies.mov --style noir --narrate \
    --save-description --save-script
```

The clip length is detected automatically with ffprobe, and it sets the target word count (about 2.5 words per second, so a 28-second clip gets about 70 words). Pass `--length N` to override it.

Every output is optional and is written next to the video:

- `--save-description` writes `<name>.description.txt` (the stage 1 output). Reuse it to iterate on stage 2 quickly.
- `--save-script` writes `<name>.script.txt` (the stage 2 output). Reuse it to iterate on stage 3 quickly.
- `--narrate` writes `<name>.voiceover.wav` (stage 3).
  - Pauses between sentences are capped at 0.45 seconds by default. The Attenborough prompt asks for ellipses and paragraph breaks as dramatic beats, and both voice engines honor every one, which sounds like dead air in a short clip. A pause counts even when it isn't silent: Chatterbox fills its pauses with the breath and room tone of your reference clip, so anything more than 28 dB under the speech level is treated as pause. Silence before the first word is trimmed to 0.15 seconds and after the last word to 0.5 seconds. Tighten it with `--max-pause 0.35`, or pass `--max-pause 0` to keep the engine's own timing. To change what counts as a pause, edit `PAUSE_BELOW_SPEECH_DB` in `catplanet/generate_voice.py`: lower it (25 catches more of the breathy gaps) to trim more, raise it (30) if a soft word ending ever gets clipped.
  - The voiceover is normalized to −16 LUFS (`--target-lufs`), the usual level for podcasts and short-form video. Neither engine does this on its own; Kokoro writes at about −24 LUFS. It matters for the music too, because the music is leveled relative to this file. Speech is peaky, so the true-peak guard often lands a little short of the target rather than clipping, which is expected. `--target-lufs 0` skips normalization.
- `--score` writes `<name>.music.wav` (stage 4). See [music.md](music.md).

Use `--audio-out path.wav` or `--music-out path.wav` to send the audio somewhere else, or `--out-dir folder/` to send every output to another folder.

### Resuming a run that failed

Add `--resume` to save every stage's result as it finishes, plus a `<name>.status.json` progress file. If the run fails, run the same command again and only the missing stages run. Settings you change in between redo only the stages they affect, and replaced files are kept in a `<name>.attempt-N/` folder. `--fresh` redoes every stage, keeping the old files the same way. `batch.py` always works this way; without either option, `pipeline.py` simply overwrites, as it always has. The rules are explained in [batch.md](batch.md#retries-pick-up-where-they-stopped).

```bash
python pipeline.py data/scratch/zoomies.mov --narrate --score --resume
```

## Stage 1: describe a video

```bash
python -m catplanet.describe_video data/scratch/cat_clip.mov
python -m catplanet.describe_video data/scratch/cat_clip.mov > data/scratch/cat_clip.description.txt
```

The description is laid out as a one-line setting, then 3 to 6 key moments in order, then the funniest moment. Each moment and the funniest moment carry the stretch of the clip they cover, such as `(2.5-5.0 s)`. Moments rather than frames keep the description short, which makes stages 1 and 2 faster, and stop the model from getting stuck restating one frame.

The model gets those times because the prompt lists each frame's time in the clip, in the same order as the frames. The times are only as fine as the gap between frames: 24 frames spread over a 20-second clip are about 0.85 seconds apart. Descriptions saved before timestamps existed have no times; a resumed or batch run redoes them, because the prompt changed.

Useful options:

- `--fps 2` sets how many frames per second to sample (default 2). Raise it to 4 for fast action such as zoomies or pouncing, and drop it to 1 for slow footage such as sleeping or grooming.
- `--max-frames 24` caps the number of frames sent to the model. 16 to 24 is the sweet spot between quality and speed. More is worse, not better: in a November 2026 test on 11 clips, `--fps 4 --max-frames 40` (with `--num-ctx 24576`) missed or only partly caught the main event more often, timed the payoff right in 2 of 9 clips against 5 of 9, took about 70% longer, and two 30-second clips didn't fit in the context at all. Only the shortest clip (10 s) was timed better.
- `--prompt "..."` replaces the default description prompt. Stage 2 only suggests a cut if a custom prompt still asks for timed moments in the same layout.
- `--show-prompt` prints the prompt the model will see, including your cats, and exits.
- `--cats-file path.json` reads cats from another file; see [cats.md](cats.md).

## Stage 2: write a script from a description

```bash
# From a saved description
python -m catplanet.generate_script --description data/scratch/zoomies.description.txt --length 28

# Chained straight from stage 1
python -m catplanet.describe_video data/scratch/zoomies.mov \
  | python -m catplanet.generate_script --length 28 --style attenborough
```

There are four styles: `attenborough` (the default, a dry documentary), `sports` (excitable play-by-play), `noir` (a hardboiled detective) and `reality_tv` (a dramatic confessional). To change how a style sounds, edit its prompt in the `STYLES` dictionary at the top of `catplanet/generate_script.py`.

Stage 2 works like a comedy writers' room, in four steps:

1. **Facts.** A careful, low-temperature call writes a beat sheet: the main action the script must be about, and 3 to 6 plain sentences on what happens, in order.
2. **Pitch.** One call pitches a different comic angle for each draft, each built on the main action. Each angle takes a different lens: raising the stakes, playing the cats' personalities off each other, and a misdirection with a deflating punchline.
3. **Draft.** One draft of the narration is written around each angle. Every draft must say the main action plainly and narrate only actions from the facts. Jokes can exaggerate why the cats do things, never what they do: lying on her back doesn't become a backflip.
4. **Judge.** One last call checks every draft (does it show the main action? does it invent or upgrade any action?) and ranks them by funniness. The code, not the model, keeps the funniest draft that passed. If every draft fails, one more "faithful" draft (no angle, facts only) is written and checked; if that fails too, the funniest draft is kept with a warning.

### Suggested cut

When the description's moments have times, the beat sheet also names the setup (the earliest moment a viewer must see), the payoff (the moment with the main action) and any quiet moments where nothing happens. The code turns those into a suggested cut: start 0.5 seconds before the setup, end 1 second after the payoff, and drop quiet moments of 1.5 seconds or more between the two. The rules match the cat-videos-short-form skill's editing rules, and the numbers are `CUT_LEAD_IN`, `CUT_TAIL` and `CUT_MIN_DROP` in `catplanet/generate_script.py`.

`pipeline.py` and `batch.py` save it as `<name>.cuts.json` and list it under "Suggested cut" in `<name>.script_drafts.txt`. Running stage 2 by itself, `--cut-out cuts.json` saves it. It is a starting point for the edit, not a decision: nothing else in the pipeline uses it, and the script is still written for the whole clip.

### Priority moments

Some moments do better with your audience than others. List them under `priority_moments` in `config/local_settings.json`:

```json
"priority_moments": [
  "The cats play fighting",
  "One of the cats doing a weird, unusual move",
  "The cats staring at something in unison",
  "The cats having the zoomies",
  "A cat accepting pets from a human"
]
```

When the beat sheet finds that a clip clearly shows one of them, that moment becomes the main action, so every draft is built around it. If a clip shows several, the one listed first wins, so put your favourite at the top. A moment only counts when the description plainly describes it. Editing the list redoes the script on the next resumed or batch run.

`--drafts 3` is the default; `--drafts 1` writes a single draft and skips the judging, which is the quickest way to iterate. `--drafts-out drafts.txt` saves the facts (and the priority moment, if one matched), every angle and draft, each draft's check, the funniness ranking and why the winner won. `pipeline.py` and `batch.py` always save this as `<name>.script_drafts.txt` next to the script, so if you prefer a different draft you can paste it into `<name>.script.txt` and re-run stage 3. Each extra draft adds roughly 10 to 30 seconds, and the beat sheet another 10 to 15.

The script opens with a short, funny title for the clip (two to six words, in the narrator's voice) on its own line, followed by a blank line and the narration. Stage 3 reads the whole script, so the narrator says the title first. To leave room for it, the narration aims for 5 fewer words than the clip length alone would suggest. `--no-title` leaves the title out; `pipeline.py` and the control panel ("Spoken title") have the same switch.

The script writer is also given your cats' names, pronouns and personalities from `config/local_settings.json` (see [cats.md](cats.md#personalities)), and is told to call each cat by the name the description uses, never "the creature". A cat the description calls "one of the cats" stays unnamed.

Other options: `--temperature 0.9` (raise to 1.1 for wilder output, drop to 0.7 for safer output), `--wps 2.5` (the target speaking pace in words per second) and `--model qwen2.5:14b` (a different text model).

When you run through `pipeline.py --resume` or `batch.py`, editing a style's prompt or your cats list counts as a settings change, so the script and voiceover are redone on the next run.

## Stage 3: turn a script into a voiceover

```bash
# Kokoro (the default engine)
python -m catplanet.generate_voice --script data/scratch/zoomies.script.txt --out data/scratch/zoomies.wav

# Chained from stage 2
python -m catplanet.generate_script --description d.txt --length 28 \
    | python -m catplanet.generate_voice --out data/scratch/zoomies.wav

# Chatterbox, cloning a voice clip
python -m catplanet.generate_voice --script s.txt --out out.wav \
    --engine chatterbox --reference-audio data/voices/my_narrator.wav \
    --exaggeration 0.6
```

The choice between the two engines, the voice options and the tuning recipe are in [voice.md](voice.md).

## Stage 4: score a clip from a saved description

This runs the music stage without re-running the vision model. The music server must be running.

```bash
python -m catplanet.generate_music --description data/scratch/five.description.txt \
    --video data/scratch/five.mov --style noir --takes 3 \
    --voiceover data/scratch/five.voiceover.wav --out data/scratch/five.music.wav

# Print the music caption without generating anything
python -m catplanet.generate_music --description d.txt --length 18 --print-caption
```

Every music option is listed in [music.md](music.md).
