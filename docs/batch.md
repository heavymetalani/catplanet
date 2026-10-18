# Batch processing

`batch.py` runs the pipeline over every video in `data/inbox/`, one at a time. It numbers the clips 1, 2, 3 and so on, and the numbering continues across runs: if the last batch ended at 52, the next video becomes 53.

## Where things go

```
data/inbox/              drop new videos here (this is the queue)
data/outputs/53/         everything for clip 53:
    53.mov                 the video, moved here once the run succeeds
    53.description.txt     what the vision model saw (stage 1)
    53.script.txt          the narration script (stage 2)
    53.script_drafts.txt   every draft stage 2 wrote, and why the script won
    53.music_plan.json     the music caption, tempo and key (stage 2b)
    53.voiceover.wav       the voiceover (stage 3)
    53.music.1.wav ...     the music takes (stage 4)
    53.status.json         which stages finished, and with which settings
    attempt-1/             older files that a later run replaced (only if any were)
data/batch_state.json    the counter and a log of every run
```

Each stage saves its result into the clip's folder as soon as it finishes. The video itself moves out of the inbox only after the whole run succeeds, so `data/inbox/` always shows exactly what is still outstanding.

## Retries pick up where they stopped

When a run starts, the video is first renamed in place to its number (for example `IMG_4410.mov` becomes `53.mov`). If the run then fails, the file stays in the inbox already numbered. The next time you run `python batch.py`, the file is recognized as clip 53 and retried under the same number instead of using up a new one.

The retry does not start from scratch. It looks at `53.status.json` and skips every stage that already finished, so only the missing work runs. For example, if clip 53 failed while making music take 2, the retry reuses the description, script, music plan, voiceover and take 1, and generates only takes 2 and 3. The batch summary tells you where each failed clip stopped:

```
  [FAIL]   53  IMG_4410.mov  (212s)
               stopped at stage 4 (music take 2 of 3): stage 4 failed: ...
```

`python batch.py --dry-run` shows the same information before you retry.

### If you change settings before retrying

Each stage remembers the settings and inputs it used. When you retry with different settings, only the stages those settings affect are redone, and everything after them follows:

| What you changed | What is redone |
| --- | --- |
| `--style` | the script, the music plan, the voiceover and the music |
| a voice setting such as `--exaggeration` or the reference clip | the voiceover only (and the music, if its loudness target moves) |
| a music setting such as `--music-seed` or `--music-thinking` | the music only |
| `--music-takes 3` to `--music-takes 4` | only the new take 4 |
| your cats in `config/local_settings.json`, or the vision settings | everything, starting with the description |
| a setting that doesn't change the result, such as `--music-timeout` | nothing |

Whenever a stage is redone and an older file of it exists, the older file is moved into an `attempt-1/` folder inside the clip's folder (then `attempt-2/`, and so on) instead of being overwritten. Each attempt folder also holds a copy of the progress file from before that run.

### Starting a clip over: `--fresh`

`python batch.py --fresh` ignores the saved progress and redoes every stage for the clips in the inbox. Nothing is lost: every file it replaces is moved into the clip's next `attempt-N/` folder.

### Interrupted runs

Every file is written under a hidden temporary name (for example `.53.voiceover.wav.partial`) and gets its real name only when it is complete. If the Mac crashes or you press Ctrl-C, a file with a real name is therefore never half-written, and the next run deletes any leftover `.partial` files before it starts.

## The number guard

Retrying under the same number has a guard. If you drop in a video that merely happens to be called `1.mov`, reusing number 1 would overwrite the real clip 1. So a file named with a bare number only reuses that number when `data/outputs/<number>/` does not already contain a finished video. Otherwise it is treated as a new clip, given the next free number, and you get a warning.

## Safety checks before anything is renamed

Several checks run up front, so a mistake fails immediately instead of after half the queue has been renamed.

- Every argument is checked against `pipeline.py`'s real option list. A typo like `--musictakes` stops the run before any video is touched.
- `--audio-out`, `--music-out` and `--out-dir` are rejected, because `batch.py` decides those paths itself. Use `--outputs` to move the whole output folder, or run `pipeline.py` directly for a one-off.
- If music is enabled, the ACE-Step server must be answering. Start it with `"MUSIC GEN/start_api_server.command"` first.
- If a reference voice clip is set, the file must exist.

## Do not delete `data/batch_state.json`

It holds the counter. Losing it restarts numbering at 1, which would overwrite your existing clips. It is not in git (it lists your original video filenames), so it only exists on your machine. Back it up along with `data/outputs/`.

## The default settings, and how to change them

The settings used for every clip live in `DEFAULT_PIPELINE_ARGS` near the top of `catplanet/batch.py`:

```
--narrate --engine chatterbox --exaggeration 0.3 --cfg-weight 0.35
--style attenborough --score --music-takes 3 --num-ctx 32768
```

`batch.py` also adds `--out-dir data/outputs/<number>` and `--resume` (or `--fresh`) for each clip.

The one exception is the voice that Chatterbox clones. A recording of a real person is personal, so it comes from `reference_audio` in `config/local_settings.json` instead. Without it, Chatterbox narrates in its built-in default voice.

Anything `batch.py` does not recognize is passed on to `pipeline.py` and added after the defaults, so it wins for that run:

```bash
python batch.py --dry-run              # show the plan and touch nothing (uses no numbers)
python batch.py --music-takes 1        # override a default for this run
python batch.py --limit 2              # only the first two videos in the inbox
python batch.py --style noir
```

## Options that belong to `batch.py` itself

| Option | Default | What it does |
| --- | --- | --- |
| `--inbox DIR` | `data/inbox/` | The queue folder to read videos from. |
| `--outputs DIR` | `data/outputs/` | Where each clip's numbered folder is created. |
| `--state FILE` | `data/batch_state.json` | The counter and run log. |
| `--limit N` | none | Process at most N videos this run. |
| `--dry-run` | off | Print the plan without renaming, generating or using up numbers. It also shows where each queued clip stopped last time. |
| `--fresh` | off | Redo every stage instead of resuming. Replaced files go into each clip's `attempt-N/` folder. |
| `--exact-args` | off | Use only the pipeline options on the command line: skip `DEFAULT_PIPELINE_ARGS` and the saved `reference_audio`. The control panel always uses this, so what it shows is exactly what runs. |
