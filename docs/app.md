# The control panel

The control panel is a page in your browser for choosing settings and running the pipeline without typing any options. It runs only on your Mac and doesn't go on the internet.

## Opening it

```bash
cd ~/genai-projs/video-interpreter
source .venv/bin/activate
python app.py
```

Your browser opens at `http://localhost:8765`. Leave the Terminal window open while you use the panel. To quit, press Ctrl-C in that window (closing the window works too). Quitting stops any run that is going (it can resume later) and stops the music server if the panel started it.

If the port is busy, run `python app.py --port 8766`. To start it without opening a browser tab, add `--no-browser`.

## Setting up a run

**Batch or single video.** Use the switch at the top right. *Batch: whole inbox* runs `batch.py` over `data/inbox/`. *Single video* runs `pipeline.py` on one video you pick. Its results always go into their own folder, `data/scratch/<video name>/`, and the video itself is never moved or changed. Running the same video again uses the same folder and resumes from there.

**Profiles.** A profile is a saved set of settings. The panel starts with three:

| Profile | What it does |
| --- | --- |
| My usual | Your normal batch settings: Attenborough style, Chatterbox cloning your voice clip, three music takes. |
| Quick test (Kokoro, no music) | A fast check of a new clip: the built-in Kokoro voice and no music. |
| Noir | Like My usual, in the noir style and a little more expressive. |

When you change a setting, its row gets an ochre bar and the profile bar says how many changes you've made. *Save to this profile* keeps them, *Save as new…* makes a new profile, and *Undo changes* goes back to the saved values. Profiles are stored in `config/ui_profiles.json`, which is not in git.

**The ⓘ buttons.** Every option has one. Tap it to read, in plain English, what changing that option does to the narration or the music. The note also shows your profile's value and the pipeline's own default.

**Only relevant options are shown.** For example, the voice-clip, expressiveness and pacing options appear only with Chatterbox, and Kokoro's voice and speed only with Kokoro. Turning music off hides the music options. The *Advanced* section is closed by default and holds the rarely changed options, grouped by what they affect.

**Custom values.** Dropdowns with number settings end with *Custom number…*, which opens a box for any value.

## The ready check

The right-hand panel checks everything a run needs:

- **Ollama** must be running for new descriptions and scripts. If it isn't, run `brew services start ollama`.
- **The music server** is needed only when music is on. Press **Start it** and the panel starts it in the background. The first start takes a minute or two, and Start unlocks when it's ready. Its output goes to `data/app_logs/music_server.log`. If you started the server yourself in another window, the panel uses that one and leaves it alone when you quit.
- **The voice clip** must exist if Chatterbox is cloning one.
- **The inbox** (batch) lists the waiting videos and notes any that stopped part-way last time. In single-video mode this row checks the video instead.

When something blocks the run, the Start button is disabled and the reason appears under it.

## Will run

This box shows the exact command the panel will run, and it updates as you change settings. You can copy it to run the same thing in Terminal. In batch mode it includes `--exact-args`, which tells `batch.py` to use exactly these settings and nothing else (no hidden defaults).

*Preview (dry run)* shows what the batch would do, including the clip numbers and which clips would resume, without changing anything.

## While it runs

Each clip gets a trail of stages: describe, script, music plan, voiceover and music. A paw print means the stage is finished, and a faded paw print means it was kept from an earlier attempt. The ochre ring marks the stage in progress, and a cross marks where a clip stopped. The music stage counts its takes, for example `music 2/3`.

**Stop** works like pressing Ctrl-C: the current step is abandoned, everything finished so far is kept, and the next run resumes from there. If a run doesn't stop, press **Force stop**.

The *Live log* shows the full output. Every run's log is also saved in `data/app_logs/`.

## Results

When the run ends you see each clip with its script, a player for the voiceover and every music take, and **Show in Finder**. A clip that failed or was stopped says where it stopped and has a **Retry** button, which resumes it from there.

## How it is built

| File | What it holds |
| --- | --- |
| `app.py` | The shortcut you run. |
| `catplanet/app.py` | The local web server, the run manager and the music-server manager. Standard library only. |
| `catplanet/app_options.py` | Every option's label, dropdown choices, section, when it applies, and its ⓘ note. |
| `catplanet/ui/index.html` | The page itself. |

The option list is checked against `pipeline.py`'s real options every time the panel starts. If an option is added to the pipeline later and not yet described in `app_options.py`, it still appears under *Advanced → Other*, with the pipeline's own help text as its ⓘ note. Running `python -m catplanet.app_options` reports any option that is missing a label or an ⓘ note.
