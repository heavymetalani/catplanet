# Cat Planet: nature documentary narration for house cats

This project turns a short cat video into an Animal-Planet-style voiceover with a string quartet score. Everything runs locally on a Mac (built for a Mac Mini M4 with 24 GB of memory).

## How it works

Each video goes through four stages.

| Stage | What it does | Model | Code |
| --- | --- | --- | --- |
| 1 | Watches the video and writes a detailed description of what the cats do. | MiniCPM-V 4.5 (via Ollama) | `catplanet/describe_video.py` |
| 2 | Pitches a few comic angles, writes a draft of the narration for each in a chosen style, and keeps the funniest. | Qwen2.5-14B (via Ollama) | `catplanet/generate_script.py` |
| 3 | Reads the script aloud and saves a voiceover `.wav`. | Kokoro or Chatterbox | `catplanet/generate_voice.py` |
| 4 | Writes and records a string quartet cue to sit under the narration. | ACE-Step 1.5 (separate local server) | `catplanet/generate_music.py` |

`pipeline.py` runs all of the stages for one video. `batch.py` runs the pipeline over every video in your inbox and files each finished clip into its own folder.

The voiceover and the music come out as separate `.wav` files, already leveled so the music sits about 9 LU under the voice. You drop both into your video editor over the clip; the pipeline does not mix them into the video for you.

## Project layout

```
video-interpreter/
├── app.py                    the control panel:    python app.py
├── pipeline.py               run one video:        python pipeline.py <video>
├── batch.py                  run the whole inbox:  python batch.py
├── catplanet/                the code for every stage
│   ├── describe_video.py       stage 1
│   ├── generate_script.py      stage 2
│   ├── generate_voice.py       stage 3
│   ├── generate_music.py       stages 2b and 4
│   ├── pipeline.py             chains the stages for one clip
│   ├── batch.py                numbering, queue and filing
│   ├── paths.py                every folder location, in one place
│   ├── progress.py             safe file writes and resuming failed runs
│   ├── app.py                  the control panel's local web server
│   ├── app_options.py          every option's label, choices and ⓘ note for the panel
│   ├── ui/index.html           the control panel page
│   ├── local_settings.py       reads config/local_settings.json
│   └── audio_utils.py          loudness measurement and leveling
├── config/
│   ├── local_settings.json     your voice clip and your cats (private, not in git)
│   ├── ui_profiles.json        the control panel's saved profiles (private, not in git)
│   └── local_settings.example.json
├── data/                     all personal and generated media (not in git)
│   ├── inbox/                  drop new videos here
│   ├── voices/                 voice clips for Chatterbox to clone
│   ├── outputs/                one folder per finished clip (see below)
│   ├── scratch/                anywhere you like to test single clips by hand
│   └── batch_state.json        clip-number counter and run log (back this up)
├── docs/                     detailed guides (listed at the bottom)
├── MUSIC GEN/                ACE-Step launchers and notes (the music server)
├── review/                   old experiments set aside for you to look through
├── install.sh
└── requirements.txt
```

Every folder path the code uses is defined in `catplanet/paths.py`. If you want to move a folder, change it there.

## One-time setup

1. Run the install script. It installs Homebrew packages, creates the Python virtualenv, pulls the two Ollama models (about 14 GB), creates the `data/` folders, and copies the settings template into `config/local_settings.json` if you don't have one yet. It is safe to re-run.

   ```bash
   git clone https://github.com/heavymetalani/catplanet.git
   cd catplanet
   bash install.sh
   ```

2. Edit `config/local_settings.json`. Put a 5 to 10 second voice clip in `data/voices/` and point `reference_audio` at it (for example `"data/voices/narrator.wav"`), and describe your cats so the vision model can tell them apart. Every field is optional. See [docs/cats.md](docs/cats.md) for how to describe your cats well.

3. If you want music, set up ACE-Step once by following [MUSIC GEN/README.md](MUSIC%20GEN/README.md).

## Daily use

Activate the virtualenv once per terminal session:

```bash
source .venv/bin/activate
```

### The easy way: the control panel

```bash
python app.py
```

This opens a page in your browser with every option as a dropdown. Each option has an ⓘ button that explains, in plain English, what changing it does. The page checks that Ollama, the music server and your voice clip are ready (and can start the music server for you), shows the exact command it will run, and plays the results when the run is done. See [docs/app.md](docs/app.md).

Everything below is the same work done from the Terminal.

### Process a batch of clips (the usual way)

If you are generating music (the default), start the music server first in its own terminal window and leave it open:

```bash
"MUSIC GEN/start_api_server.command"
```

Then drop your videos into `data/inbox/` and run:

```bash
python batch.py --dry-run   # optional: preview the plan without touching anything
python batch.py
```

Each video gets the next number, and all of its files land together in one folder:

```
data/outputs/53/
├── 53.mov               the original video, renamed to its number
├── 53.description.txt   what the vision model saw
├── 53.script.txt        the narration script
├── 53.script_drafts.txt every draft written, and why the script won
├── 53.music_plan.json   the music caption, tempo and key
├── 53.voiceover.wav     the narrated voiceover
├── 53.music.1.wav       string quartet take 1
├── 53.music.2.wav       take 2
├── 53.music.3.wav       take 3
└── 53.status.json       which stages finished, and with which settings
```

A video that fails keeps its number and stays in `data/inbox/`. Running `python batch.py` again resumes it from the stage where it stopped, so finished work (such as a good voiceover) is kept. If you change settings first, only the stages they affect are redone, and any file that gets replaced is kept in an `attempt-N/` folder. `python batch.py --fresh` redoes every stage instead. Do not delete `data/batch_state.json`; it holds the counter. See [docs/batch.md](docs/batch.md) for the full rules and options.

### Process a single clip by hand

`pipeline.py` writes its outputs next to the video you give it, so a scratch folder such as `data/scratch/` is a good place for experiments.

```bash
python pipeline.py data/scratch/zoomies.mov                     # print the script only
python pipeline.py data/scratch/zoomies.mov --narrate           # script and voiceover
python pipeline.py data/scratch/zoomies.mov --narrate --score   # plus music (server must be running)
```

Run `python pipeline.py --help` to see every option. Each stage can also run on its own, which is the fast way to iterate on one step; see [docs/stages.md](docs/stages.md).

## Detailed guides

| Guide | What it covers |
| --- | --- |
| [docs/app.md](docs/app.md) | Using the control panel. |
| [docs/batch.md](docs/batch.md) | How numbering, retries and the safety checks in `batch.py` work, and how to change the default settings. |
| [docs/stages.md](docs/stages.md) | Running each stage on its own, with its options and the narration styles. |
| [docs/cats.md](docs/cats.md) | Teaching the vision model your cats' names. |
| [docs/voice.md](docs/voice.md) | Choosing between Kokoro and Chatterbox, voice cloning, and the Attenborough tuning recipe. |
| [docs/music.md](docs/music.md) | Adding the string quartet score, and every music option. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | A first-run sanity check and fixes for common errors. |
| [docs/roadmap.md](docs/roadmap.md) | Ideas for what to build next. |
| [docs/MUSIC_PLAN.md](docs/MUSIC_PLAN.md) | The original design notes for the music stage. |
| [MUSIC GEN/README.md](MUSIC%20GEN/README.md) | Installing and running the ACE-Step music server. |
