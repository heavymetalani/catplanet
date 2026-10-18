"""
app_options.py

Every option the control panel shows, in plain language: which section it
sits in, its dropdown choices, a short note on when to change it, and when it
is relevant at all (the Kokoro voice list only matters with Kokoro, and so on).

The command-line parser in pipeline.py stays the source of truth. At start-up
`schema()` checks every option here against it, and any parser option this
file doesn't describe yet is still shown, under Advanced, with the parser's
own help text. So a flag added to the pipeline later appears automatically.

`build_command()` turns the panel's values into the exact command that runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import local_settings, paths

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".aiff"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}

STYLE_CHOICES = [
    ("attenborough", "Attenborough – dry nature documentary"),
    ("sports", "Sports – excitable play-by-play"),
    ("noir", "Noir – hardboiled detective"),
    ("reality_tv", "Reality TV – dramatic confessional"),
]

# Options the panel sets for you, so they never appear as fields.
#   --out-dir, --audio-out, --music-out   where outputs go (batch decides, or
#                                         the single-video "Save to" field)
#   --resume, --fresh                     the "If it ran before" field
#   --save-description, --save-script     always saved in resume mode
MANAGED = {"out_dir", "audio_out", "music_out", "resume", "fresh",
           "save_description", "save_script", "help"}

# ---------------------------------------------------------------------------
# Field definitions
#
#   key      the parser's dest name (or a panel-only name for batch/single)
#   kind     select | toggle (store_true flag) | inverse (flag means "off")
#            | text
#   type     how a chosen value is passed: str, int or float
#   choices  [(value, label)]; value None means "leave the flag out"
#   custom   "number" or "text" adds a "Custom…" entry with an input box
#   show     {key: value or [values]}; the field only applies when all match
# ---------------------------------------------------------------------------
SECTIONS = [
    ("story", "Story"),
    ("voice", "Voice"),
    ("music", "Music"),
    ("run", "This run"),
    ("advanced", "Advanced"),
]

FIELDS: list[dict] = [
    # ------------------------------------------------------------- story --
    dict(key="style", section="story", kind="select", type=str,
         label="Narration style", choices=STYLE_CHOICES,
         help="The voice of the script. The music follows it unless you pick "
              "a different music mood."),
    dict(key="length", section="story", kind="select", type=int,
         label="Clip length",
         choices=[(None, "Detect automatically"), (10, "10 seconds"),
                  (15, "15 seconds"), (20, "20 seconds"), (30, "30 seconds"),
                  (45, "45 seconds"), (60, "60 seconds")],
         custom="number",
         help="Sets how many words the script aims for, about 2.5 per second. "
              "Leave it on automatic unless detection fails."),
    dict(key="temperature", section="story", kind="select", type=float,
         label="Script creativity",
         choices=[(0.7, "0.7 – safer, more predictable"),
                  (0.9, "0.9 – balanced"),
                  (1.1, "1.1 – wilder, more surprising")],
         custom="number",
         help="How adventurous the script writer is."),
    dict(key="no_title", section="story", kind="inverse",
         label="Spoken title", on="On – the narrator reads a title first",
         off="Off – narration only",
         help="A short, funny title for the clip, read aloud before the "
              "narration."),

    # ------------------------------------------------------------- voice --
    dict(key="narrate", section="voice", kind="toggle",
         label="Voiceover", on="On – make a narrated .wav", off="Off – script only",
         help="Turn off to get just the script, which is much faster."),
    dict(key="engine", section="voice", kind="select", type=str,
         label="Voice engine", show={"narrate": True},
         choices=[("chatterbox", "Chatterbox – clones your voice clip (slower)"),
                  ("kokoro", "Kokoro – built-in British narrator (fast)")],
         help="Chatterbox copies the voice clip you pick below. Kokoro uses a "
              "preset voice and is about ten times faster."),
    dict(key="reference_audio", section="voice", kind="select", type=str,
         label="Voice clip to clone", show={"narrate": True, "engine": "chatterbox"},
         choices="voices", custom="text",
         help="A clean 5 to 15 second recording in data/voices/. Its pacing "
              "and pauses carry over into the narration."),
    dict(key="exaggeration", section="voice", kind="select", type=float,
         label="Expressiveness", show={"narrate": True, "engine": "chatterbox"},
         choices=[(0.2, "0.2 – very flat"),
                  (0.3, "0.3 – calm, deadpan"),
                  (0.4, "0.4 – a little livelier"),
                  (0.5, "0.5 – balanced"),
                  (0.6, "0.6 – expressive"),
                  (0.7, "0.7 – very emotive")],
         custom="number",
         help="Lower suits the dry Attenborough style; higher suits sports or "
              "reality TV."),
    dict(key="cfg_weight", section="voice", kind="select", type=float,
         label="Pacing", show={"narrate": True, "engine": "chatterbox"},
         choices=[(0.25, "0.25 – relaxed, lets pauses breathe"),
                  (0.35, "0.35 – natural, follows punctuation"),
                  (0.5, "0.5 – even"),
                  (0.6, "0.6 – tight, sticks closely to the text")],
         custom="number",
         help="If the voice rushes and ignores punctuation, go lower. If it "
              "doesn't sound like the clip, go higher."),
    dict(key="voice", section="voice", kind="select", type=str,
         label="Kokoro voice", show={"narrate": True, "engine": "kokoro"},
         choices=[("bm_george", "George – warm, mature"),
                  ("bm_lewis", "Lewis – younger, conversational"),
                  ("bm_daniel", "Daniel – crisp, faster"),
                  ("bm_fable", "Fable – storyteller")],
         custom="text",
         help="Other Kokoro voices (am_*, af_*, bf_*) work too; use Custom."),
    dict(key="speed", section="voice", kind="select", type=float,
         label="Kokoro speed", show={"narrate": True, "engine": "kokoro"},
         choices=[(0.9, "0.9 – slow"), (0.95, "0.95 – a touch of gravitas"),
                  (1.0, "1.0 – normal"), (1.1, "1.1 – energetic")],
         custom="number"),
    dict(key="max_pause", section="voice", kind="select", type=float,
         label="Longest pause", show={"narrate": True},
         choices=[(0.25, "0.25 s – tight"), (0.35, "0.35 s – natural"),
                  (0.5, "0.5 s – relaxed"), (0, "No limit – keep the engine's timing")],
         custom="number",
         help="Caps the silence between sentences so short clips don't have "
              "dead air."),

    # ------------------------------------------------------------- music --
    dict(key="score", section="music", kind="toggle",
         label="String quartet", on="On – score the clip", off="Off – no music",
         help="Needs the music server. The panel can start it for you."),
    dict(key="music_takes", section="music", kind="select", type=int,
         label="Takes", show={"score": True},
         choices=[(i, f"{i} take{'s' if i > 1 else ''}") for i in range(1, 9)],
         help="Each take is a different version to audition, at about a "
              "minute each."),
    dict(key="music_style", section="music", kind="select", type=str,
         label="Music mood", show={"score": True},
         choices=[(None, "Same as the narration style")] + STYLE_CHOICES,
         help="Pick a different mood to, say, put noir music under an "
              "Attenborough script."),

    # ---------------------------------------------------------- this run --
    dict(key="retry", section="run", kind="select", type=str, mode="both",
         label="If a clip ran before",
         choices=[("resume", "Resume – keep finished stages"),
                  ("fresh", "Redo every stage – keep old files in attempt-N/")],
         help="Resume only redoes what is missing or what your new settings "
              "change."),
    dict(key="limit", section="run", kind="select", type=int, mode="batch",
         label="Videos to process",
         choices=[(None, "Everything in the inbox"), (1, "Just the first"),
                  (2, "The first 2"), (3, "The first 3"), (5, "The first 5"),
                  (10, "The first 10")],
         custom="number"),
    dict(key="video", section="run", kind="select", type=str, mode="single",
         label="Video", choices="videos", custom="text",
         help="Pick a video, or choose Custom and paste a full path."),

    # ---------------------------------------------------------- advanced --
    dict(key="fps", section="advanced", group="Watching the video", kind="select",
         type=float, label="Frames per second",
         choices=[(1.0, "1 – slow footage (sleeping, grooming)"), (2.0, "2 – most clips"),
                  (3.0, "3 – lively"), (4.0, "4 – fast action (zoomies, pouncing)")],
         custom="number"),
    dict(key="max_frames", section="advanced", group="Watching the video", kind="select",
         type=int, label="Most frames sent",
         choices=[(12, "12"), (16, "16"), (24, "24 – the sweet spot"), (32, "32")],
         custom="number", help="More frames cost memory and time."),
    dict(key="num_ctx", section="advanced", group="Watching the video", kind="select",
         type=int, label="Vision context size",
         choices=[(8192, "8,192"), (16384, "16,384"), (32768, "32,768"),
                  (65536, "65,536")],
         custom="number",
         help="Each frame needs about 500. Raise it if Ollama says the request "
              "doesn't fit."),
    dict(key="vision_model", section="advanced", group="Models", kind="text",
         type=str, label="Vision model"),
    dict(key="text_model", section="advanced", group="Models", kind="text",
         type=str, label="Text model"),

    dict(key="target_lufs", section="advanced", group="Voice", kind="select",
         type=float, label="Voiceover loudness", show={"narrate": True},
         choices=[(-14.0, "−14 LUFS – loud (streaming)"),
                  (-16.0, "−16 LUFS – short-form standard"),
                  (-18.0, "−18 LUFS – quieter"), (0.0, "Don't normalize")],
         custom="number", help="The music is leveled relative to this."),
    dict(key="max_chars", section="advanced", group="Voice", kind="select",
         type=int, label="Chatterbox chunk size",
         show={"narrate": True, "engine": "chatterbox"},
         choices=[(150, "150"), (180, "180"), (250, "250"), (350, "350"),
                  (0, "No chunking")],
         custom="number",
         help="Lower it if long scripts turn into gibberish."),
    dict(key="device", section="advanced", group="Voice", kind="select",
         type=str, label="Chatterbox device",
         show={"narrate": True, "engine": "chatterbox"},
         choices=[(None, "Automatic"), ("mps", "Apple GPU (mps)"),
                  ("cpu", "CPU – slower, most reliable"), ("cuda", "CUDA")],
         help="Use CPU if you see a Metal error."),

    dict(key="no_music_vary", section="advanced", group="Music", kind="inverse",
         label="How takes differ", show={"score": True},
         on="Each take different (one request per take)",
         off="One batched request (faster, nearly identical)"),
    dict(key="no_music_llm", section="advanced", group="Music", kind="inverse",
         label="Music plan", show={"score": True},
         on="Let the text model shape the mood",
         off="Bare preset (fast, predictable)"),
    dict(key="music_thinking", section="advanced", group="Music", kind="toggle",
         label="Arrangement planning", show={"score": True},
         on="On – better musicality, more memory", off="Off"),
    dict(key="music_bpm", section="advanced", group="Music", kind="select",
         type=int, label="Tempo", show={"score": True},
         choices=[(None, "From the style preset"), (60, "60 BPM"), (72, "72 BPM"),
                  (90, "90 BPM"), (120, "120 BPM"), (144, "144 BPM")],
         custom="number"),
    dict(key="music_seed", section="advanced", group="Music", kind="select",
         type=int, label="Seed", show={"score": True},
         choices=[(None, "Random")], custom="number",
         help="Fix a seed while tuning one setting, so you hear only that "
              "setting's effect."),
    dict(key="music_lufs", section="advanced", group="Music", kind="select",
         type=float, label="Music loudness", show={"score": True},
         choices=[(None, "9 LU under the voiceover"), (-24.0, "−24 LUFS – a bit more present"),
                  (-22.0, "−22 LUFS – starts competing with the voice")],
         custom="number"),
    dict(key="music_bit_depth", section="advanced", group="Music", kind="select",
         type=int, label="Bit depth", show={"score": True},
         choices=[(16, "16-bit"), (24, "24-bit"), (32, "32-bit")]),
    dict(key="save_music_prompt", section="advanced", group="Music", kind="toggle",
         label="Save each take's caption", show={"score": True},
         on="On – a .json next to each take", off="Off"),
    dict(key="music_timeout", section="advanced", group="Music", kind="select",
         type=float, label="Give up on a take after", show={"score": True},
         choices=[(120.0, "2 minutes"), (300.0, "5 minutes"), (600.0, "10 minutes")],
         custom="number"),
    dict(key="music_server", section="advanced", group="Music", kind="text",
         type=str, label="Music server address", show={"score": True}),

    dict(key="inbox", section="advanced", group="Batch folders", kind="text",
         type=str, mode="batch", label="Inbox folder", default=None,
         placeholder="data/inbox"),
    dict(key="outputs", section="advanced", group="Batch folders", kind="text",
         type=str, mode="batch", label="Outputs folder", default=None,
         placeholder="data/outputs"),
    dict(key="state", section="advanced", group="Batch folders", kind="text",
         type=str, mode="batch", label="Counter file", default=None,
         placeholder="data/batch_state.json"),
]

# Panel-only fields (not pipeline flags) and their defaults.
PANEL_DEFAULTS = {"retry": "resume", "limit": None, "video": None,
                  "inbox": None, "outputs": None, "state": None}


# ---------------------------------------------------------------------------
# Starter profiles. "My usual" mirrors batch.py's DEFAULT_PIPELINE_ARGS.
# ---------------------------------------------------------------------------
def starter_profiles() -> dict:
    ref = local_settings.get("reference_audio")
    usual = {
        "style": "attenborough", "narrate": True, "engine": "chatterbox",
        "reference_audio": ref, "exaggeration": 0.3, "cfg_weight": 0.35,
        "score": True, "music_takes": 3, "num_ctx": 32768,
    }
    quick = {
        "style": "attenborough", "narrate": True, "engine": "kokoro",
        "voice": "bm_george", "score": False, "num_ctx": 32768,
    }
    noir = dict(usual, style="noir", exaggeration=0.4)
    return {"My usual": usual, "Quick test (Kokoro, no music)": quick, "Noir": noir}


# ---------------------------------------------------------------------------
# Schema for the page
# ---------------------------------------------------------------------------
def _parser():
    from . import pipeline
    return pipeline.build_parser()


def _relative(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(paths.REPO))
    except ValueError:
        return str(p)


def voice_choices() -> list:
    found = []
    if paths.VOICES_DIR.is_dir():
        for p in sorted(paths.VOICES_DIR.iterdir(), key=lambda p: p.name.lower()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                found.append((_relative(p), p.name))
    return found + [(None, "Chatterbox's built-in voice")]


def video_choices() -> list:
    """Videos the single-video mode can pick, grouped by where they are."""
    groups = []
    for title, folder, recursive in [
        ("Scratch", paths.DATA_DIR / "scratch", True),
        ("Inbox", paths.INBOX_DIR, False),
        ("Finished clips", paths.OUTPUTS_DIR, True),
    ]:
        if not folder.is_dir():
            continue
        items = folder.rglob("*") if recursive else folder.iterdir()
        vids = [p for p in items
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                and not p.name.startswith(".") and "attempt-" not in str(p.parent.name)]
        if title == "Finished clips":
            vids.sort(key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.name))
        else:
            vids.sort(key=lambda p: p.name.lower())
        if vids:
            groups.append({"group": title,
                           "choices": [(_relative(p), _relative(p)) for p in vids]})
    return groups


def schema(mode_filter: str | None = None) -> dict:
    """Everything the page needs to draw the form."""
    parser = _parser()
    actions = {a.dest: a for a in parser._actions if a.option_strings}
    known = {f["key"] for f in FIELDS}
    out = []
    for f in FIELDS:
        f = dict(f)
        f.setdefault("mode", "both")
        a = actions.get(f["key"])
        if a is not None:
            f["flag"] = a.option_strings[0]
            if "default" not in f:
                f["default"] = (not a.default) if f["kind"] == "inverse" else a.default
            if f["kind"] == "text":
                f.setdefault("placeholder", str(a.default) if a.default is not None else "")
        else:
            f.setdefault("default", PANEL_DEFAULTS.get(f["key"]))
        if f.get("choices") == "voices":
            f["choices"] = voice_choices()
        elif f.get("choices") == "videos":
            f["groups"] = video_choices()
            f["choices"] = []
        f.pop("type", None)
        f["info"] = INFO.get(f["key"], f.get("help", ""))
        out.append(f)

    # Anything the parser has that this file doesn't describe yet.
    for dest, a in actions.items():
        if dest in known or dest in MANAGED:
            continue
        kind = "toggle" if a.nargs == 0 else "text"
        out.append(dict(key=dest, flag=a.option_strings[0], section="advanced",
                        group="Other", kind=kind, mode="both", label=a.option_strings[0],
                        default=a.default, help=a.help or "", info=a.help or "",
                        on="On", off="Off",
                        placeholder="" if a.default is None else str(a.default)))
    return {"sections": SECTIONS, "fields": out}


# ---------------------------------------------------------------------------
# Values -> command
# ---------------------------------------------------------------------------
def _field_map() -> dict:
    return {f["key"]: f for f in schema()["fields"]}


def _types() -> dict:
    return {f["key"]: f.get("type", str) for f in FIELDS}


def applies(field: dict, values: dict, mode: str) -> bool:
    if field.get("mode", "both") not in ("both", mode):
        return False
    for k, want in (field.get("show") or {}).items():
        have = values.get(k)
        if isinstance(want, list):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


def normalize(values: dict) -> dict:
    """Fill in defaults and coerce types, so the command builder sees clean values."""
    fields = _field_map()
    types = _types()
    clean = {}
    for key, f in fields.items():
        v = values.get(key, f.get("default"))
        if f["kind"] in ("toggle", "inverse"):
            v = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "on", "yes")
        elif v in ("", None):
            v = None
        else:
            t = types.get(key, str)
            try:
                v = t(v) if t is not float else float(v)
                if t is int and isinstance(v, float):
                    v = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"{f['label']}: '{v}' is not a valid number")
        clean[key] = v
    return clean


def pipeline_args(values: dict, mode: str) -> list[str]:
    """The pipeline flags for these values, leaving out anything at its default."""
    fields = _field_map()
    args: list[str] = []
    for key, f in fields.items():
        if "flag" not in f or not applies(f, values, mode):
            continue
        v = values.get(key)
        if f["kind"] == "toggle":
            if v:
                args.append(f["flag"])
        elif f["kind"] == "inverse":
            if not v:
                args.append(f["flag"])
        else:
            if v is None or v == f.get("default"):
                continue
            args += [f["flag"], _fmt(v)]
    return args


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v)) if abs(v) >= 1 or v == 0 else str(v)
    return str(v)


def single_output_dir(video: Path) -> Path:
    """Where a single-video run saves its results: data/scratch/<video name>/."""
    return paths.DATA_DIR / "scratch" / Path(video).stem


def build_command(mode: str, values: dict, dry_run: bool = False) -> list[str]:
    """The exact argv the panel runs, starting with the script name."""
    values = normalize(values)
    if mode == "batch":
        cmd = ["batch.py", "--exact-args"]
        if values.get("limit"):
            cmd += ["--limit", str(values["limit"])]
        if values.get("retry") == "fresh":
            cmd.append("--fresh")
        for key, flag in (("inbox", "--inbox"), ("outputs", "--outputs"), ("state", "--state")):
            if values.get(key):
                cmd += [flag, values[key]]
        if dry_run:
            cmd.append("--dry-run")
        return cmd + pipeline_args(values, mode)

    video = values.get("video")
    if not video:
        raise ValueError("Pick a video first.")
    video_path = paths.resolve(video)
    # Single-video results always get their own folder in data/scratch/, so
    # they never mix into the inbox or overwrite a finished batch clip.
    cmd = ["pipeline.py", _relative(video_path),
           "--out-dir", _relative(single_output_dir(video_path))]
    cmd.append("--fresh" if values.get("retry") == "fresh" else "--resume")
    return cmd + pipeline_args(values, mode)


def check_against_parser() -> list[str]:
    """Problems with this file, e.g. a field whose flag no longer exists."""
    actions = {a.dest for a in _parser()._actions if a.option_strings}
    panel_only = set(PANEL_DEFAULTS)
    return [f"field '{f['key']}' has no matching pipeline option"
            for f in FIELDS if f["key"] not in actions and f["key"] not in panel_only]


# ---------------------------------------------------------------------------
# The ⓘ notes: what changing each option does to what you get, in plain
# English. Every field must have one; check_info() enforces it.
# ---------------------------------------------------------------------------
INFO: dict[str, str] = {
    "style": (
        "This decides the personality of the narrator. Attenborough sounds like a calm, "
        "dry nature documentary. Sports sounds like an excited commentator calling a match. "
        "Noir sounds like a tired detective describing a crime. Reality TV sounds like a "
        "dramatic confession to the camera. The music mood follows this choice unless you "
        "pick a different one under Music."
    ),
    "length": (
        "This tells the script writer how long the clip is, so it knows how many words to "
        "write: about 2.5 words per second. Leave it on automatic and the length is read "
        "from the video. If you set it longer than the real clip, the narration will run "
        "past the end of the video; shorter, and there will be silence at the end."
    ),
    "temperature": (
        "This controls how adventurous the script writer is. A lower number gives safer, "
        "more predictable jokes that stick closely to what happens in the video. A higher "
        "number gives wilder, more surprising lines, but they are more likely to go off "
        "topic or stop making sense."
    ),
    "no_title": (
        "When this is on, the script writer also makes up a short, funny title for the "
        "clip, two to six words, and the narrator reads it aloud before the narration. "
        "It also sits at the top of the script file. The script is kept a few words "
        "shorter so the voiceover still fits the clip. Turn it off for narration only."
    ),
    "narrate": (
        "When this is on, the script is read aloud and saved as a voiceover file. When it "
        "is off, you only get the written script, which is much faster and handy for "
        "trying out styles before spending time on the voice."
    ),
    "engine": (
        "This picks the program that reads the script aloud. Chatterbox copies the voice "
        "in the clip you choose below, including its pace and pauses, but takes 10 to 20 "
        "seconds per clip. Kokoro uses a ready-made British narrator voice and is about "
        "ten times faster, but it cannot sound like a specific person."
    ),
    "reference_audio": (
        "Chatterbox imitates this recording. It copies not just the sound of the voice "
        "but also how fast it talks and how long it pauses, so a slow, soft-spoken clip "
        "gives a slow, soft-spoken narration. A clean 8 to 15 second clip with no music "
        "or background noise works best. Choosing the built-in voice uses Chatterbox's "
        "own default narrator instead."
    ),
    "exaggeration": (
        "This sets how much emotion goes into the delivery. Lower numbers sound flatter "
        "and more deadpan, which suits the dry Attenborough humor. Higher numbers sound "
        "more animated and dramatic, which suits sports or reality TV. Go too high and "
        "it can start to sound over-acted."
    ),
    "cfg_weight": (
        "This balances sticking to the words against sounding natural. A lower number "
        "gives the voice more freedom, so it breathes and pauses more naturally; try "
        "lowering it if the narration rushes and ignores commas and ellipses. A higher "
        "number makes the delivery tighter and closer to the voice clip, but it can sound "
        "a little stiff."
    ),
    "voice": (
        "This picks which ready-made Kokoro narrator reads the script. George is warm "
        "and mature, the closest to a classic documentary voice. Lewis is younger and "
        "chattier, Daniel is crisper and quicker, and Fable sounds like a storyteller."
    ),
    "speed": (
        "This sets how fast Kokoro talks. Slightly slower than 1.0 adds gravitas and "
        "suits documentary narration. Faster adds energy, but it also means fewer "
        "dramatic pauses and a shorter voiceover."
    ),
    "max_pause": (
        "This caps how long the silence between sentences can be. The Attenborough "
        "style uses lots of dramatic pauses, and in a short clip long pauses feel like "
        "dead air. A smaller number keeps things moving; a larger one keeps more drama. "
        "No limit keeps every pause exactly as the voice engine made it."
    ),
    "score": (
        "When this is on, a string quartet piece is composed for the clip and saved as "
        "separate music files, already turned down to sit under the narration. It needs "
        "the music server running and adds roughly a minute per take."
    ),
    "music_takes": (
        "This sets how many different versions of the music you get to choose from. "
        "Each take uses a slightly different tempo, key and texture, so they really "
        "sound different. More takes give you more choice but take longer, about a "
        "minute each."
    ),
    "music_style": (
        "This sets the mood of the music. By default it matches the narration style: "
        "a stately quartet for Attenborough, a driving one for sports, a brooding one "
        "for noir and a tense one for reality TV. Pick a different mood to mix them, "
        "for example noir music under an Attenborough script."
    ),
    "retry": (
        "This matters only for clips that already ran, for example one that failed "
        "halfway. Resume keeps every stage that already finished, such as a good "
        "voiceover, and only redoes what is missing or what your new settings change. "
        "Redo every stage starts the clip over completely; the old files are moved into "
        "an attempt folder, not deleted."
    ),
    "limit": (
        "This sets how many videos from the inbox get processed in this run, starting "
        "from the top of the list. Pick just the first one to test new settings on a "
        "single clip before running the whole inbox."
    ),
    "video": (
        "This is the single video to narrate. You can pick a test clip from "
        "data/scratch, a video waiting in the inbox, or a finished clip to try it again "
        "with different settings. Choose Custom to paste the full path of any other "
        "video on your Mac. The results are saved in their own folder, "
        "data/scratch/<video name>/, and the video itself is never moved or changed."
    ),
    "fps": (
        "This sets how many still frames per second the vision model looks at. More "
        "frames help it catch quick action like pouncing or zoomies, but take longer and "
        "use more memory. Fewer frames are fine for slow scenes like sleeping or grooming."
    ),
    "max_frames": (
        "This caps the total number of frames sent to the vision model, however long "
        "the clip is. More frames can mean a more detailed description, but slower runs "
        "and a higher chance of running out of room. 16 to 24 is the sweet spot."
    ),
    "num_ctx": (
        "This is how much the vision model can take in at once, and each frame uses "
        "about 500 of it. If a run fails saying the request does not fit, raise this or "
        "send fewer frames. A bigger value uses more memory but does not change the "
        "description by itself."
    ),
    "vision_model": (
        "This is the model that watches the video and describes it. Change it only if "
        "you have installed a different vision model in Ollama. A different model will "
        "notice different details and may describe your cats differently."
    ),
    "text_model": (
        "This is the model that writes the script and plans the music. Change it only "
        "if you have installed a different text model in Ollama. A bigger model usually "
        "writes funnier, more coherent scripts but runs slower."
    ),
    "target_lufs": (
        "This sets how loud the voiceover file is. −16 is the usual level for "
        "short-form video. A number closer to zero is louder. The music is turned down "
        "relative to this, so changing it moves both. Don't normalize leaves the voice "
        "at whatever level the engine produced, which is usually too quiet."
    ),
    "max_chars": (
        "Chatterbox can garble long passages into nonsense, so the script is read in "
        "chunks of about this many characters and stitched together. If the end of a "
        "narration turns into gibberish, lower this. If the joins between chunks sound "
        "abrupt, raise it."
    ),
    "device": (
        "This picks which part of the Mac runs Chatterbox. Automatic uses the Apple "
        "GPU, which is fastest. Switch to CPU if you see an error mentioning Metal; it is "
        "slower but always works. The voice sounds the same either way."
    ),
    "no_music_vary": (
        "With one request per take, each take gets its own tempo, key and texture, so "
        "you get real alternatives to choose from. One batched request is faster, but "
        "the takes come back sounding almost identical."
    ),
    "no_music_llm": (
        "When on, the text model reads the video description and adjusts the music's "
        "mood and texture to fit what happens in the clip. The bare preset skips that "
        "step and uses the standard music for the style, which is faster and always "
        "sounds the same for a given style."
    ),
    "music_thinking": (
        "When on, the music generator first plans the arrangement before composing. "
        "This tends to give more musical, better-structured pieces, but it uses more "
        "memory and takes longer."
    ),
    "music_bpm": (
        "This sets the tempo of the music, in beats per minute. By default each style "
        "picks a suitable tempo, slow for Attenborough and fast for sports. Setting it "
        "yourself overrides that for every take."
    ),
    "music_seed": (
        "The seed is the starting point for the music generator's randomness. Random "
        "gives new music every time. A fixed number gives the same music again for the "
        "same settings, which is useful when you want to hear what changing one other "
        "setting does."
    ),
    "music_lufs": (
        "This sets how loud the music is. By default it sits 9 units below the "
        "voiceover, quiet enough to stay in the background. A number closer to zero "
        "makes the music more present; around −22 it starts to compete with the voice."
    ),
    "music_bit_depth": (
        "This is the audio quality of the music files. 24-bit is recommended. 16-bit "
        "makes smaller files but can add a faint grittiness to quiet, sustained strings. "
        "32-bit makes larger files with no audible benefit."
    ),
    "save_music_prompt": (
        "When on, a small text file is saved next to each music take describing the "
        "instructions it was made from (mood, tempo, key). It doesn't change the music; "
        "it is only for your reference."
    ),
    "music_timeout": (
        "This is how long to wait for one music take before treating it as failed. A "
        "longer wait helps if the server is slow; a shorter one gives up sooner if "
        "something is stuck. It doesn't change the music."
    ),
    "music_server": (
        "This is the address of the music server on your Mac. Change it only if you "
        "started the server on a different port."
    ),
    "inbox": (
        "This is the folder the batch reads new videos from. Change it only if you "
        "keep a separate queue somewhere else."
    ),
    "outputs": (
        "This is where each finished clip's folder is created. Change it only if you "
        "want this batch's results kept somewhere separate."
    ),
    "state": (
        "This file holds the clip-number counter. Pointing it somewhere else starts a "
        "separate numbering. Don't change it unless you know you want that."
    ),
}


def check_info() -> list[str]:
    return [f"field '{f['key']}' has no ⓘ note" for f in FIELDS if f["key"] not in INFO]


if __name__ == "__main__":
    problems = check_against_parser() + check_info()
    print("\n".join(problems) or "all fields match the pipeline's options")
    print(json.dumps(starter_profiles(), indent=2))
    sys.exit(1 if problems else 0)
