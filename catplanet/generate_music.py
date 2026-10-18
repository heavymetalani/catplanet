#!/usr/bin/env python3
"""
generate_music.py

Stages 2b and 4 of the pipeline: score a clip with a classical string quartet.

  Stage 2b  build_music_spec()   description -> caption + BPM + key   (Qwen2.5)
  Stage 4   generate_music()     caption     -> <name>.music.wav      (ACE-Step)

Stage 4 talks to ACE-Step over HTTP rather than importing it. ACE-Step lives in
its own venv (MUSIC GEN/ACE-Step-1.5/.venv) with its own pinned torch/MLX that
would fight kokoro and chatterbox if merged. Start the server first:

    "MUSIC GEN/start_api_server.command"

How the caption is built
------------------------
A preset keyed on the voiceover style fixes the frame -- tempo band, mode, and
the technique vocabulary. The LLM then adjusts *within* that frame, returning
JSON only; Python assembles the final caption deterministically. The LLM never
writes the caption itself, because a freehand caption drifts toward naming
instruments a string quartet does not have.

Usage
-----
    # Full stage 2b + 4, from a saved description
    python -m catplanet.generate_music --description data/scratch/five.description.txt \\
        --length 18 --style attenborough --out data/scratch/five.music.wav

    # Chain from stage 1
    python -m catplanet.describe_video data/scratch/five.mov \\
        | python -m catplanet.generate_music --length 18 --out five.music.wav

    # Skip the LLM -- bare preset only (fast, deterministic)
    python -m catplanet.generate_music --length 18 --style noir --no-llm --out cue.wav

    # Audition three takes
    python -m catplanet.generate_music --description d.txt --length 18 --takes 3 --out cue.wav

    # Match the bed to an existing voiceover (targets 9 LU below it)
    python -m catplanet.generate_music --description d.txt --length 18 \\
        --voiceover data/scratch/five.voiceover.wav --out data/scratch/five.music.wav

Requires
--------
    - ACE-Step API server on http://127.0.0.1:8001
    - Ollama with qwen2.5:14b (unless --no-llm)
    - ffmpeg + ffprobe on PATH
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path

import ollama
import requests

from . import audio_utils, progress

# =========================================================
# Defaults
# =========================================================
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_SERVER = "http://127.0.0.1:8001"
DEFAULT_TEMPERATURE = 0.7

# ACE-Step generation settings. Turbo wants 8 steps; see MUSIC GEN/instructions.md 4.1.
DEFAULT_STEPS = 8
DEFAULT_INFER_METHOD = "ode"

# ACE-Step's floor is 10s and short durations are unstable -- the model rushes a
# musical arc into the window. Generate long, trim back.
MIN_REQUEST_DURATION = 20
DURATION_PAD = 4

# Bed level. Voiceover-relative when we have a voiceover to measure.
DEFAULT_LUFS = -25.0
VOICEOVER_OFFSET_LU = 9.0

# Never let true peak exceed this after gain. We normally attenuate, so this
# only bites if someone asks for a loud target.
TRUE_PEAK_CEILING_DB = -2.0

# Bed-level audio in 16-bit wastes most of the word: at a -21 dBFS peak only
# ~12 bits are in use, and sustained strings pick up audible quantization
# grit. 24-bit costs 50% more disk and removes the problem outright.
DEFAULT_BIT_DEPTH = 24
_BIT_DEPTH_CODECS = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}

FADE_IN = 0.5
FADE_OUT = 1.5

POLL_INTERVAL = 2.0
DEFAULT_TIMEOUT = 300.0


# =========================================================
# Presets -- the frame the LLM may not leave
# =========================================================
@dataclass
class Preset:
    """A musical frame keyed to a voiceover style."""
    character: str
    bpm_range: tuple[int, int]
    keys: list[str]

    @property
    def default_bpm(self) -> int:
        lo, hi = self.bpm_range
        return (lo + hi) // 2

    @property
    def default_key(self) -> str:
        return self.keys[0]


# Note on tempo: the character strings below carry NO Italian tempo markings.
# ACE-Step weights caption words heavily -- heavier than the bpm field, which
# the guide calls "an anchor, not a command" (instructions.md 4.5). An earlier
# version said "stately adagio" here and the cues came out sluggish no matter
# what bpm we sent, because the caption was arguing with the metadata. Tempo
# now lives only in bpm_range; the captions describe motion and energy instead.
PRESETS: dict[str, Preset] = {
    "attenborough": Preset(
        character=(
            "sustained legato lines over a steady walking pulse, gentle "
            "pizzicato underneath, patient and quietly dignified, always "
            "moving forward"
        ),
        bpm_range=(76, 96),
        keys=["C Major", "G Major", "D Major", "A Minor"],
    ),
    "sports": Preset(
        character=(
            "rapid spiccato bowing over a driving cello ostinato, breathless "
            "and building, relentless forward drive"
        ),
        bpm_range=(132, 168),
        keys=["D Major", "A Major", "E Minor"],
    ),
    "noir": Preset(
        character=(
            "brooding and sparse, muted sul tasto violins, chromatic "
            "descending cello line over a walking bass, unresolved"
        ),
        bpm_range=(84, 104),
        keys=["D Minor", "C Minor", "B Flat Minor"],
    ),
    "reality_tv": Preset(
        character=(
            "tense tremolo strings, sudden unison stabs, suspensions that "
            "refuse to resolve, restless"
        ),
        bpm_range=(100, 124),
        keys=["A Minor", "E Minor", "F Major"],
    ),
}

# How takes differ from one another. Take 1 is the spec as written; each later
# take gets an accent phrase, a BPM nudge, the next key in the preset, and a
# different seed. Seed alone was not enough -- batched takes came back nearly
# identical because the caption and metadata dominate the conditioning.
VARIATION_ACCENTS = [
    "",
    "sparser, more space between phrases, fewer voices sounding at once",
    "denser and more insistent, busier inner voices",
    "freer phrasing, rhythmically loose and rubato",
    "tighter and more rhythmic, crisp short articulation",
    "warmer and lower, weighted toward the viola and cello",
    "brighter and higher, weighted toward the two violins",
    "more contrapuntal, voices entering one after another in imitation",
]
VARIATION_BPM_FACTORS = [1.00, 0.90, 1.12, 0.96, 1.07, 0.93, 1.09, 1.03]

# The ensemble and room never change. Only mood/textures/arc do.
ENSEMBLE = "solo classical string quartet, two violins, viola and cello"
ROOM = "chamber recording, warm close-miked acoustic, natural room decay"
NEGATIVES = (
    "instrumental, no percussion, no drums, no piano, no synth, no vocals"
)

# Any texture containing one of these gets dropped. ACE-Step will happily
# arrange a "quartet with a light drum kit" if you let the words through.
BANNED_WORDS = {
    "drum", "drums", "percussion", "kit", "beat", "piano", "keyboard",
    "synth", "synthesizer", "pad", "pads", "guitar", "bass guitar", "brass",
    "trumpet", "horn", "flute", "clarinet", "oboe", "sax", "saxophone",
    "woodwind", "choir", "vocal", "vocals", "voice", "singer", "organ",
    "harp", "accordion", "electronic", "808", "orchestra", "orchestral",
    "full string section", "timpani", "snare", "cymbal",
}

# Static slow-tempo markings fight the bpm field and win, so they are dropped
# the same way. Motion words (accelerando, ritardando) are fine and useful --
# they describe change rather than pinning an absolute tempo.
BANNED_TEMPO_WORDS = {
    "adagio", "adagietto", "largo", "larghetto", "lento", "grave",
    "andante", "andantino", "moderato", "allegro", "presto", "vivace",
    "slow", "slowly", "languid", "unhurried", "leisurely", "sluggish",
}


# =========================================================
# The spec that stage 2b produces and stage 4 consumes
# =========================================================
@dataclass
class MusicSpec:
    caption: str
    bpm: int
    key_scale: str
    style: str
    mood: str = ""
    textures: list[str] = field(default_factory=list)
    arc: str = ""
    llm_used: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "MusicSpec":
        return cls(**json.loads(text))


# =========================================================
# Stage 2b -- constrained LLM adjust
# =========================================================
SYSTEM_PROMPT = """You are a music director. You decide how a classical string \
quartet -- two violins, a viola, and a cello, and nothing else -- should score a \
short video clip.

You reply with a single JSON object and nothing else. No prose, no markdown \
fences, no explanation.

Schema:
{
  "mood":     string,   // 2-6 words. The emotional character of THIS clip.
  "textures": [string], // 2-3 items. String-quartet playing techniques only.
  "arc":      string,   // one short clause: how the cue evolves start to end.
  "bpm":      integer,
  "key_scale": string
}

Hard rules:
  - "textures" must name only techniques a string quartet can play: arco,
    pizzicato, tremolo, sul ponticello, sul tasto, col legno, harmonics,
    double stops, spiccato, legato, glissando, unison, drone, ostinato.
  - Never name an instrument outside the quartet. No drums, piano, synth,
    guitar, brass, woodwind, or voices. There are none available.
  - Write about MUSIC, not about the video. Do not mention cats, animals,
    furniture, or the camera. The music expresses the clip; it does not
    narrate it.
  - Never use Italian tempo markings (adagio, andante, allegro, presto) or
    words like "slow" and "unhurried" in any text field. Tempo is set by the
    bpm field alone, and tempo words in the text override it. Words for
    *changing* speed (accelerando, ritardando) are fine.
  - Keep the cue moving. These are short clips; a cue that sits still reads
    as dead air under narration."""

USER_PROMPT_TEMPLATE = """Below is a description of a {length}-second video clip.

--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---

The cue is already framed as: {character}

Choose how this specific clip departs from that frame.

Constraints:
  - "bpm" must be an integer between {bpm_lo} and {bpm_hi}.
  - "key_scale" must be exactly one of: {keys}
  - Match the energy and shape of what happens in the clip. If the action
    changes partway through, put that turn in "arc".

Reply with the JSON object only."""


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM reply. Tolerates fences/preamble."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])


def _clean_textures(raw, limit: int = 3) -> list[str]:
    """Keep short texture phrases that name no forbidden instrument."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.split()).strip(" .,;")
        if not phrase or len(phrase) > 60:
            continue
        words = set(re.findall(r"[a-z]+", phrase.lower()))
        if words & BANNED_WORDS:
            print(f"[warn] dropped texture naming a non-quartet instrument: {phrase!r}",
                  file=sys.stderr)
            continue
        if words & BANNED_TEMPO_WORDS:
            print(f"[warn] dropped texture pinning a tempo: {phrase!r}", file=sys.stderr)
            continue
        out.append(phrase)
        if len(out) >= limit:
            break
    return out


def _clean_phrase(raw, limit: int = 90) -> str:
    """Sanitize a free-text field the same way, returning '' if unusable."""
    if not isinstance(raw, str):
        return ""
    phrase = " ".join(raw.split()).strip(" .,;")
    if not phrase or len(phrase) > limit:
        return ""
    words = set(re.findall(r"[a-z]+", phrase.lower()))
    if words & BANNED_WORDS:
        print(f"[warn] dropped phrase naming a non-quartet instrument: {phrase!r}",
              file=sys.stderr)
        return ""
    if words & BANNED_TEMPO_WORDS:
        print(f"[warn] dropped phrase pinning a tempo: {phrase!r}", file=sys.stderr)
        return ""
    return phrase


def _match_key(raw, preset: Preset) -> str:
    """Snap the LLM's key to one the preset allows."""
    if not isinstance(raw, str):
        return preset.default_key
    want = raw.strip().lower().replace("-", " ").replace("_", " ")
    want = want.replace("b flat", "b flat").replace("bb", "b flat")
    for key in preset.keys:
        if want == key.lower():
            return key
    # Looser: match on tonic + mode ("Am" -> "A Minor", "c maj" -> "C Major")
    m = re.match(r"^([a-g])\s*(flat|sharp|b|#)?\s*(m|min|minor|maj|major)?", want)
    if m:
        tonic = m.group(1).upper()
        accidental = {"flat": " Flat", "b": " Flat", "sharp": " Sharp", "#": " Sharp"}.get(
            m.group(2), ""
        )
        mode = "Minor" if (m.group(3) or "").startswith("m") and not (
            m.group(3) or ""
        ).startswith("maj") else "Major"
        candidate = f"{tonic}{accidental} {mode}"
        for key in preset.keys:
            if candidate.lower() == key.lower():
                return key
    print(f"[warn] key {raw!r} not in preset; using {preset.default_key}", file=sys.stderr)
    return preset.default_key


def _clamp_bpm(raw, preset: Preset) -> int:
    lo, hi = preset.bpm_range
    try:
        bpm = int(round(float(raw)))
    except (TypeError, ValueError):
        return preset.default_bpm
    if not lo <= bpm <= hi:
        print(f"[warn] bpm {bpm} outside {lo}-{hi}; clamping", file=sys.stderr)
    return max(lo, min(hi, bpm))


def assemble_caption(
    preset: Preset, mood: str = "", textures: list[str] | None = None, arc: str = ""
) -> str:
    """Build the ACE-Step caption. Character words only -- BPM and key go in metadata."""
    parts = [ENSEMBLE, preset.character]
    if mood:
        parts.append(mood)
    if textures:
        parts.extend(textures)
    if arc:
        parts.append(arc)
    parts.extend([ROOM, NEGATIVES])
    return ", ".join(parts)


def build_music_spec(
    description: str,
    video_length: int,
    style: str = "attenborough",
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    use_llm: bool = True,
    release_model: bool = True,
) -> MusicSpec:
    """Stage 2b. Turn a video description into a quartet caption + metadata.

    Never raises on LLM trouble -- falls back to the bare preset, because a
    generic cue beats a dead pipeline.

    release_model=True unloads the model from Ollama afterwards. Stage 2b is
    the last Ollama call, and ACE-Step needs the memory (see docs/MUSIC_PLAN.md section 7).
    """
    if style not in PRESETS:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: {', '.join(sorted(PRESETS))}"
        )
    preset = PRESETS[style]

    if not use_llm or not description.strip():
        return MusicSpec(
            caption=assemble_caption(preset),
            bpm=preset.default_bpm,
            key_scale=preset.default_key,
            style=style,
            llm_used=False,
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        length=video_length,
        description=description.strip(),
        character=preset.character,
        bpm_lo=preset.bpm_range[0],
        bpm_hi=preset.bpm_range[1],
        keys=", ".join(preset.keys),
    )

    data: dict | None = None
    for attempt in (1, 2):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                # Same context as stage 2, so Ollama keeps the model loaded
                # instead of reloading it with a different context size.
                options={"temperature": temperature, "top_p": 0.9, "num_ctx": 8192},
                # Unload after the final attempt; ACE-Step wants the memory.
                keep_alive=0 if (release_model and attempt == 2) else "5m",
            )
            data = _extract_json(response["message"]["content"])
            break
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[warn] stage 2b attempt {attempt}: bad JSON ({e})", file=sys.stderr)
        except Exception as e:  # ollama down, model missing, etc.
            print(f"[warn] stage 2b attempt {attempt} failed: {e}", file=sys.stderr)
            break

    if release_model:
        release_ollama_model(model)

    if data is None:
        print("[warn] stage 2b falling back to bare preset", file=sys.stderr)
        return MusicSpec(
            caption=assemble_caption(preset),
            bpm=preset.default_bpm,
            key_scale=preset.default_key,
            style=style,
            llm_used=False,
        )

    mood = _clean_phrase(data.get("mood"), limit=60)
    textures = _clean_textures(data.get("textures"))
    arc = _clean_phrase(data.get("arc"), limit=90)
    bpm = _clamp_bpm(data.get("bpm"), preset)
    key_scale = _match_key(data.get("key_scale"), preset)

    return MusicSpec(
        caption=assemble_caption(preset, mood, textures, arc),
        bpm=bpm,
        key_scale=key_scale,
        style=style,
        mood=mood,
        textures=textures,
        arc=arc,
        llm_used=True,
    )


@dataclass
class TakeVariant:
    """One audition take: a spec, a seed, and a sampler."""
    spec: MusicSpec
    seed: int | None
    infer_method: str
    label: str


def make_take_variants(
    spec: MusicSpec, takes: int, base_seed: int | None = None
) -> list[TakeVariant]:
    """Build `takes` genuinely different versions of one cue.

    Asking ACE-Step for batch_size=N returns N takes that differ only by seed,
    and they come back sounding nearly identical: the caption and metadata
    dominate the conditioning, so the seed barely moves the result. Real
    variation needs the conditioning itself to differ, so each take gets an
    accent phrase, a BPM nudge, the next key in the preset, and the stochastic
    sampler ("sde" -- ode is the deterministic one, per instructions.md 5).
    """
    preset = PRESETS[spec.style]
    lo, hi = preset.bpm_range
    variants: list[TakeVariant] = []

    for i in range(max(1, takes)):
        accent = VARIATION_ACCENTS[i % len(VARIATION_ACCENTS)]
        factor = VARIATION_BPM_FACTORS[i % len(VARIATION_BPM_FACTORS)]
        bpm = max(lo, min(hi, int(round(spec.bpm * factor))))
        key = preset.keys[i % len(preset.keys)] if i else spec.key_scale

        textures = list(spec.textures)
        if accent:
            textures = textures + [accent]

        variant_spec = MusicSpec(
            caption=assemble_caption(preset, spec.mood, textures, spec.arc),
            bpm=bpm,
            key_scale=key,
            style=spec.style,
            mood=spec.mood,
            textures=textures,
            arc=spec.arc,
            llm_used=spec.llm_used,
        )
        variants.append(
            TakeVariant(
                spec=variant_spec,
                seed=(base_seed + i) if base_seed is not None else None,
                # Take 1 is the reference reading; later takes explore.
                infer_method="ode" if i == 0 else "sde",
                label=accent or "as written",
            )
        )
    return variants


def release_ollama_model(model: str) -> None:
    """Unload a model from Ollama so ACE-Step can have the memory."""
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass  # Best-effort; ollama's idle timeout is the backstop.


# =========================================================
# Stage 4 -- ACE-Step client
# =========================================================
class AceStepError(RuntimeError):
    pass


def check_server(server: str = DEFAULT_SERVER, timeout: float = 5.0) -> None:
    """Raise AceStepError with a usable instruction if the server isn't up."""
    try:
        r = requests.get(f"{server.rstrip('/')}/health", timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AceStepError(
            f"no ACE-Step server at {server} ({e}).\n"
            f'Start it in another terminal and leave it running:\n'
            f'    "MUSIC GEN/start_api_server.command"\n'
            f"First launch loads ~5GB of weights; wait for port 8001 to answer."
        ) from e


def request_music(
    spec: MusicSpec,
    duration: float,
    server: str = DEFAULT_SERVER,
    takes: int = 1,
    seed: int | None = None,
    steps: int = DEFAULT_STEPS,
    thinking: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    infer_method: str = DEFAULT_INFER_METHOD,
) -> list[bytes]:
    """Submit one generation task, poll it, and return the raw .wav bytes."""
    base = server.rstrip("/")
    payload = {
        "prompt": spec.caption,
        "lyrics": "",                       # instrumental
        "audio_duration": float(duration),
        "bpm": spec.bpm,
        "key_scale": spec.key_scale,
        "time_signature": "4",
        "inference_steps": steps,
        "infer_method": infer_method,
        "batch_size": max(1, min(8, takes)),
        "audio_format": "wav",
        "thinking": thinking,
        # Both default to True and let ACE-Step's planner rewrite the caption --
        # which is exactly where "and a light drum kit" gets added back in.
        "use_cot_caption": False,
        "use_cot_language": False,
    }
    if seed is not None:
        payload["use_random_seed"] = False
        payload["seed"] = seed

    try:
        r = requests.post(f"{base}/release_task", json=payload, timeout=60)
        r.raise_for_status()
        task_id = r.json()["data"]["task_id"]
    except (requests.RequestException, KeyError, ValueError) as e:
        raise AceStepError(f"could not submit task: {e}") from e

    deadline = time.monotonic() + timeout
    files: list[str] = []
    while True:
        if time.monotonic() > deadline:
            raise AceStepError(f"timed out after {timeout:.0f}s waiting for {task_id}")
        time.sleep(POLL_INTERVAL)
        try:
            q = requests.post(
                f"{base}/query_result", json={"task_id_list": [task_id]}, timeout=30
            )
            q.raise_for_status()
            entry = q.json()["data"][0]
        except (requests.RequestException, KeyError, IndexError, ValueError):
            continue  # transient; keep polling until the deadline

        status = entry.get("status", 0)
        if status == 2:
            raise AceStepError(f"generation failed: {entry.get('result') or entry}")
        if status == 1:
            try:
                results = json.loads(entry.get("result") or "[]")
                files = [x["file"] for x in results if x.get("file")]
            except (json.JSONDecodeError, TypeError):
                files = []
            break

    if not files:
        raise AceStepError("task succeeded but returned no audio")

    blobs: list[bytes] = []
    for f in files:
        url = f if f.startswith("http") else base + f
        try:
            d = requests.get(url, timeout=120)
            d.raise_for_status()
            blobs.append(d.content)
        except requests.RequestException as e:
            print(f"[warn] could not download {url}: {e}", file=sys.stderr)
    if not blobs:
        raise AceStepError("could not download any generated audio")
    return blobs


# =========================================================
# Post-processing (ffmpeg)
# =========================================================
def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise AceStepError(f"{tool} not found on PATH. Install with: brew install ffmpeg")


# Measurement and gain live in audio_utils, shared with stage 3.
_analyze = audio_utils.analyze
measure_lufs = audio_utils.measure_lufs


def _probe_sample_rate(raw: bytes) -> int | None:
    """Sample rate of piped audio, so post-processing preserves it."""
    if not shutil.which("ffprobe"):
        return None
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", "pipe:0"],
        input=raw, capture_output=True,
    )
    try:
        return int(r.stdout.decode().strip())
    except (ValueError, AttributeError):
        return None


def post_process(
    raw_wav: bytes,
    output_path: Path,
    duration: float,
    lufs_target: float = DEFAULT_LUFS,
    fade_in: float = FADE_IN,
    fade_out: float = FADE_OUT,
    bit_depth: int = DEFAULT_BIT_DEPTH,
) -> Path:
    """Trim to length, fade both ends, and set the bed level with a static gain.

    Loudness is matched by measuring the trimmed audio and applying one
    constant `volume` gain -- not by putting loudnorm in the output chain.
    Single-pass loudnorm is a dynamic processor: it rides the gain and
    look-ahead limits, which on sustained string tones is plainly audible as
    distortion. A static gain is exact here anyway, since we only ever need to
    attenuate down to bed level.
    """
    _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fade_out = min(fade_out, max(0.0, duration / 3))
    fade_in = min(fade_in, max(0.0, duration / 3))
    fade_out_start = max(0.0, duration - fade_out)
    trim = f"atrim=0:{duration:.3f},asetpts=N/SR/TB"

    # Measure what we are about to write, then work out one gain for it.
    measured_i, measured_tp = _analyze(raw_wav, pre_filter=trim)
    gain_db = audio_utils.gain_to_target(
        measured_i, measured_tp, lufs_target,
        ceiling_db=TRUE_PEAK_CEILING_DB, label="music",
    )

    chain = (
        f"{trim},"
        f"afade=t=in:st=0:d={fade_in:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}"
    )
    if abs(gain_db) > 0.01:
        chain += f",volume={gain_db:.2f}dB"

    codec = _BIT_DEPTH_CODECS.get(bit_depth, _BIT_DEPTH_CODECS[DEFAULT_BIT_DEPTH])
    if bit_depth == 16:
        # Only 16-bit needs dither; 24-bit's noise floor is already inaudible.
        chain += ",aresample=out_sample_fmt=s16:dither_method=triangular"

    cmd = ["ffmpeg", "-nostdin", "-y", "-i", "pipe:0", "-af", chain]
    # Pin the output rate to the source's. Without this a resampling filter can
    # silently hand back 192 kHz, quadrupling the file for no benefit.
    sr = _probe_sample_rate(raw_wav)
    if sr:
        cmd += ["-ar", str(sr)]
    # Write under a temporary name and rename when done, so a crash never
    # leaves a cut-off take that looks complete.
    tmp = progress.partial_path(output_path)
    cmd += ["-c:a", codec, "-f", "wav", str(tmp)]

    result = subprocess.run(cmd, input=raw_wav, capture_output=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = result.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise AceStepError("ffmpeg post-processing failed:\n  " + "\n  ".join(tail))
    return progress.commit_partial(tmp, output_path)


def resolve_lufs_target(
    explicit: float | None, voiceover: Path | None
) -> float:
    """Explicit wins; else 9 LU under the voiceover; else the default bed level."""
    if explicit is not None:
        return explicit
    if voiceover and voiceover.exists():
        measured = measure_lufs(voiceover)
        if measured is not None:
            target = measured - VOICEOVER_OFFSET_LU
            print(
                f"[info] voiceover at {measured:.1f} LUFS; "
                f"targeting music at {target:.1f} LUFS",
                file=sys.stderr,
            )
            return target
        print("[warn] could not measure voiceover loudness", file=sys.stderr)
    return DEFAULT_LUFS


# =========================================================
# Public entry point
# =========================================================
def generate_music(
    spec: MusicSpec,
    output_path: Path,
    video_length: float,
    server: str = DEFAULT_SERVER,
    takes: int = 1,
    seed: int | None = None,
    steps: int = DEFAULT_STEPS,
    thinking: bool = False,
    lufs_target: float = DEFAULT_LUFS,
    timeout: float = DEFAULT_TIMEOUT,
    vary: bool = True,
    save_specs: bool = False,
    bit_depth: int = DEFAULT_BIT_DEPTH,
    reuse: dict[int, Path] | None = None,
    on_take=None,
) -> list[Path]:
    """Stage 4. Generate, trim, level. Returns the written paths.

    With takes > 1 the files are numbered <stem>.1.wav, <stem>.2.wav, ...

    reuse maps take numbers to files already finished by an earlier run; those
    takes are skipped and returned as they are (per-take mode only). on_take,
    if given, is called as on_take(number, path) after each take is written,
    so progress survives a crash part-way through.

    vary=True (default) sends one request per take, each with a different
    accent, BPM and key, so the takes are actually distinguishable. That costs
    one generation per take. vary=False falls back to a single batched request
    -- faster, but the takes come back nearly identical.
    """
    check_server(server)
    _require_ffmpeg()

    request_duration = max(MIN_REQUEST_DURATION, video_length + DURATION_PAD)

    def _target(i: int, total: int) -> Path:
        if total == 1:
            return output_path
        return output_path.with_suffix(f".{i}{output_path.suffix}")

    written: list[Path] = []

    if takes <= 1 or not vary:
        blobs = request_music(
            spec, duration=request_duration, server=server, takes=takes,
            seed=seed, steps=steps, thinking=thinking, timeout=timeout,
        )
        for i, blob in enumerate(blobs, start=1):
            target = _target(i, len(blobs))
            written.append(
                post_process(blob, target, duration=video_length,
                             lufs_target=lufs_target, bit_depth=bit_depth)
            )
            if save_specs:
                progress.write_text(target.with_suffix(".json"), spec.to_json())
            if on_take:
                on_take(i, target)
        return written

    reuse = reuse or {}

    variants = make_take_variants(spec, takes, base_seed=seed)
    for i, v in enumerate(variants, start=1):
        if i in reuse:
            print(f"[take {i}/{len(variants)}] reusing {reuse[i].name} from an "
                  f"earlier run", file=sys.stderr, flush=True)
            written.append(reuse[i])
            continue
        print(
            f"[take {i}/{len(variants)}] {v.spec.bpm} BPM, {v.spec.key_scale}, "
            f"{v.infer_method} -- {v.label}",
            file=sys.stderr, flush=True,
        )
        blobs = request_music(
            v.spec, duration=request_duration, server=server, takes=1,
            seed=v.seed, steps=steps, thinking=thinking, timeout=timeout,
            infer_method=v.infer_method,
        )
        target = _target(i, len(variants))
        written.append(
            post_process(blobs[0], target, duration=video_length,
                         lufs_target=lufs_target, bit_depth=bit_depth)
        )
        if save_specs:
            progress.write_text(target.with_suffix(".json"), v.spec.to_json())
        if on_take:
            on_take(i, target)
    return written


def probe_duration(path: Path) -> float | None:
    """Duration in seconds via ffprobe, or None."""
    if not shutil.which("ffprobe") or not path.exists():
        return None
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


# =========================================================
# CLI
# =========================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score a clip with a generated classical string quartet cue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--description",
        type=str,
        default=None,
        help="Path to the stage-1 description. Omit or pass '-' for stdin.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output .wav path.")
    ap.add_argument(
        "--length",
        type=float,
        default=None,
        help="Clip length in seconds. Auto-detected from --video if given.",
    )
    ap.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Clip to match the length of (via ffprobe).",
    )
    ap.add_argument(
        "--style",
        choices=sorted(PRESETS.keys()),
        default="attenborough",
        help="Preset frame, matching the voiceover style (default: attenborough).",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip stage 2b; use the bare preset. Fast and deterministic.",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model for stage 2b (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Stage 2b sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    ap.add_argument(
        "--takes",
        type=int,
        default=1,
        help="Generate N takes to audition (1-8, default 1).",
    )
    ap.add_argument("--seed", type=int, default=None, help="Fixed seed (default: random).")
    ap.add_argument(
        "--bpm",
        type=int,
        default=None,
        help="Override the tempo entirely, ignoring the preset range and the LLM.",
    )
    ap.add_argument(
        "--no-vary",
        action="store_true",
        help=(
            "Send one batched request instead of one per take. Faster, but the "
            "takes come back nearly identical."
        ),
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"DiT inference steps (default: {DEFAULT_STEPS}; turbo wants 8).",
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="Let ACE-Step's 5Hz LM plan the arrangement. Better musicality, more memory.",
    )
    ap.add_argument(
        "--voiceover",
        type=Path,
        default=None,
        help="Voiceover .wav to sit under; music is leveled 9 LU below it.",
    )
    ap.add_argument(
        "--lufs",
        type=float,
        default=None,
        help=f"Explicit loudness target (default: voiceover - 9 LU, else {DEFAULT_LUFS}).",
    )
    ap.add_argument(
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        default=DEFAULT_BIT_DEPTH,
        help=(
            f"Output word length (default: {DEFAULT_BIT_DEPTH}). A bed sits far "
            f"below full scale, so 16-bit wastes most of the word and adds "
            f"audible grit to sustained strings."
        ),
    )
    ap.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"ACE-Step API base URL (default: {DEFAULT_SERVER}).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Seconds to wait for generation (default: {DEFAULT_TIMEOUT:.0f}).",
    )
    ap.add_argument(
        "--save-prompt",
        action="store_true",
        help="Write the assembled caption + metadata next to the audio as .json.",
    )
    ap.add_argument(
        "--print-caption",
        action="store_true",
        help="Print the caption and exit without generating audio.",
    )
    args = ap.parse_args()

    # Length: explicit > probed from video > error.
    length = args.length
    if length is None and args.video is not None:
        length = probe_duration(args.video)
        if length is not None:
            print(f"[info] detected clip length: {length:.1f}s", file=sys.stderr)
    if length is None:
        print("error: pass --length, or --video to probe it", file=sys.stderr)
        return 1

    # Description: file, stdin, or none (bare preset).
    description = ""
    if not args.no_llm:
        if args.description in (None, "-"):
            if not sys.stdin.isatty():
                description = sys.stdin.read()
        else:
            p = Path(args.description)
            if not p.exists():
                print(f"error: {p} not found", file=sys.stderr)
                return 1
            description = p.read_text()
        if not description.strip():
            print("[warn] no description given; using the bare preset", file=sys.stderr)

    # Stage 2b
    if description.strip():
        print(f"[stage 2b] writing {args.style} quartet caption via {args.model}...",
              file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    spec = build_music_spec(
        description=description,
        video_length=int(round(length)),
        style=args.style,
        model=args.model,
        temperature=args.temperature,
        use_llm=not args.no_llm,
    )
    if spec.llm_used:
        print(f"[stage 2b] done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    if args.bpm is not None:
        print(f"[info] overriding {spec.bpm} BPM -> {args.bpm} BPM", file=sys.stderr)
        spec.bpm = args.bpm

    print(f"[info] {spec.bpm} BPM, {spec.key_scale}", file=sys.stderr)
    print(f"[info] caption: {spec.caption}", file=sys.stderr)

    if args.save_prompt:
        spec_path = args.out.with_suffix(".json")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(spec_path, spec.to_json())
        print(f"[info] saved spec to {spec_path}", file=sys.stderr)

    if args.print_caption:
        return 0

    # Stage 4
    lufs = resolve_lufs_target(args.lufs, args.voiceover)
    print(f"[stage 4] generating {args.takes} take(s) via ACE-Step...",
          file=sys.stderr, flush=True)
    t1 = time.perf_counter()
    try:
        written = generate_music(
            spec=spec,
            output_path=args.out,
            video_length=length,
            server=args.server,
            takes=args.takes,
            seed=args.seed,
            steps=args.steps,
            thinking=args.thinking,
            lufs_target=lufs,
            timeout=args.timeout,
            vary=not args.no_vary,
            save_specs=args.save_prompt and args.takes > 1,
            bit_depth=args.bit_depth,
        )
    except AceStepError as e:
        print(f"stage 4 failed: {e}", file=sys.stderr)
        return 2

    print(f"[stage 4] done in {time.perf_counter() - t1:.1f}s", file=sys.stderr)
    for p in written:
        print(f"[saved] {p}", file=sys.stderr)
    print("\n".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
