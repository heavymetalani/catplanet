#!/usr/bin/env python3
"""
describe_video.py

Extract frames from a cat video and ask MiniCPM-V 4.5 (running locally
via Ollama) what is happening in the clip. The output is a vivid,
concrete description of the action, body language, and environment,
laid out as 3 to 6 key moments in order, intended to be fed into a
second-stage script-writing model.

Cat identification
------------------
The prompt is built from the "cats" list in config/local_settings.json, which is
gitignored so your cats' names stay private (copy config/local_settings.example.json
to start). Each entry has a name, pronouns, and an appearance description.
The vision model is told to use the name whenever it can identify a cat by
coat pattern, and to fall back to "one of the cats" when uncertain. Remove
the list to disable identification.

Usage
-----
    python -m catplanet.describe_video path/to/video.mp4
    python -m catplanet.describe_video path/to/video.mp4 --fps 4 --max-frames 24
    python -m catplanet.describe_video path/to/video.mp4 --prompt "Just list actions, no commentary."

Requires
--------
    - Ollama running at http://localhost:11434 with openbmb/minicpm-v4.5 pulled.
    - ffmpeg on PATH.
    - Python deps from requirements.txt installed in a venv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ollama

from . import local_settings

DEFAULT_MODEL = "openbmb/minicpm-v4.5"

# Ollama defaults to a 4096-token context, and images are expensive: MiniCPM-V
# spends roughly 450-500 tokens per frame, so the default 24 frames need ~11.5k
# tokens and Ollama rejects the request outright. Ask for room explicitly.
# If you hit "exceeds the available context size", either raise this or lower
# --max-frames; both work, and fewer frames is the cheaper of the two.
DEFAULT_NUM_CTX = 16384

# MiniCPM-V 4.5 can "think" before it answers, and Ollama turns that on unless
# told otherwise. Left on, it can think until the context is full and never
# write the description: one clip spent 30 minutes on 21,000 tokens of
# thinking and came back empty. The description needs no reasoning, so it's off.
THINK = False

# A cap on how much the model may write, so a runaway reply costs a minute or
# two, not half an hour. A key-moments description runs ~400-800 tokens.
DEFAULT_NUM_PREDICT = 1536

# Repeat penalty: off (1.0). The old frame-by-frame prompt got the model stuck
# restating one frame ("In the twenty-twenty-tenth frame, the cat is sitting
# upright...") until it hit the cap; asking for key moments fixed that on its
# own. A penalty made things worse: on the same clip and seed, 1.1 gave
# 150-word run-on sentences and dropped the cats' names to "both animals",
# and 1.05 repeated its "Funniest moment" line and invented a human foot.
# If looping comes back on some clip, try 1.05 before anything stronger.
REPEAT_PENALTY = 1.0
REPEAT_LAST_N = 256

# Everything about the model call that changes what comes back, for the
# pipeline's settings fingerprint.
GENERATION_SETTINGS = {
    "think": THINK,
    "num_predict": DEFAULT_NUM_PREDICT,
    "temperature": 0.3,
    "repeat_penalty": REPEAT_PENALTY,
    "repeat_last_n": REPEAT_LAST_N,
}

# Prompt template. {cat_block} is replaced with cat-identification rules built
# from config/local_settings.json, or an empty string if no cats are configured.
_PROMPT_TEMPLATE = """You are analyzing a short video of a domestic cat (or cats) in an apartment.
The frames below are in chronological order, sampled uniformly across the clip.
{cat_block}
Describe the clip as 3 to 6 key moments, in the order they happen. Do not go
frame by frame: a moment covers a stretch of the clip in which something
changes, and many frames belong to the same moment. Never repeat a moment.

Use exactly this layout:

Setting: one or two sentences on the room and the objects in it.

Moments:
1. (start) What happens, in two to four sentences.
2. (early) ...
3. (middle) ...
4. (late) ...
5. (end) ...

Funniest moment: one sentence on the best comedic material in the clip.

Label each moment with where it falls in the clip: start, early, middle,
late or end. For each moment, cover:
  - Each cat's actions and motion (walking, pouncing, grooming, stretching, etc.)
  - Body language: ears (forward / back / flat), tail (up, twitching, puffed), eyes (wide, squinting, dilated), posture (crouched, loaf, belly-up)
  - Objects the cat interacts with

Be specific, concrete, and vivid. Avoid hedging language like "seems" or "appears".
Describe only what is in the frames; write no narration or jokes yourself.
This description will be used as input for a funny Animal-Planet-style voiceover,
so call out anything that would make good comedic material (dramatic poses,
suspicious glances, attempted crimes against furniture, etc.).
"""


def load_cats(path: Path | None = None) -> list[dict]:
    """Load cat identities. Returns [] if none are configured.

    By default they come from the "cats" list in config/local_settings.json, which is
    gitignored so your cats' names stay private. Pass `path` to read a separate
    file in the same {"cats": [...]} shape instead.
    """
    if path is None:
        cats = local_settings.get("cats", [])
    else:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"[warn] could not parse {path}: {e}. Falling back to generic prompt.",
                file=sys.stderr,
            )
            return []
        cats = data.get("cats", [])
    if not isinstance(cats, list):
        return []
    # Sanity-filter any entries missing required fields.
    return [
        c for c in cats
        if isinstance(c, dict) and all(k in c for k in ("name", "pronouns", "appearance"))
    ]


def _format_cat_block(cats: list[dict]) -> str:
    """Build the cat-identification portion of the prompt. Empty string if no cats."""
    if not cats:
        return ""

    lines = ["\nThe cats in this household:"]
    for cat in cats:
        lines.append(f"  - **{cat['name']}** ({cat['pronouns']}): {cat['appearance']}")

    lines.append("")
    lines.append("When you see a cat in the frame, identify them by name based on coat pattern:")
    for cat in cats:
        lines.append(
            f"  - If the cat matches {cat['name']}'s description, refer to them as "
            f"\"{cat['name']}\" and use {cat['pronouns']} pronouns."
        )
    lines.append(
        "  - If both cats appear in the same frame, distinguish them by name throughout."
    )
    lines.append(
        "  - If you cannot confidently tell which cat it is (partial view, bad angle, "
        "unusual lighting), say \"one of the cats\" rather than guessing."
    )
    lines.append("")
    return "\n".join(lines)


def build_prompt(cats: list[dict] | None = None) -> str:
    """Assemble the full description prompt, weaving in cat identification rules."""
    if cats is None:
        cats = load_cats()
    return _PROMPT_TEMPLATE.format(cat_block=_format_cat_block(cats))


# Built at import time so pipeline.py can import DEFAULT_PROMPT as before.
DEFAULT_PROMPT = build_prompt()


def extract_frames(video_path: Path, fps: float, out_dir: Path) -> list[Path]:
    """Extract frames at the given fps using ffmpeg. Returns sorted frame paths."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with `brew install ffmpeg`."
        )

    out_pattern = out_dir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        str(out_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return sorted(out_dir.glob("frame_*.jpg"))


def subsample(frames: list[Path], max_frames: int) -> list[Path]:
    """Keep first and last frames, subsample the middle evenly."""
    if len(frames) <= max_frames:
        return frames
    # Evenly pick max_frames indices from [0, len(frames)-1]
    step = (len(frames) - 1) / (max_frames - 1)
    indices = [round(i * step) for i in range(max_frames)]
    # Deduplicate while preserving order
    seen: set[int] = set()
    picked = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            picked.append(frames[idx])
    return picked


def describe(
    video_path: Path,
    model: str,
    fps: float,
    max_frames: int,
    prompt: str,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """Extract frames, send them to the vision model, return the description."""
    with tempfile.TemporaryDirectory(prefix="catvid_") as tmp:
        tmp_path = Path(tmp)
        t0 = time.perf_counter()
        frames = extract_frames(video_path, fps, tmp_path)
        if not frames:
            raise RuntimeError(
                f"No frames extracted from {video_path}. Is it a valid video file?"
            )
        frames = subsample(frames, max_frames)
        t_extract = time.perf_counter() - t0

        print(
            f"[info] extracted {len(frames)} frames in {t_extract:.2f}s; "
            f"calling {model}...",
            file=sys.stderr,
            flush=True,
        )

        t1 = time.perf_counter()
        request = dict(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(f) for f in frames],
                }
            ],
            options={
                "temperature": GENERATION_SETTINGS["temperature"],
                "num_ctx": num_ctx,
                "num_predict": GENERATION_SETTINGS["num_predict"],
                "repeat_penalty": GENERATION_SETTINGS["repeat_penalty"],
                "repeat_last_n": GENERATION_SETTINGS["repeat_last_n"],
            },
        )
        try:
            response = ollama.chat(**request, think=THINK)
        except ollama.ResponseError as e:
            # A vision model without a thinking mode may refuse the setting.
            if "think" not in str(e).lower():
                raise
            response = ollama.chat(**request)
        t_infer = time.perf_counter() - t1
        print(
            f"[info] model responded in {t_infer:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        description = (response["message"].get("content") or "").strip()
        cut_off = response.get("done_reason") == "length"
        if not description:
            raise RuntimeError(
                "the vision model returned an empty description"
                + (" (it used its whole output budget without answering)" if cut_off else "")
                + ". Run the clip again; if it keeps happening, try a different "
                "--vision-model."
            )
        if cut_off:
            print(
                f"[warn] the description hit the {GENERATION_SETTINGS['num_predict']}-token "
                f"cap and was cut short; it is still used.",
                file=sys.stderr,
            )
        return description


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Describe a cat video using MiniCPM-V 4.5 via Ollama."
    )
    ap.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Path to video file (mp4, mov, etc.). Not needed with --show-prompt.",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})"
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to sample (default: 2.0)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Max frames sent to the model (default: 24)",
    )
    ap.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=(
            f"Context window for the vision model (default: {DEFAULT_NUM_CTX}). "
            f"Each frame costs ~450-500 tokens. Raise it, or lower --max-frames, "
            f"if Ollama says the request exceeds the available context size."
        ),
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="Override the default description prompt.",
    )
    ap.add_argument(
        "--cats-file",
        type=Path,
        default=None,
        help=(
            'Read cats from this JSON file ({"cats": [...]}) instead of the '
            '"cats" list in config/local_settings.json.'
        ),
    )
    ap.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the built prompt to stderr and exit without calling the model.",
    )
    args = ap.parse_args()

    # Build the prompt (possibly from a custom cats file).
    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = build_prompt(load_cats(args.cats_file))

    if args.show_prompt:
        print(prompt, file=sys.stderr)
        return 0

    if args.video is None:
        ap.error("the following arguments are required: video")
    if not args.video.exists():
        print(f"error: {args.video} does not exist", file=sys.stderr)
        return 1
    if not args.video.is_file():
        print(f"error: {args.video} is not a file", file=sys.stderr)
        return 1

    try:
        description = describe(
            video_path=args.video,
            model=args.model,
            fps=args.fps,
            max_frames=args.max_frames,
            prompt=prompt,
            num_ctx=args.num_ctx,
        )
    except ollama.ResponseError as e:
        msg = str(e)
        if "not found" in msg.lower():
            print(
                f"\nerror: model '{args.model}' is not pulled.\n"
                f"Run: ollama pull {args.model}",
                file=sys.stderr,
            )
        else:
            print(f"ollama error: {msg}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except ConnectionError:
        print(
            "error: could not connect to Ollama at http://localhost:11434\n"
            "Start it with: brew services start ollama   (or: ollama serve)",
            file=sys.stderr,
        )
        return 3

    print(description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
