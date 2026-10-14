#!/usr/bin/env python3
"""
generate_script.py

Stage 2 of the cat-video pipeline. Takes a detailed description of what
is happening in a clip (produced by describe_video.py) and writes a
funny short-form voiceover script in the requested style.

Usage
-----
    # From a file
    python generate_script.py --description descriptions/zoomies.txt --length 15

    # From stdin (chain with stage 1)
    python describe_video.py samples/zoomies.mp4 \\
        | python generate_script.py --length 15

    # Pick a style
    python generate_script.py --description d.txt --length 20 --style noir

Styles
------
    attenborough  - deadpan nature-documentary narrator (default)
    sports        - excitable play-by-play sportscaster
    noir          - hardboiled 1940s detective voiceover
    reality_tv    - dramatic reality-TV confessional
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ollama

DEFAULT_MODEL = "qwen2.5:14b"

STYLES: dict[str, str] = {
    "attenborough": """You are a comedy writer producing a short-form video voiceover in the style
of a David Attenborough nature documentary — but applied to domestic cats,
with a deadpan, absurdly dramatic tone.

Voice & tone:
  - Dry, understated, documentary-narrator delivery.
  - Treat mundane cat behavior with epic seriousness.
  - Sprinkle in one or two absurd "scientific" observations per clip
    (fake species names, invented evolutionary explanations, etc.).
  - Never break character. Never explain the joke.

Cadence — this is critical, the output will be read aloud by a TTS engine
that follows punctuation literally:
  - Use SHORT declarative sentences. Fragments are good. No run-ons.
  - Use an ellipsis ("...") for a dramatic pause of about one beat.
    Place these BEFORE key revelations or punchlines.
  - Use an em dash (" — ") for a shorter mid-thought pause.
  - End sentences with a period so the narrator breathes.
  - Break into 2-4 short paragraphs for a clip of ~15-30 seconds;
    each paragraph break adds a longer breath.
  - Prefer concrete nouns and specific actions over adjectives.

Vocabulary Attenborough actually uses: "here we see...", "observe...",
"the creature...", "in an astonishing display of...", "and yet...",
"having assessed...", "what appears to be...".

Example beat structure (for a 15-second clip):

    Here... we observe the domestic feline. A creature of habit.

    It has chosen, once again — against all reason — to occupy
    the empty box beside the couch.

    Evolution... offers no explanation.""",

    "sports": """You are a high-energy sports play-by-play announcer calling a dramatic
moment live. Your subject: a house cat doing ordinary cat things.

Style rules:
  - Fast, breathless, building tension and payoff.
  - Use sports cliches ("and here comes the crossover", "textbook form", "he's gonna do it").
  - Treat every cat action as a decisive championship moment.
  - Short beats. Lots of exclamation.
  - Never break character.""",

    "noir": """You are a hardboiled 1940s film-noir detective narrating a case. The case
happens to be a house cat doing cat things, but you deliver it like the
city is on the line.

Style rules:
  - First person, weary, cigarette-in-voice.
  - Rain-soaked metaphors. Distrust everyone, including the cat.
  - Short sentences. Heavy pauses implied by periods.
  - Never break character.""",

    "reality_tv": """You are a reality-TV confessional writer. The "cast member" is a house cat,
and every small action is a betrayal, a power play, or a personal journey.

Style rules:
  - Petty, dramatic, over-invested in minor events.
  - Mix confessional voice ("I knew. I just KNEW.") with narrator beats.
  - Short, clipped sentences for punch.
  - Never break character.""",
}

USER_PROMPT_TEMPLATE = """Below is a detailed description of a {length}-second short-form cat video.

--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---

Write the voiceover narration for this clip.

Constraints:
  - Target length: approximately {word_count} words (matches roughly {length} seconds at a lively short-form pace).
  - Output ONLY the spoken narration words. No stage directions, no bracketed cues, no timestamps, no section headers, no explanation.
  - Do not address the cat's name (you don't know it).
  - If the description mentions multiple distinct moments, weave the narration to hit them in order.

Begin."""


def build_system_prompt(style: str) -> str:
    if style not in STYLES:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: {', '.join(STYLES)}"
        )
    return STYLES[style]


def generate_script(
    description: str,
    video_length: int,
    style: str,
    model: str,
    words_per_second: float = 2.5,
    temperature: float = 0.9,
) -> str:
    """Call the text LLM and return the generated voiceover narration."""
    system_prompt = build_system_prompt(style)
    word_count = max(12, int(video_length * words_per_second))

    user_prompt = USER_PROMPT_TEMPLATE.format(
        length=video_length,
        description=description.strip(),
        word_count=word_count,
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "top_p": 0.95,
        },
    )
    return response["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a funny voiceover script from a video description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--description",
        type=str,
        default=None,
        help="Path to a file with the description. Omit or pass '-' to read from stdin.",
    )
    ap.add_argument(
        "--length",
        type=int,
        required=True,
        help="Video length in seconds (drives target word count).",
    )
    ap.add_argument(
        "--style",
        choices=sorted(STYLES.keys()),
        default="attenborough",
        help="Voiceover style preset (default: attenborough).",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama text model (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--wps",
        type=float,
        default=2.5,
        help="Target words per second of video (default: 2.5).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9). Higher = wilder.",
    )
    args = ap.parse_args()

    # Read description from file or stdin
    if args.description is None or args.description == "-":
        description = sys.stdin.read()
    else:
        desc_path = Path(args.description)
        if not desc_path.exists():
            print(f"error: description file {desc_path} not found", file=sys.stderr)
            return 1
        description = desc_path.read_text()

    if not description.strip():
        print("error: description is empty", file=sys.stderr)
        return 1

    print(
        f"[info] generating {args.style} script for {args.length}s clip via {args.model}...",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        script = generate_script(
            description=description,
            video_length=args.length,
            style=args.style,
            model=args.model,
            words_per_second=args.wps,
            temperature=args.temperature,
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
    except ConnectionError:
        print(
            "error: could not connect to Ollama at http://localhost:11434\n"
            "Start it with: brew services start ollama   (or: ollama serve)",
            file=sys.stderr,
        )
        return 3

    elapsed = time.perf_counter() - t0
    print(f"[info] generated in {elapsed:.2f}s", file=sys.stderr, flush=True)
    print(script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
