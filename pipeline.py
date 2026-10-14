#!/usr/bin/env python3
"""
pipeline.py

End-to-end driver: video file in, funny voiceover (text + optional audio) out.

Chains:
  Stage 1   describe_video.describe()         -> detailed description (MiniCPM-V 4.5)
  Stage 2   generate_script.generate_script() -> voiceover narration (Qwen2.5)
  Stage 2b  generate_music.build_music_spec() -> quartet caption (Qwen2.5)   [--score]
  Stage 3   generate_voice.generate_voice()   -> narrated .wav (Kokoro TTS)  [--narrate]
  Stage 4   generate_music.generate_music()   -> string quartet .wav         [--score]

Usage
-----
    python pipeline.py samples/zoomies.mov
    python pipeline.py samples/zoomies.mov --style noir --fps 3 --max-frames 20
    python pipeline.py samples/zoomies.mov --narrate
    python pipeline.py samples/zoomies.mov --narrate --engine kokoro --voice bm_lewis --speed 0.95
    python pipeline.py samples/zoomies.mov --narrate --engine chatterbox
    python pipeline.py samples/zoomies.mov --narrate --engine chatterbox \\
        --reference-audio voices/narrator.wav --exaggeration 0.4
    python pipeline.py samples/zoomies.mov --save-description --save-script --narrate

    # With a string quartet score (needs the ACE-Step server running)
    python pipeline.py samples/zoomies.mov --narrate --score
    python pipeline.py samples/zoomies.mov --narrate --score --music-takes 3

Duration is detected automatically via ffprobe (falls back to --length if that fails).

Stage 4 needs the ACE-Step API server up in a separate terminal:
    "MUSIC GEN/start_api_server.command"

Note on ordering: stage 2b runs immediately after stage 2 because both use the
same Ollama model, so it loads once; Ollama is then released before stage 4 so
ACE-Step gets the memory. See MUSIC_PLAN.md 7.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import describe_video
import generate_music
import generate_script
import generate_voice


def probe_duration(video_path: Path) -> float | None:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Exposed separately so batch.py can validate args up front."""
    ap = argparse.ArgumentParser(
        description="Cat video -> funny voiceover, in one shot."
    )
    ap.add_argument("video", type=Path, help="Path to the video file.")
    ap.add_argument(
        "--style",
        choices=sorted(generate_script.STYLES.keys()),
        default="attenborough",
        help="Voiceover style preset.",
    )
    ap.add_argument(
        "--length",
        type=int,
        default=None,
        help="Override video length in seconds (auto-detected via ffprobe otherwise).",
    )
    ap.add_argument(
        "--vision-model",
        default=describe_video.DEFAULT_MODEL,
        help=f"Vision model (default: {describe_video.DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--text-model",
        default=generate_script.DEFAULT_MODEL,
        help=f"Text model (default: {generate_script.DEFAULT_MODEL}).",
    )
    ap.add_argument("--fps", type=float, default=2.0, help="Frames/sec to sample.")
    ap.add_argument("--max-frames", type=int, default=24, help="Cap on frames.")
    ap.add_argument(
        "--num-ctx",
        type=int,
        default=describe_video.DEFAULT_NUM_CTX,
        help=(
            f"Context window for the vision model (default: "
            f"{describe_video.DEFAULT_NUM_CTX}). Each frame costs ~450-500 tokens."
        ),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temp for the script (default 0.9).",
    )

    # Output persistence toggles
    ap.add_argument(
        "--save-description",
        action="store_true",
        help="Save the stage-1 description alongside the video as <name>.description.txt.",
    )
    ap.add_argument(
        "--save-script",
        action="store_true",
        help="Save the stage-2 script alongside the video as <name>.script.txt.",
    )

    # Stage 3 (TTS) options
    ap.add_argument(
        "--narrate",
        action="store_true",
        help="Run stage 3: synthesize the voiceover to <name>.voiceover.wav.",
    )
    ap.add_argument(
        "--engine",
        choices=["kokoro", "chatterbox"],
        default=generate_voice.DEFAULT_ENGINE,
        help=f"TTS engine (default: {generate_voice.DEFAULT_ENGINE}).",
    )
    ap.add_argument(
        "--audio-out",
        type=Path,
        default=None,
        help="Override the voiceover output path (default: <video>.voiceover.wav).",
    )

    # Kokoro-only
    ap.add_argument(
        "--voice",
        default=generate_voice.DEFAULT_VOICE,
        help=f"Kokoro voice preset (default: {generate_voice.DEFAULT_VOICE}). Ignored if engine=chatterbox.",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Kokoro speed multiplier (default: 1.0). Ignored if engine=chatterbox.",
    )
    ap.add_argument(
        "--target-lufs",
        type=float,
        default=generate_voice.DEFAULT_TARGET_LUFS,
        help=(
            f"Normalize the voiceover to this loudness (default: "
            f"{generate_voice.DEFAULT_TARGET_LUFS}). The music bed is leveled "
            f"relative to it, so this sets both. Pass 0 to skip."
        ),
    )
    ap.add_argument(
        "--max-pause",
        type=float,
        default=generate_voice.DEFAULT_MAX_PAUSE_SEC,
        help=(
            f"Cap pauses between sentences at this many seconds (default: "
            f"{generate_voice.DEFAULT_MAX_PAUSE_SEC}). Both engines. 0 keeps "
            f"the engine's own timing."
        ),
    )

    # Chatterbox-only
    ap.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help="Chatterbox reference .wav for voice cloning (5-10s of target voice).",
    )
    ap.add_argument(
        "--exaggeration",
        type=float,
        default=generate_voice.DEFAULT_EXAGGERATION,
        help=f"Chatterbox emotion intensity 0-1 (default: {generate_voice.DEFAULT_EXAGGERATION}).",
    )
    ap.add_argument(
        "--cfg-weight",
        type=float,
        default=generate_voice.DEFAULT_CFG_WEIGHT,
        help=f"Chatterbox CFG weight 0-1 (default: {generate_voice.DEFAULT_CFG_WEIGHT}).",
    )
    ap.add_argument(
        "--device",
        choices=["mps", "cpu", "cuda"],
        default=None,
        help="Torch device for Chatterbox (default: auto).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=generate_voice.DEFAULT_CHATTERBOX_MAX_CHARS,
        help=(
            f"Chatterbox: max chars per synthesized chunk "
            f"(default: {generate_voice.DEFAULT_CHATTERBOX_MAX_CHARS}). Long scripts "
            f"are split at paragraph/sentence boundaries to avoid garbled output. "
            f"0 disables chunking."
        ),
    )

    # Stages 2b + 4 (music) options
    ap.add_argument(
        "--score",
        action="store_true",
        help=(
            "Run stages 2b and 4: generate a string quartet cue to "
            "<name>.music.wav. Needs the ACE-Step server running."
        ),
    )
    ap.add_argument(
        "--music-out",
        type=Path,
        default=None,
        help="Override the music output path (default: <video>.music.wav).",
    )
    ap.add_argument(
        "--music-style",
        choices=sorted(generate_music.PRESETS.keys()),
        default=None,
        help="Musical preset (default: follows --style).",
    )
    ap.add_argument(
        "--music-takes",
        type=int,
        default=1,
        help="Generate N takes to audition (1-8, default 1). N>1 numbers the files.",
    )
    ap.add_argument(
        "--music-seed",
        type=int,
        default=None,
        help="Fixed seed for the music (default: random).",
    )
    ap.add_argument(
        "--music-bpm",
        type=int,
        default=None,
        help="Override the tempo entirely, ignoring the preset range and the LLM.",
    )
    ap.add_argument(
        "--no-music-vary",
        action="store_true",
        help=(
            "Send one batched request instead of one per take. Faster, but the "
            "takes come back nearly identical."
        ),
    )
    ap.add_argument(
        "--music-lufs",
        type=float,
        default=None,
        help=(
            "Loudness target for the music bed. Default: 9 LU below the "
            f"voiceover, or {generate_music.DEFAULT_LUFS} if there isn't one."
        ),
    )
    ap.add_argument(
        "--music-bit-depth",
        type=int,
        choices=[16, 24, 32],
        default=generate_music.DEFAULT_BIT_DEPTH,
        help=(
            f"Output word length for the music (default: "
            f"{generate_music.DEFAULT_BIT_DEPTH}). A bed sits far below full "
            f"scale, so 16-bit adds audible grit to sustained strings."
        ),
    )
    ap.add_argument(
        "--music-server",
        default=generate_music.DEFAULT_SERVER,
        help=f"ACE-Step API base URL (default: {generate_music.DEFAULT_SERVER}).",
    )
    ap.add_argument(
        "--music-timeout",
        type=float,
        default=generate_music.DEFAULT_TIMEOUT,
        help=f"Seconds to wait for generation (default: {generate_music.DEFAULT_TIMEOUT:.0f}).",
    )
    ap.add_argument(
        "--music-thinking",
        action="store_true",
        help="Let ACE-Step's 5Hz LM plan the arrangement. Better musicality, more memory.",
    )
    ap.add_argument(
        "--no-music-llm",
        action="store_true",
        help="Skip stage 2b; score from the bare preset. Fast and deterministic.",
    )
    ap.add_argument(
        "--save-music-prompt",
        action="store_true",
        help="Save the assembled caption + metadata as <name>.music.json.",
    )

    return ap


def main() -> int:
    args = build_parser().parse_args()

    if not args.video.exists():
        print(f"error: {args.video} does not exist", file=sys.stderr)
        return 1

    # Fail fast: check the music server now, not after three minutes of stages 1-3.
    if args.score:
        try:
            generate_music.check_server(args.music_server)
        except generate_music.AceStepError as e:
            print(f"error: {e}", file=sys.stderr)
            return 5

    # 1. Determine video length
    length: int
    if args.length is not None:
        length = args.length
    else:
        dur = probe_duration(args.video)
        if dur is None:
            print(
                "warning: could not probe duration (ffprobe missing or failed). "
                "Defaulting to 15s. Pass --length to override.",
                file=sys.stderr,
            )
            length = 15
        else:
            length = max(1, round(dur))
            print(f"[info] detected clip length: {length}s", file=sys.stderr)

    # 2. Stage 1 - describe the video
    print("[stage 1] describing video...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        description = describe_video.describe(
            video_path=args.video,
            model=args.vision_model,
            fps=args.fps,
            max_frames=args.max_frames,
            prompt=describe_video.DEFAULT_PROMPT,
            num_ctx=args.num_ctx,
        )
    except Exception as e:
        print(f"stage 1 failed: {e}", file=sys.stderr)
        if "context" in str(e).lower():
            print(
                f"\nThe frames didn't fit in the model's context window.\n"
                f"  Either raise it:      --num-ctx {args.num_ctx * 2}\n"
                f"  Or send fewer frames: --max-frames {max(6, args.max_frames // 2)}\n"
                f"Each frame costs roughly 450-500 tokens, so {args.max_frames} frames "
                f"need about {args.max_frames * 500:,}.\n"
                f"Fewer frames is the cheaper fix and usually costs little detail.",
                file=sys.stderr,
            )
        return 2
    t_stage1 = time.perf_counter() - t0
    print(f"[stage 1] done in {t_stage1:.1f}s", file=sys.stderr, flush=True)

    if args.save_description:
        desc_path = args.video.with_suffix(".description.txt")
        desc_path.write_text(description)
        print(f"[info] saved description to {desc_path}", file=sys.stderr)

    # 3. Stage 2 - generate the script
    print(
        f"[stage 2] writing {args.style} voiceover ({length}s target)...",
        file=sys.stderr,
        flush=True,
    )
    t1 = time.perf_counter()
    try:
        script = generate_script.generate_script(
            description=description,
            video_length=length,
            style=args.style,
            model=args.text_model,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"stage 2 failed: {e}", file=sys.stderr)
        return 3
    t_stage2 = time.perf_counter() - t1
    print(f"[stage 2] done in {t_stage2:.1f}s", file=sys.stderr, flush=True)

    if args.save_script:
        script_path = args.video.with_suffix(".script.txt")
        script_path.write_text(script)
        print(f"[info] saved script to {script_path}", file=sys.stderr)

    # 3b. Stage 2b - music caption (optional)
    #
    # This runs here, not next to stage 4, because it uses the same Ollama model
    # stage 2 just loaded. Doing it now means one load instead of two, and lets
    # us release Ollama before ACE-Step needs the memory.
    music_spec = None
    if args.score:
        music_style = args.music_style or args.style
        print(
            f"[stage 2b] writing {music_style} quartet caption...",
            file=sys.stderr,
            flush=True,
        )
        t_2b = time.perf_counter()
        # build_music_spec never raises on LLM trouble - it falls back to the
        # bare preset, because a generic cue beats a dead pipeline.
        music_spec = generate_music.build_music_spec(
            description=description,
            video_length=length,
            style=music_style,
            model=args.text_model,
            temperature=0.7,
            use_llm=not args.no_music_llm,
            release_model=True,
        )
        if args.music_bpm is not None:
            print(
                f"[info] overriding {music_spec.bpm} BPM -> {args.music_bpm} BPM",
                file=sys.stderr,
            )
            music_spec.bpm = args.music_bpm
        print(
            f"[stage 2b] done in {time.perf_counter() - t_2b:.1f}s "
            f"({music_spec.bpm} BPM, {music_spec.key_scale})",
            file=sys.stderr,
        )
        print(f"[info] caption: {music_spec.caption}", file=sys.stderr)

        # Stage 1's vision model may still be resident; ACE-Step wants the room.
        generate_music.release_ollama_model(args.vision_model)

    # 4. Stage 3 - TTS (optional)
    audio_path = None
    if args.narrate:
        audio_path = (
            args.audio_out
            if args.audio_out is not None
            else args.video.with_suffix(".voiceover.wav")
        )
        if args.engine == "kokoro":
            print(
                f"[stage 3] synthesizing via Kokoro voice={args.voice} "
                f"speed={args.speed}...",
                file=sys.stderr,
                flush=True,
            )
        else:
            ref_tag = (
                f"cloning={args.reference_audio.name}"
                if args.reference_audio
                else "voice=default"
            )
            print(
                f"[stage 3] synthesizing via Chatterbox {ref_tag} "
                f"exaggeration={args.exaggeration} cfg={args.cfg_weight}...",
                file=sys.stderr,
                flush=True,
            )
        t2 = time.perf_counter()
        try:
            generate_voice.generate_voice(
                text=script,
                output_path=audio_path,
                engine=args.engine,
                voice=args.voice,
                speed=args.speed,
                reference_audio=args.reference_audio,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                device=args.device,
                max_chars_per_chunk=args.max_chars,
                max_pause=args.max_pause,
                target_lufs=args.target_lufs or None,
            )
        except ImportError as e:
            print(
                f"stage 3 failed: missing dep {e}. Run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 4
        except FileNotFoundError as e:
            print(
                f"stage 3 failed: {e}\nDid you install espeak-ng? "
                "brew install espeak-ng",
                file=sys.stderr,
            )
            return 4
        except Exception as e:
            print(f"stage 3 failed: {e}", file=sys.stderr)
            return 4
        t_stage3 = time.perf_counter() - t2
        print(
            f"[stage 3] wrote {audio_path} in {t_stage3:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    # 5. Stage 4 - music (optional)
    #
    # A failure here must not throw away the voiceover we already paid for, so
    # we record the exit code and still print the script below.
    exit_code = 0
    if args.score and music_spec is not None:
        music_path = (
            args.music_out
            if args.music_out is not None
            else args.video.with_suffix(".music.wav")
        )

        # With multiple varied takes, generate_music writes one spec per take
        # next to its audio, since each take's caption differs.
        if args.save_music_prompt and args.music_takes <= 1:
            spec_path = music_path.with_suffix(".json")
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(music_spec.to_json())
            print(f"[info] saved music spec to {spec_path}", file=sys.stderr)

        lufs = generate_music.resolve_lufs_target(args.music_lufs, audio_path)

        print(
            f"[stage 4] generating {args.music_takes} take(s) via ACE-Step...",
            file=sys.stderr,
            flush=True,
        )
        t3 = time.perf_counter()
        try:
            written = generate_music.generate_music(
                spec=music_spec,
                output_path=music_path,
                video_length=length,
                server=args.music_server,
                takes=args.music_takes,
                seed=args.music_seed,
                thinking=args.music_thinking,
                lufs_target=lufs,
                timeout=args.music_timeout,
                vary=not args.no_music_vary,
                save_specs=args.save_music_prompt and args.music_takes > 1,
                bit_depth=args.music_bit_depth,
            )
        except generate_music.AceStepError as e:
            print(f"stage 4 failed: {e}", file=sys.stderr)
            exit_code = 5
        else:
            t_stage4 = time.perf_counter() - t3
            for p in written:
                print(f"[stage 4] wrote {p}", file=sys.stderr)
            print(f"[stage 4] done in {t_stage4:.1f}s", file=sys.stderr, flush=True)

    total = time.perf_counter() - t0
    print(f"[pipeline] total {total:.1f}s\n", file=sys.stderr, flush=True)

    # Final output on stdout: always the script text.
    print(script)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
