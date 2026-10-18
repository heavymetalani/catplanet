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
    python pipeline.py data/scratch/zoomies.mov
    python pipeline.py data/scratch/zoomies.mov --style noir --fps 3 --max-frames 20
    python pipeline.py data/scratch/zoomies.mov --narrate
    python pipeline.py data/scratch/zoomies.mov --narrate --engine kokoro --voice bm_lewis --speed 0.95
    python pipeline.py data/scratch/zoomies.mov --narrate --engine chatterbox
    python pipeline.py data/scratch/zoomies.mov --narrate --engine chatterbox \\
        --reference-audio data/voices/narrator.wav --exaggeration 0.4
    python pipeline.py data/scratch/zoomies.mov --save-description --save-script --narrate

    # With a string quartet score (needs the ACE-Step server running)
    python pipeline.py data/scratch/zoomies.mov --narrate --score
    python pipeline.py data/scratch/zoomies.mov --narrate --score --music-takes 3

    # Retry-friendly: save every stage, skip what already finished
    python pipeline.py data/scratch/zoomies.mov --narrate --score --resume
    python pipeline.py data/scratch/zoomies.mov --narrate --score --fresh

Resume mode (--resume, which batch.py always uses) saves each stage's result as
soon as it finishes and records it in <name>.status.json. Run the same command
again after a failure and only the missing stages run. Changed settings redo
just the stages they affect; replaced files are kept in attempt-N/. --fresh
redoes every stage. See catplanet/progress.py and docs/batch.md.

Duration is detected automatically via ffprobe (falls back to --length if that fails).

Stage 4 needs the ACE-Step API server up in a separate terminal:
    "MUSIC GEN/start_api_server.command"

Note on ordering: stage 2b runs immediately after stage 2 because both use the
same Ollama model, so it loads once; Ollama is then released before stage 4 so
ACE-Step gets the memory. See docs/MUSIC_PLAN.md section 7.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import describe_video, generate_music, generate_script, generate_voice, progress


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
        prog="pipeline.py",
        description="Cat video -> funny voiceover, in one shot.",
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
    ap.add_argument(
        "--no-title",
        action="store_true",
        help="Leave out the spoken title at the start of the script.",
    )

    # Where outputs go, and resume
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Folder for every output (default: the video's own folder).",
    )
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Save every stage's result plus a <name>.status.json, and skip "
            "stages that already finished with the same settings. Rerun the "
            "same command after a failure to pick up where it stopped."
        ),
    )
    resume.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Like --resume, but redo every stage. Files it replaces are moved "
            "into attempt-N/ rather than overwritten."
        ),
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


def _music_files(music_path: Path) -> list[Path]:
    """Every file stage 4 can write for this clip, in single- or multi-take form."""
    files = [music_path, music_path.with_suffix(".json")]
    for i in range(1, 9):
        take = music_path.with_suffix(f".{i}{music_path.suffix}")
        files += [take, take.with_suffix(".json")]
    return files


def main() -> int:
    args = build_parser().parse_args()

    if not args.video.exists():
        print(f"error: {args.video} does not exist", file=sys.stderr)
        return 1

    # ---- where everything goes -------------------------------------------
    # Resume mode (--resume / --fresh) always saves the description, script
    # and music plan, because a retry needs them to skip the earlier stages.
    tracked = args.resume or args.fresh
    out_dir = args.out_dir if args.out_dir is not None else args.video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem

    def out(kind: str) -> Path:
        return out_dir / f"{stem}.{kind}"

    desc_path = out("description.txt")
    script_path = out("script.txt")
    plan_path = out("music_plan.json")
    voice_target = args.audio_out if args.audio_out is not None else out("voiceover.wav")
    music_path = args.music_out if args.music_out is not None else out("music.wav")

    # Clear away half-written files left by an interrupted run.
    progress.remove_partials(out_dir, stem)
    for target in (voice_target, *_music_files(music_path)):
        progress.partial_path(target).unlink(missing_ok=True)

    clip = progress.ClipProgress(out_dir, stem, enabled=tracked, fresh=args.fresh)

    # Fail fast: check the music server now, not after three minutes of stages 1-3.
    if args.score:
        try:
            generate_music.check_server(args.music_server)
        except generate_music.AceStepError as e:
            print(f"error: {e}", file=sys.stderr)
            return 5

    # ---- video length -----------------------------------------------------
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

    # ---- settings fingerprints ---------------------------------------------
    # Each one includes the stage before it, so a change flows downstream and
    # redoes exactly the stages it affects. See catplanet/progress.py.
    fp = progress.fingerprint
    music_style = args.music_style or args.style
    fp_describe = fp(
        "describe",
        progress.file_fingerprint(args.video) if tracked else None,
        args.vision_model, args.fps, args.max_frames, args.num_ctx,
        describe_video.DEFAULT_PROMPT,           # includes your cats
    )
    fp_script = fp("script", fp_describe, args.style, args.text_model,
                   args.temperature, length,
                   generate_script.prompt_signature(args.style, not args.no_title))
    fp_plan = fp("music_plan", fp_describe, music_style, args.text_model,
                 not args.no_music_llm, length, args.music_bpm)
    if args.engine == "kokoro":
        voice_settings = {"engine": "kokoro", "voice": args.voice, "speed": args.speed}
    else:
        voice_settings = {
            "engine": "chatterbox",
            "reference": (progress.file_fingerprint(args.reference_audio)
                          if tracked and args.reference_audio else None),
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
            "max_chars": args.max_chars,
        }
    fp_voice = fp("voice", fp_script, voice_settings, args.max_pause, args.target_lufs)

    clip.start_run(
        f"style={args.style} narrate={args.narrate} engine={args.engine} "
        f"score={args.score} takes={args.music_takes}"
    )

    def fail(code: int, message: str) -> int:
        print(message, file=sys.stderr)
        clip.finish_run(False, message)
        return code

    t0 = time.perf_counter()
    try:
        # ---- stage 1: describe the video ------------------------------------
        clip.step("stage 1 (describe the video)")
        if clip.reusable("describe", fp_describe, desc_path):
            description = desc_path.read_text()
            clip.mark("describe", "reused")
            print(f"[stage 1] reusing saved description ({desc_path.name})",
                  file=sys.stderr, flush=True)
        else:
            clip.keep_old(desc_path)
            print("[stage 1] describing video...", file=sys.stderr, flush=True)
            t1 = time.perf_counter()
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
                return fail(2, f"stage 1 failed: {e}")
            print(f"[stage 1] done in {time.perf_counter() - t1:.1f}s",
                  file=sys.stderr, flush=True)
            if tracked or args.save_description:
                progress.write_text(desc_path, description)
                print(f"[info] saved description to {desc_path}", file=sys.stderr)
            clip.record("describe", fp_describe, desc_path)

        # ---- stage 2: write the script --------------------------------------
        clip.step("stage 2 (write the script)")
        if clip.reusable("script", fp_script, script_path):
            script = script_path.read_text()
            clip.mark("script", "reused")
            print(f"[stage 2] reusing saved script ({script_path.name})",
                  file=sys.stderr, flush=True)
        else:
            clip.keep_old(script_path)
            print(f"[stage 2] writing {args.style} voiceover ({length}s target)...",
                  file=sys.stderr, flush=True)
            t2 = time.perf_counter()
            try:
                script = generate_script.generate_script(
                    description=description,
                    video_length=length,
                    style=args.style,
                    model=args.text_model,
                    temperature=args.temperature,
                    title=not args.no_title,
                )
            except Exception as e:
                return fail(3, f"stage 2 failed: {e}")
            print(f"[stage 2] done in {time.perf_counter() - t2:.1f}s",
                  file=sys.stderr, flush=True)
            if tracked or args.save_script:
                progress.write_text(script_path, script)
                print(f"[info] saved script to {script_path}", file=sys.stderr)
            clip.record("script", fp_script, script_path)

        # ---- stage 2b: music plan (optional) ---------------------------------
        # This runs here, not next to stage 4, because it uses the same Ollama
        # model stage 2 just loaded. Doing it now means one load instead of
        # two, and lets us release Ollama before ACE-Step needs the memory.
        music_spec = None
        if args.score:
            clip.step("stage 2b (plan the music)")
            if clip.reusable("music_plan", fp_plan, plan_path):
                music_spec = generate_music.MusicSpec.from_json(plan_path.read_text())
                clip.mark("music_plan", "reused")
                print(f"[stage 2b] reusing saved music plan ({plan_path.name}): "
                      f"{music_spec.bpm} BPM, {music_spec.key_scale}",
                      file=sys.stderr, flush=True)
            else:
                clip.keep_old(plan_path)
                print(f"[stage 2b] writing {music_style} quartet caption...",
                      file=sys.stderr, flush=True)
                t_2b = time.perf_counter()
                # build_music_spec never raises on LLM trouble - it falls back
                # to the bare preset, because a generic cue beats a dead pipeline.
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
                    print(f"[info] overriding {music_spec.bpm} BPM -> "
                          f"{args.music_bpm} BPM", file=sys.stderr)
                    music_spec.bpm = args.music_bpm
                print(
                    f"[stage 2b] done in {time.perf_counter() - t_2b:.1f}s "
                    f"({music_spec.bpm} BPM, {music_spec.key_scale})",
                    file=sys.stderr,
                )
                if tracked:
                    progress.write_text(plan_path, music_spec.to_json())
                clip.record("music_plan", fp_plan, plan_path)
            print(f"[info] caption: {music_spec.caption}", file=sys.stderr)

            # Stage 1's vision model may still be resident; ACE-Step wants the room.
            generate_music.release_ollama_model(args.vision_model)

        # ---- stage 3: voiceover (optional) ----------------------------------
        audio_path = None
        if args.narrate:
            audio_path = voice_target
            clip.step("stage 3 (voiceover)")
            if clip.reusable("voice", fp_voice, audio_path):
                clip.mark("voice", "reused")
                print(f"[stage 3] reusing saved voiceover ({audio_path.name})",
                      file=sys.stderr, flush=True)
            else:
                clip.keep_old(audio_path)
                if args.engine == "kokoro":
                    print(f"[stage 3] synthesizing via Kokoro voice={args.voice} "
                          f"speed={args.speed}...", file=sys.stderr, flush=True)
                else:
                    ref_tag = (f"cloning={args.reference_audio.name}"
                               if args.reference_audio else "voice=default")
                    print(f"[stage 3] synthesizing via Chatterbox {ref_tag} "
                          f"exaggeration={args.exaggeration} cfg={args.cfg_weight}...",
                          file=sys.stderr, flush=True)
                t3 = time.perf_counter()
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
                    return fail(4, f"stage 3 failed: missing dep {e}. "
                                   f"Run: pip install -r requirements.txt")
                except FileNotFoundError as e:
                    return fail(4, f"stage 3 failed: {e}\nDid you install "
                                   f"espeak-ng? brew install espeak-ng")
                except Exception as e:
                    return fail(4, f"stage 3 failed: {e}")
                print(f"[stage 3] wrote {audio_path} in "
                      f"{time.perf_counter() - t3:.1f}s", file=sys.stderr, flush=True)
                clip.record("voice", fp_voice, audio_path)

        # ---- stage 4: music (optional) --------------------------------------
        # A failure here must not throw away the voiceover we already paid
        # for, so we record the exit code and still print the script below.
        exit_code = 0
        error = None
        if args.score and music_spec is not None:
            takes = max(1, args.music_takes)
            vary = not args.no_music_vary

            # With multiple varied takes, generate_music writes one spec per
            # take next to its audio, since each take's caption differs.
            if args.save_music_prompt and takes <= 1:
                spec_path = music_path.with_suffix(".json")
                progress.write_text(spec_path, music_spec.to_json())
                print(f"[info] saved music spec to {spec_path}", file=sys.stderr)

            lufs = generate_music.resolve_lufs_target(args.music_lufs, audio_path)
            fp_music = fp(
                "music", fp_plan,
                round(lufs * 2) / 2,                 # nearest 0.5 LU
                args.music_seed, args.music_thinking, args.music_bit_depth,
                "per-take" if vary else f"batched x{takes}",
                "single" if takes == 1 else "numbered",
            )

            # Which takes already exist from an earlier run with these settings?
            reuse = {i: p for i, p in clip.reusable_takes(fp_music).items()
                     if i <= takes}
            if not vary and len(reuse) < takes:
                reuse = {}       # one batched request: all or nothing
            keep = set(reuse.values()) | {p.with_suffix(".json") for p in reuse.values()}
            if args.save_music_prompt and takes <= 1:
                keep.add(music_path.with_suffix(".json"))
            clip.keep_old(*[f for f in _music_files(music_path) if f not in keep])

            missing = [i for i in range(1, takes + 1) if i not in reuse]
            done_takes = {"done": len(reuse), "of": takes}
            clip.mark("music", dict(done_takes, reused=not missing))
            if not missing:
                print(f"[stage 4] reusing all {takes} saved take(s)",
                      file=sys.stderr, flush=True)
            else:
                clip.step(f"stage 4 (music take {missing[0]} of {takes})")
                if reuse:
                    print(f"[stage 4] {len(reuse)} take(s) already done; "
                          f"generating {len(missing)} more via ACE-Step...",
                          file=sys.stderr, flush=True)
                else:
                    print(f"[stage 4] generating {takes} take(s) via ACE-Step...",
                          file=sys.stderr, flush=True)

                def on_take(i: int, path: Path) -> None:
                    clip.record_take(fp_music, i, path)
                    done_takes["done"] += 1
                    clip.mark("music", dict(done_takes))
                    later = [m for m in missing if m > i]
                    if later:
                        clip.step(f"stage 4 (music take {later[0]} of {takes})")

                t4 = time.perf_counter()
                try:
                    written = generate_music.generate_music(
                        spec=music_spec,
                        output_path=music_path,
                        video_length=length,
                        server=args.music_server,
                        takes=takes,
                        seed=args.music_seed,
                        thinking=args.music_thinking,
                        lufs_target=lufs,
                        timeout=args.music_timeout,
                        vary=vary,
                        save_specs=args.save_music_prompt and takes > 1,
                        bit_depth=args.music_bit_depth,
                        reuse=reuse,
                        on_take=on_take,
                    )
                except generate_music.AceStepError as e:
                    error = f"stage 4 failed: {e}"
                    print(error, file=sys.stderr)
                    exit_code = 5
                else:
                    for p in written:
                        print(f"[stage 4] wrote {p}", file=sys.stderr)
                    print(f"[stage 4] done in {time.perf_counter() - t4:.1f}s",
                          file=sys.stderr, flush=True)

    except KeyboardInterrupt:
        clip.finish_run(False, "interrupted (Ctrl-C)")
        raise

    clip.finish_run(exit_code == 0, error)
    print(f"[pipeline] total {time.perf_counter() - t0:.1f}s\n",
          file=sys.stderr, flush=True)

    # Final output on stdout: always the script text.
    print(script)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
