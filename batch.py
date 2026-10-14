#!/usr/bin/env python3
"""
batch.py

Process every video in video_samples/ through the pipeline, one at a time,
numbering them 1, 2, 3, ... across runs.

Layout
------
    video_samples/              <- drop new videos here (the queue)
    video_samples/processed/    <- finished videos, renamed 7.mov, plus their
                                   7.script.txt and 7.description.txt
    voice_samples/              <- 7.wav
    music_samples/              <- 7.1.wav, 7.2.wav, 7.3.wav
    batch_state.json            <- the version counter and a log of every run

Videos move to processed/ only after they succeed, so video_samples/ always
shows exactly what is still outstanding. A failed video keeps its number and
stays in the queue: a file already named as a bare number is recognized as
versioned and reuses it rather than burning a new one on every retry.

Usage
-----
    python batch.py                       # the default profile below
    python batch.py --dry-run             # show the plan, touch nothing
    python batch.py --music-takes 1       # unknown flags pass to pipeline.py
    python batch.py --style noir --limit 2

Any argument batch.py does not recognize is forwarded to pipeline.py and
appended after the defaults, so it wins. Requires the ACE-Step server when
scoring:  "MUSIC GEN/start_api_server.command"
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import local_settings

REPO = Path(__file__).resolve().parent


def _last_value(args: list[str], flag: str) -> str | None:
    """The value of the LAST occurrence of `flag`, matching argparse's rule."""
    value = None
    for i, a in enumerate(args[:-1]):
        if a == flag:
            value = args[i + 1]
    return value

DEFAULT_VIDEO_DIR = REPO / "video_samples"
DEFAULT_VOICE_DIR = REPO / "voice_samples"
DEFAULT_MUSIC_DIR = REPO / "music_samples"
DEFAULT_STATE = REPO / "batch_state.json"
PROCESSED_SUBDIR = "processed"

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}

# The command you run per clip. Only the video path varies; everything here is
# fixed. Override any of it from the command line -- unknown flags are
# forwarded and appended after these, and argparse takes the last occurrence.
#
# The voice to clone is deliberately NOT here: a recording of a real person is
# personal, so its path lives in local_settings.json ("reference_audio"),
# which is gitignored. Without it, Chatterbox uses its built-in default voice.
DEFAULT_PIPELINE_ARGS = [
    "--narrate",
    "--engine", "chatterbox",
    "--exaggeration", "0.3",
    "--cfg-weight", "0.35",
    "--style", "attenborough",
    "--save-script",
    "--score",
    "--music-takes", "3",
    "--num-ctx", "32768",
]


# =========================================================
# State
# =========================================================
def load_state(path: Path) -> dict:
    if not path.exists():
        return {"last_version": 0, "entries": []}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # Never silently restart numbering -- that would overwrite earlier work.
        raise SystemExit(
            f"error: could not read {path}: {e}\n"
            f"Fix or delete it, but note that deleting restarts numbering at 1 "
            f"and may overwrite existing outputs."
        )
    data.setdefault("last_version", 0)
    data.setdefault("entries", [])
    return data


def save_state(path: Path, state: dict) -> None:
    """Write atomically; a half-written state file would corrupt numbering."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


# =========================================================
# Discovery and versioning
# =========================================================
def find_videos(video_dir: Path) -> list[Path]:
    """Videos waiting at the top level of the queue. processed/ is skipped."""
    if not video_dir.is_dir():
        return []
    return sorted(
        (p for p in video_dir.iterdir()
         if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def existing_version(path: Path) -> int | None:
    """A stem that is a bare number means this file was versioned already."""
    return int(path.stem) if path.stem.isdigit() else None


def version_is_taken(
    version: int, video_dir: Path, voice_dir: Path, music_dir: Path
) -> bool:
    """Does this number already have finished work behind it?

    Reusing a number is right for retrying a failed clip, and catastrophic for
    a video that merely happens to be named "1.mov" -- that would overwrite the
    real 1's outputs. Finished work is the difference between the two.
    """
    return any(
        [
            any((video_dir / PROCESSED_SUBDIR).glob(f"{version}.*")),
            (voice_dir / f"{version}.wav").exists(),
            any(music_dir.glob(f"{version}.wav")),
            any(music_dir.glob(f"{version}.[0-9]*.wav")),
        ]
    )


def claim_version(
    state: Path,
    data: dict,
    video: Path,
    dirs: tuple[Path, Path, Path],
    dry_run: bool = False,
) -> int:
    """Reserve the next number, persisting it before any work starts.

    Persisting first matters: if the machine dies mid-clip, the number is spent
    rather than handed out twice. A dry run previews numbers without spending
    them, so it can be run freely.
    """
    existing = existing_version(video)
    if existing is not None:
        if version_is_taken(existing, *dirs):
            # Not a retry -- a different video that happens to be named "1.mov".
            # Reusing the number here would overwrite the real 1's outputs.
            print(
                f"    [warn] {video.name} looks versioned, but {existing} already "
                f"has finished outputs. Treating it as new to avoid overwriting.",
                file=sys.stderr,
            )
        else:
            if existing > data["last_version"]:
                data["last_version"] = existing
                if not dry_run:
                    save_state(state, data)
            return existing

    data["last_version"] += 1
    if not dry_run:
        save_state(state, data)
    return data["last_version"]


# =========================================================
# Running one video
# =========================================================
def run_one(
    video: Path,
    version: int,
    video_dir: Path,
    voice_dir: Path,
    music_dir: Path,
    pipeline_args: list[str],
    dry_run: bool,
) -> tuple[bool, Path, list[str]]:
    """Version, run, and file away one video. Returns (ok, final_path, outputs)."""
    numbered = video_dir / f"{version}{video.suffix.lower()}"
    voice_out = voice_dir / f"{version}.wav"
    music_out = music_dir / f"{version}.wav"

    cmd = [sys.executable, "pipeline.py", str(numbered)] + pipeline_args + [
        "--audio-out", str(voice_out),
        "--music-out", str(music_out),
    ]

    if dry_run:
        print(f"    would rename -> {numbered.name}")
        print(f"    would run    : {' '.join(cmd[1:])}")
        return True, numbered, []

    # Rename in place first. If the pipeline then fails, the file stays in the
    # queue already numbered, so the retry reuses this number.
    if video != numbered:
        if numbered.exists():
            print(f"    [warn] {numbered.name} already exists; leaving {video.name} alone",
                  file=sys.stderr)
            return False, video, []
        video.rename(numbered)

    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"    [fail] pipeline exited {result.returncode}; "
              f"{numbered.name} stays in the queue for a retry", file=sys.stderr)
        return False, numbered, []

    # Success: move the video and its text outputs out of the queue.
    processed = video_dir / PROCESSED_SUBDIR
    processed.mkdir(parents=True, exist_ok=True)
    for extra in (numbered, numbered.with_suffix(".script.txt"),
                  numbered.with_suffix(".description.txt")):
        if extra.exists():
            shutil.move(str(extra), str(processed / extra.name))

    # Label by directory -- "7.wav" means different things in each.
    outputs = [f"{voice_dir.name}/{p.name}" for p in sorted(voice_dir.glob(f"{version}.wav"))]
    outputs += [f"{music_dir.name}/{p.name}"
                for p in sorted(music_dir.glob(f"{version}.wav"))]
    outputs += [f"{music_dir.name}/{p.name}"
                for p in sorted(music_dir.glob(f"{version}.[0-9]*.wav"))]
    return True, processed / numbered.name, outputs


# =========================================================
# CLI
# =========================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch-process every video in video_samples/ through the pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
                    help=f"Queue directory (default: {DEFAULT_VIDEO_DIR.name}/).")
    ap.add_argument("--voice-dir", type=Path, default=DEFAULT_VOICE_DIR,
                    help=f"Voiceover output directory (default: {DEFAULT_VOICE_DIR.name}/).")
    ap.add_argument("--music-dir", type=Path, default=DEFAULT_MUSIC_DIR,
                    help=f"Music output directory (default: {DEFAULT_MUSIC_DIR.name}/).")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE,
                    help=f"Version counter file (default: {DEFAULT_STATE.name}).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N videos this run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without renaming or generating.")
    args, passthrough = ap.parse_known_args()

    # batch.py owns these -- it routes outputs into the numbered directories.
    # Accepting them from the command line would silently do nothing, since
    # batch's own copies are appended last and win.
    for owned in ("--audio-out", "--music-out"):
        if owned in passthrough:
            print(
                f"error: {owned} is managed by batch.py (outputs go to "
                f"{args.voice_dir.name}/ and {args.music_dir.name}/ by version "
                f"number).\nUse --voice-dir / --music-dir to move them, or run "
                f"pipeline.py directly for a one-off.",
                file=sys.stderr,
            )
            return 1

    # Defaults, then your private voice from local_settings.json, then anything
    # typed on the command line -- later wins, so a one-off --reference-audio
    # still overrides the saved one.
    personal: list[str] = []
    saved_ref = local_settings.get("reference_audio")
    if saved_ref and "--reference-audio" not in passthrough:
        personal = ["--reference-audio", str(saved_ref)]
    pipeline_args = DEFAULT_PIPELINE_ARGS + personal + passthrough
    scoring = "--score" in pipeline_args

    # Validate the whole arg list against pipeline.py's real parser now, once.
    # Without this a typo like --musictakes is only discovered per clip, after
    # every video has been renamed and failed in turn.
    try:
        import pipeline
        parser = pipeline.build_parser()
        try:
            parser.parse_args(["__validate__.mov"] + pipeline_args)
        except SystemExit:
            print("\nerror: the arguments above are not accepted by pipeline.py; "
                  "nothing was processed.", file=sys.stderr)
            return 1
    except ImportError as e:
        print(f"[warn] could not validate pipeline args ({e})", file=sys.stderr)

    for d in (args.video_dir, args.voice_dir, args.music_dir):
        d.mkdir(parents=True, exist_ok=True)

    videos = find_videos(args.video_dir)
    if not videos:
        print(f"Nothing to do: no videos in {args.video_dir.name}/")
        print(f"Drop some in ({', '.join(sorted(VIDEO_EXTS))}) and run again.")
        return 0
    if args.limit is not None:
        videos = videos[: args.limit]

    # Fail fast before renaming anything: a missing music server should not
    # leave half the queue versioned and unprocessed.
    if scoring and not args.dry_run:
        try:
            import generate_music
            server = _last_value(pipeline_args, "--music-server") or "http://127.0.0.1:8001"
            generate_music.check_server(server)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 5

    ref = _last_value(pipeline_args, "--reference-audio")
    if ref is not None:
        if not (REPO / ref).exists():
            print(f"error: reference audio not found: {ref}", file=sys.stderr)
            return 1
    elif _last_value(pipeline_args, "--engine") == "chatterbox":
        print(
            "[info] no reference_audio in local_settings.json; narrating with "
            "Chatterbox's default voice.",
            file=sys.stderr,
        )

    data = load_state(args.state)
    print(f"queue      : {args.video_dir.name}/  ({len(videos)} video"
          f"{'s' if len(videos) != 1 else ''})")
    print(f"numbering  : continuing from {data['last_version']}")
    print(f"pipeline   : {' '.join(pipeline_args)}")
    if args.dry_run:
        print("mode       : DRY RUN, nothing will be written")
    print()

    results: list[dict] = []
    t_start = time.perf_counter()

    for i, video in enumerate(videos, start=1):
        version = claim_version(
            args.state, data, video,
            (args.video_dir, args.voice_dir, args.music_dir),
            dry_run=args.dry_run,
        )
        print(f"[{i}/{len(videos)}] {video.name}  ->  version {version}")
        t0 = time.perf_counter()
        try:
            ok, final, outputs = run_one(
                video, version, args.video_dir, args.voice_dir, args.music_dir,
                pipeline_args, args.dry_run,
            )
        except KeyboardInterrupt:
            print("\n[interrupted] stopping. Unprocessed videos stay in the queue.",
                  file=sys.stderr)
            break
        except Exception as e:  # one bad clip must not end the batch
            print(f"    [fail] {e}", file=sys.stderr)
            ok, final, outputs = False, video, []

        elapsed = time.perf_counter() - t0
        results.append({"version": version, "original": video.name,
                        "ok": ok, "outputs": outputs, "seconds": round(elapsed, 1)})
        if ok and not args.dry_run:
            print(f"    done in {elapsed:.0f}s -> {', '.join(outputs) or 'no outputs'}")
            data["entries"].append({
                "version": version,
                "original": video.name,
                "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "outputs": outputs,
            })
            save_state(args.state, data)
        print()

    # ---- summary ----
    done = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    total = time.perf_counter() - t_start

    print("=" * 60)
    print(f"{len(done)} succeeded, {len(failed)} failed, {total/60:.1f} min total")
    for r in results:
        mark = "ok  " if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['version']:>4}  {r['original']}  ({r['seconds']:.0f}s)")
    if failed:
        print(f"\n{len(failed)} video(s) stayed in {args.video_dir.name}/, "
              f"already numbered. Fix the cause and run again to retry them "
              f"with the same numbers.")
    if done and not args.dry_run:
        print(f"\nvoiceovers : {args.voice_dir.name}/")
        print(f"music      : {args.music_dir.name}/")
        print(f"videos     : {args.video_dir.name}/{PROCESSED_SUBDIR}/ (with scripts)")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
