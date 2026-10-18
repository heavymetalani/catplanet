#!/usr/bin/env python3
"""
batch.py

Process every video in data/inbox/ through the pipeline, one at a time,
numbering them 1, 2, 3, ... across runs. Each finished clip gets its own
folder, so everything you need for one edit sits together.

Layout
------
    data/inbox/                 <- drop new videos here (the queue)
    data/outputs/7/             <- one folder per clip:
        7.mov                        the video, moved here once the run succeeds
        7.description.txt            what the vision model saw
        7.script.txt                 the narration script
        7.script_drafts.txt          every draft, and why the script won
        7.music_plan.json            the music caption, tempo and key
        7.voiceover.wav              the narrated voiceover
        7.music.1.wav ...            the string quartet takes
        7.status.json                which stages finished, with which settings
        attempt-1/                   older files replaced by a later run, if any
    data/batch_state.json       <- the version counter and a log of every run

A video moves out of the inbox only after it succeeds, so data/inbox/ always
shows exactly what is still outstanding. A failed video keeps its number and
stays in the inbox: a file already named as a bare number is recognized as
versioned and reuses it rather than burning a new one on every retry.

Retries pick up where the last run stopped. Each stage's result is saved into
the clip's folder as soon as it finishes, so a clip that failed at music take
3 only redoes take 3. If you change settings between runs, only the stages
those settings affect are redone, and the files they replace are kept in
attempt-N/. Use --fresh to redo every stage. See catplanet/progress.py.

Usage
-----
    python batch.py                       # the default profile below
    python batch.py --dry-run             # show the plan, touch nothing
    python batch.py --music-takes 1       # unknown flags pass to pipeline.py
    python batch.py --style noir --limit 2
    python batch.py --fresh               # redo every stage of the queued clips

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

from . import local_settings, paths, progress

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}

# The command you run per clip. Only the video path varies; everything here is
# fixed. Override any of it from the command line -- unknown flags are
# forwarded and appended after these, and argparse takes the last occurrence.
#
# The voice to clone is deliberately NOT here: a recording of a real person is
# personal, so its path lives in config/local_settings.json ("reference_audio"),
# which is gitignored. Without it, Chatterbox uses its built-in default voice.
DEFAULT_PIPELINE_ARGS = [
    "--narrate",
    "--engine", "chatterbox",
    "--exaggeration", "0.3",
    "--cfg-weight", "0.35",
    "--style", "attenborough",
    "--score",
    "--music-takes", "3",
    "--num-ctx", "32768",
]


def _last_value(args: list[str], flag: str) -> str | None:
    """The value of the LAST occurrence of `flag`, matching argparse's rule."""
    value = None
    for i, a in enumerate(args[:-1]):
        if a == flag:
            value = args[i + 1]
    return value


def _label(path: Path) -> str:
    """A short, readable name for a path: relative to the project root."""
    try:
        return str(path.resolve().relative_to(paths.REPO))
    except ValueError:
        return str(path)


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
def find_videos(inbox: Path) -> list[Path]:
    """Videos waiting at the top level of the inbox."""
    if not inbox.is_dir():
        return []
    return sorted(
        (p for p in inbox.iterdir()
         if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def existing_version(path: Path) -> int | None:
    """A stem that is a bare number means this file was versioned already."""
    return int(path.stem) if path.stem.isdigit() else None


def clip_dir(outputs: Path, version: int) -> Path:
    """The folder that holds everything for one clip."""
    return outputs / str(version)


def version_is_taken(version: int, outputs: Path) -> bool:
    """Has this number already been finished?

    Reusing a number is right for retrying a failed clip, and catastrophic for
    a video that merely happens to be named "1.mov" -- that would overwrite the
    real 1's outputs. The video is moved into its clip folder only when the
    whole run succeeds, so its presence there is what "finished" means. A
    failed attempt can leave a partial voiceover behind; the retry overwrites
    it, which is what you want.
    """
    folder = clip_dir(outputs, version)
    return any(
        p.suffix.lower() in VIDEO_EXTS for p in folder.glob(f"{version}.*")
    )


def claim_version(
    state: Path,
    data: dict,
    video: Path,
    outputs: Path,
    dry_run: bool = False,
) -> int:
    """Reserve the next number, persisting it before any work starts.

    Persisting first matters: if the machine dies mid-clip, the number is spent
    rather than handed out twice. A dry run previews numbers without spending
    them, so it can be run freely.
    """
    existing = existing_version(video)
    if existing is not None:
        if version_is_taken(existing, outputs):
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
    inbox: Path,
    outputs: Path,
    pipeline_args: list[str],
    dry_run: bool,
    fresh: bool = False,
) -> tuple[bool, Path, list[str], str | None]:
    """Version, run, and file away one video.

    Returns (ok, final_path, outputs, why_it_stopped).
    """
    numbered = inbox / f"{version}{video.suffix.lower()}"
    folder = clip_dir(outputs, version)

    # Every output goes straight into the clip's folder, and resume mode skips
    # whatever an earlier attempt already finished there.
    cmd = [sys.executable, "-m", "catplanet.pipeline", str(numbered)] + pipeline_args + [
        "--out-dir", str(folder),
        "--fresh" if fresh else "--resume",
    ]

    if dry_run:
        status = folder / f"{version}.status.json"
        print(f"    would rename -> {numbered.name}")
        print(f"    would file in: {_label(folder)}/")
        if status.exists() and not fresh:
            print(f"    would resume : an earlier attempt "
                  f"{progress.describe_stop(status) or 'finished some stages'}")
        print(f"    would run    : pipeline.py {' '.join(cmd[4:])}")
        return True, numbered, [], None

    # Rename in place first. If the pipeline then fails, the file stays in the
    # inbox already numbered, so the retry reuses this number.
    if video != numbered:
        if numbered.exists():
            print(f"    [warn] {numbered.name} already exists; leaving {video.name} alone",
                  file=sys.stderr)
            return False, video, [], "a file with that number is already in the inbox"
        video.rename(numbered)

    folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=paths.REPO)
    if result.returncode != 0:
        why = (progress.describe_stop(folder / f"{version}.status.json")
               or f"pipeline exited {result.returncode}")
        print(f"    [fail] {why}\n"
              f"    {numbered.name} stays in the inbox; the next run resumes "
              f"from there", file=sys.stderr)
        return False, numbered, [], why

    # Success: move the video (and any text files from older runs) into the
    # clip's folder.
    for extra in (numbered, numbered.with_suffix(".script.txt"),
                  numbered.with_suffix(".description.txt")):
        if extra.exists():
            shutil.move(str(extra), str(folder / extra.name))

    written = [_label(p) for p in sorted(folder.iterdir())
               if p.is_file() and not p.name.startswith(".")]
    return True, folder / numbered.name, written, None


# =========================================================
# CLI
# =========================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        prog="batch.py",
        description="Batch-process every video in data/inbox/ through the pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--inbox", type=Path, default=paths.INBOX_DIR,
                    help="Queue directory (default: data/inbox/).")
    ap.add_argument("--outputs", type=Path, default=paths.OUTPUTS_DIR,
                    help="Where each clip's folder is created (default: data/outputs/).")
    ap.add_argument("--state", type=Path, default=paths.BATCH_STATE,
                    help="Version counter file (default: data/batch_state.json).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N videos this run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without renaming or generating.")
    ap.add_argument("--exact-args", action="store_true",
                    help="Use only the pipeline options given on this command "
                         "line: skip DEFAULT_PIPELINE_ARGS and the saved "
                         "reference_audio. The control panel (app.py) uses this "
                         "so what you see is exactly what runs.")
    ap.add_argument("--fresh", action="store_true",
                    help="Redo every stage instead of resuming. Replaced files "
                         "are kept in each clip's attempt-N/ folder.")
    args, passthrough = ap.parse_known_args()

    # batch.py owns these -- it routes outputs into the numbered clip folders.
    # Accepting them from the command line would silently do nothing, since
    # batch's own copies are appended last and win.
    for owned in ("--audio-out", "--music-out", "--out-dir"):
        if owned in passthrough:
            print(
                f"error: {owned} is managed by batch.py (each clip's outputs go "
                f"to {_label(args.outputs)}/<number>/).\nUse --outputs to move "
                f"them, or run pipeline.py directly for a one-off.",
                file=sys.stderr,
            )
            return 1
    if "--resume" in passthrough:
        print("error: batch.py always resumes; drop --resume (or use --fresh "
              "to redo every stage).", file=sys.stderr)
        return 1

    # Defaults, then your private voice from config/local_settings.json, then
    # anything typed on the command line -- later wins, so a one-off
    # --reference-audio still overrides the saved one.
    personal: list[str] = []
    saved_ref = local_settings.get("reference_audio")
    if saved_ref and "--reference-audio" not in passthrough and not args.exact_args:
        # Settings-file paths are relative to the project root.
        personal = ["--reference-audio", str(paths.resolve(saved_ref))]
    defaults = [] if args.exact_args else DEFAULT_PIPELINE_ARGS
    pipeline_args = defaults + personal + passthrough
    scoring = "--score" in pipeline_args

    # Validate the whole arg list against pipeline.py's real parser now, once.
    # Without this a typo like --musictakes is only discovered per clip, after
    # every video has been renamed and failed in turn.
    try:
        from . import pipeline
        parser = pipeline.build_parser()
        try:
            parser.parse_args(["__validate__.mov"] + pipeline_args)
        except SystemExit:
            print("\nerror: the arguments above are not accepted by pipeline.py; "
                  "nothing was processed.", file=sys.stderr)
            return 1
    except ImportError as e:
        print(f"[warn] could not validate pipeline args ({e})", file=sys.stderr)

    for d in (args.inbox, args.outputs):
        d.mkdir(parents=True, exist_ok=True)

    videos = find_videos(args.inbox)
    if not videos:
        print(f"Nothing to do: no videos in {_label(args.inbox)}/")
        print(f"Drop some in ({', '.join(sorted(VIDEO_EXTS))}) and run again.")
        return 0
    if args.limit is not None:
        videos = videos[: args.limit]

    # Fail fast before renaming anything: a missing music server should not
    # leave half the queue versioned and unprocessed.
    if scoring and not args.dry_run:
        try:
            from . import generate_music
            server = (_last_value(pipeline_args, "--music-server")
                      or generate_music.DEFAULT_SERVER)
            generate_music.check_server(server)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 5

    ref = _last_value(pipeline_args, "--reference-audio")
    if ref is not None:
        # The pipeline runs from the project root, so check from there too.
        if not paths.resolve(ref).exists():
            print(f"error: reference audio not found: {ref}", file=sys.stderr)
            return 1
    elif _last_value(pipeline_args, "--engine") == "chatterbox":
        print(
            "[info] no reference_audio in config/local_settings.json; narrating "
            "with Chatterbox's default voice.",
            file=sys.stderr,
        )

    data = load_state(args.state)
    print(f"queue      : {_label(args.inbox)}/  ({len(videos)} video"
          f"{'s' if len(videos) != 1 else ''})")
    print(f"numbering  : continuing from {data['last_version']}")
    print(f"pipeline   : {' '.join(pipeline_args)}")
    print(f"retries    : {'--fresh, redo every stage' if args.fresh else 'resume where each clip stopped'}")
    if args.dry_run:
        print("mode       : DRY RUN, nothing will be written")
    print()

    results: list[dict] = []
    t_start = time.perf_counter()

    for i, video in enumerate(videos, start=1):
        version = claim_version(
            args.state, data, video, args.outputs, dry_run=args.dry_run,
        )
        print(f"[{i}/{len(videos)}] {video.name}  ->  version {version}")
        t0 = time.perf_counter()
        try:
            ok, final, outputs, why = run_one(
                video, version, args.inbox, args.outputs,
                pipeline_args, args.dry_run, fresh=args.fresh,
            )
        except KeyboardInterrupt:
            print("\n[interrupted] stopping. Unprocessed videos stay in the inbox, "
                  "and the next run resumes them.", file=sys.stderr)
            break
        except Exception as e:  # one bad clip must not end the batch
            print(f"    [fail] {e}", file=sys.stderr)
            ok, final, outputs, why = False, video, [], str(e)

        elapsed = time.perf_counter() - t0
        results.append({"version": version, "original": video.name, "ok": ok,
                        "outputs": outputs, "seconds": round(elapsed, 1), "why": why})
        if ok and not args.dry_run:
            print(f"    done in {elapsed:.0f}s -> {_label(final.parent)}/")
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
        if not r["ok"] and r.get("why"):
            print(f"               {r['why']}")
    if failed:
        print(f"\n{len(failed)} video(s) stayed in {_label(args.inbox)}/, "
              f"already numbered. Fix the cause and run again: each one resumes "
              f"from the stage where it stopped, under the same number.")
    if done and not args.dry_run:
        print(f"\nfinished clips: {_label(args.outputs)}/<number>/ "
              f"(video, script, voiceover and music together)")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
