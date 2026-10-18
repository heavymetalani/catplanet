"""
progress.py

Two things that make a failed run cheap to retry.

1. Safe writes
--------------
Every output is written under a hidden temporary name first,
".35.voiceover.wav.partial", and renamed to its real name only once it is
complete. A rename is atomic, so a file with a real name is always whole. If
the Mac crashes or you press Ctrl-C mid-write, all that is left behind is a
".partial" file, which the next run removes.

2. A progress file per clip
---------------------------
In resume mode (pipeline.py --resume, which batch.py always uses) each clip
gets a small "<name>.status.json" next to its outputs. It records, for every
stage that finished, which file it wrote and a fingerprint of the settings and
inputs that produced it:

    describe    video file + vision model, frames, prompt (and your cats)
    script      the description + style, text model, temperature, length
    music_plan  the description + music style, text model, length, BPM
    voice       the script + engine and voice settings
    music       the music plan + seed, sampler and loudness, one entry per take

Each fingerprint includes the one before it, so a change flows downstream: a
new --style redoes the script, the voiceover and nothing else; new cat
descriptions redo everything from the description on. A stage is skipped on a
retry only when its fingerprint matches and its file still exists.

When a stage has to be redone and an older file of it exists, the old file is
moved into an "attempt-N" folder next to the outputs instead of being
overwritten. --fresh ignores the saved progress and redoes every stage, moving
everything it replaces the same way.

The file also records how far the latest run got ("stopped at music take 2 of
3"), which batch.py shows in its summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

STATUS_FORMAT = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================
# Safe writes
# =========================================================
def partial_path(path: Path) -> Path:
    """The hidden temporary name a file is written under before it is final."""
    return path.with_name(f".{path.name}.partial")


def commit_partial(tmp: Path, final: Path) -> Path:
    """Give a finished temporary file its real name, atomically."""
    os.replace(tmp, final)
    return final


def write_text(path: Path, text: str) -> Path:
    """Write a text file safely: complete under its real name, or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = partial_path(path)
    tmp.write_text(text)
    return commit_partial(tmp, path)


def remove_partials(folder: Path, stem: str) -> None:
    """Delete leftovers of an interrupted run for this clip."""
    if not folder.is_dir():
        return
    for p in folder.glob(f".{stem}.*.partial"):
        try:
            p.unlink()
            print(f"[info] removed unfinished file from an interrupted run: {p.name}",
                  file=sys.stderr)
        except OSError:
            pass


# =========================================================
# Fingerprints
# =========================================================
def fingerprint(*parts) -> str:
    """A short hash of settings. Equal settings give an equal fingerprint."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def file_fingerprint(path: Path | None) -> str | None:
    """A hash of a file's contents, so renaming or moving it changes nothing."""
    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# =========================================================
# The per-clip progress file
# =========================================================
class ClipProgress:
    """Reads and writes <stem>.status.json and keeps superseded files.

    With enabled=False every method is a harmless no-op and nothing is ever
    reused, which is how pipeline.py behaves without --resume or --fresh.
    """

    def __init__(self, folder: Path, stem: str, enabled: bool, fresh: bool = False):
        self.folder = folder
        self.stem = stem
        self.enabled = enabled
        self.fresh = fresh
        self.path = folder / f"{stem}.status.json"
        self._attempt_dir: Path | None = None
        self.data = self._load() if enabled else {}

    # ---------------------------------------------------------------- I/O --
    def _load(self) -> dict:
        data: dict = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"[warn] could not read {self.path.name} ({e}); "
                      f"redoing every stage for this clip", file=sys.stderr)
                data = {}
        data.setdefault("format", STATUS_FORMAT)
        data.setdefault("clip", self.stem)
        data.setdefault("stages", {})
        data.setdefault("runs", [])
        if self.fresh:
            # Forget what was done. Old files are still kept: each stage moves
            # its previous output into attempt-N/ before replacing it.
            data["stages"] = {}
        return data

    def save(self) -> None:
        if self.enabled:
            write_text(self.path, json.dumps(self.data, indent=2) + "\n")

    # --------------------------------------------------------------- runs --
    def start_run(self, settings_summary: str) -> None:
        if not self.enabled:
            return
        self.data["runs"].append({
            "started_at": _now(),
            "mode": "fresh" if self.fresh else "resume",
            "settings": settings_summary,
            "result": "running",
            "step": "starting",
        })
        self.data["runs"] = self.data["runs"][-20:]   # keep the file small
        self.save()

    def step(self, label: str) -> None:
        """Record what is running now, so a crash shows where it stopped."""
        if not self.enabled or not self.data["runs"]:
            return
        self.data["runs"][-1]["step"] = label
        self.save()

    def finish_run(self, ok: bool, error: str | None = None) -> None:
        if not self.enabled or not self.data["runs"]:
            return
        run = self.data["runs"][-1]
        run["finished_at"] = _now()
        run["result"] = "ok" if ok else "failed"
        if ok:
            run["step"] = "done"
        if error:
            run["error"] = error[:500]
        self.save()

    # -------------------------------------------- what this run has done --
    def mark(self, stage: str, how) -> None:
        """Note in the current run that `stage` was "done" or "reused".

        For music, `how` is {"done": n, "of": total}. The control panel reads
        this to draw each clip's progress.
        """
        if not self.enabled or not self.data["runs"]:
            return
        self.data["runs"][-1].setdefault("progress", {})[stage] = how
        self.save()

    # ------------------------------------------------------------- stages --
    def reusable(self, stage: str, fp: str, *files: Path) -> bool:
        """True if `stage` already finished with these settings and its files exist."""
        if not self.enabled:
            return False
        rec = self.data["stages"].get(stage)
        return bool(
            rec and rec.get("fingerprint") == fp
            and all(f.exists() for f in files)
        )

    def record(self, stage: str, fp: str, *files: Path, **extra) -> None:
        if not self.enabled:
            return
        self.data["stages"][stage] = {
            "fingerprint": fp,
            "files": [f.name for f in files],
            "done_at": _now(),
            **extra,
        }
        self.mark(stage, "done")

    # music: one entry per take, so a crash at take 3 keeps takes 1 and 2
    def reusable_takes(self, fp: str) -> dict[int, Path]:
        if not self.enabled:
            return {}
        rec = self.data["stages"].get("music")
        if not rec or rec.get("fingerprint") != fp:
            return {}
        found = {}
        for i, name in rec.get("takes", {}).items():
            p = self.folder / name
            if p.exists():
                found[int(i)] = p
        return found

    def record_take(self, fp: str, index: int, path: Path) -> None:
        if not self.enabled:
            return
        rec = self.data["stages"].get("music")
        if not rec or rec.get("fingerprint") != fp:
            rec = {"fingerprint": fp, "takes": {}}
            self.data["stages"]["music"] = rec
        rec["takes"][str(index)] = path.name
        rec["done_at"] = _now()
        self.save()

    def forget(self, stage: str) -> None:
        if self.enabled and self.data["stages"].pop(stage, None) is not None:
            self.save()

    # ------------------------------------------------------- old attempts --
    def keep_old(self, *files: Path) -> None:
        """Move files that are about to be replaced into attempt-N/.

        Only in resume/fresh mode; a plain pipeline.py run overwrites, as it
        always has. One attempt folder per run, created only when needed.
        """
        if not self.enabled:
            return
        existing = [f for f in files if f.exists()]
        if not existing:
            return
        dest = self._attempt_folder()
        for f in existing:
            shutil.move(str(f), str(dest / f.name))
        names = ", ".join(f.name for f in existing)
        print(f"[info] kept the previous {names} in {dest.name}/", file=sys.stderr)

    def _attempt_folder(self) -> Path:
        if self._attempt_dir is None:
            # In a clip's own folder (outputs/35/) plain "attempt-1" is clear;
            # in a shared folder like data/scratch/ prefix the clip's name.
            prefix = "" if self.folder.name == self.stem else f"{self.stem}."
            n = 1
            while (self.folder / f"{prefix}attempt-{n}").exists():
                n += 1
            self._attempt_dir = self.folder / f"{prefix}attempt-{n}"
            self._attempt_dir.mkdir(parents=True)
            # A snapshot of the progress file from before this run, for reference.
            if self.path.exists():
                shutil.copy2(self.path, self._attempt_dir / self.path.name)
        return self._attempt_dir


def describe_stop(status_path: Path) -> str | None:
    """One line on where the latest run stopped, for batch.py's summary."""
    try:
        data = json.loads(status_path.read_text())
        run = data["runs"][-1]
    except (OSError, json.JSONDecodeError, KeyError, IndexError):
        return None
    if run.get("result") == "ok":
        return None
    where = run.get("step", "unknown step")
    err = run.get("error")
    if run.get("result") == "running":
        return f"interrupted during {where}"
    return f"stopped at {where}" + (f": {err.splitlines()[0][:120]}" if err else "")
