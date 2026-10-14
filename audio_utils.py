#!/usr/bin/env python3
"""
audio_utils.py

Loudness measurement and static-gain normalization, shared by stage 3
(voiceover) and stage 4 (music).

Why static gain
---------------
The obvious way to hit a loudness target in ffmpeg is to put `loudnorm` in the
filter chain. Don't. In single-pass mode loudnorm is a *dynamic* processor: it
rides the gain continuously and look-ahead limits. On sustained material -- a
bowed string, a held vowel -- that is audible as distortion, measured at ~16 dB
signal-to-error on real cues from this pipeline.

So loudnorm is used here only as an analyzer, in a throwaway `-f null` pass.
The gain that actually reaches the audio is one constant multiplier, which is
exact for this job and provably transparent (~109 dB signal-to-error).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

# Standard delivery levels, for reference:
#   -14 LUFS  YouTube / streaming normalization
#   -16 LUFS  podcast and short-form web -- our narration target
#   -23 LUFS  EBU R128 broadcast
NARRATION_LUFS = -16.0

# Keep true peak below this after any gain, so nothing clips downstream.
TRUE_PEAK_CEILING_DB = -2.0


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg"))


def _parse_loudnorm_json(stderr: str) -> dict | None:
    """loudnorm prints its JSON report last on stderr."""
    start, end = stderr.rfind("{"), stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None


def analyze(
    source: Path | bytes, pre_filter: str = ""
) -> tuple[float | None, float | None]:
    """Measure (integrated LUFS, true peak dBTP), optionally after `pre_filter`.

    Measurement only -- this filter graph's audio is discarded.
    """
    if not have_ffmpeg():
        return None, None

    cmd = ["ffmpeg", "-nostdin", "-i"]
    if isinstance(source, bytes):
        cmd.append("pipe:0")
    else:
        if not source.exists():
            return None, None
        cmd.append(str(source))

    chain = f"{pre_filter}," if pre_filter else ""
    cmd += ["-af", f"{chain}loudnorm=print_format=json", "-f", "null", "-"]

    result = subprocess.run(
        cmd,
        input=source if isinstance(source, bytes) else None,
        capture_output=True,
    )
    data = _parse_loudnorm_json(result.stderr.decode("utf-8", "replace"))
    if not data:
        return None, None

    def _f(key: str) -> float | None:
        try:
            v = float(data[key])
        except (KeyError, ValueError, TypeError):
            return None
        # loudnorm reports digital silence as -99 / -120.
        return None if v <= -90 else v

    return _f("input_i"), _f("input_tp")


def measure_lufs(path: Path) -> float | None:
    """Integrated loudness of a file. None on failure."""
    return analyze(path)[0]


def gain_to_target(
    measured_i: float | None,
    measured_tp: float | None,
    target_lufs: float,
    ceiling_db: float = TRUE_PEAK_CEILING_DB,
    label: str = "audio",
) -> float:
    """The single gain, in dB, that puts `measured_i` at `target_lufs`.

    Reduced if it would push true peak above `ceiling_db` -- staying clean
    matters more than hitting the number exactly.
    """
    if measured_i is None:
        print(f"[warn] could not measure {label} loudness; leaving level alone",
              file=sys.stderr)
        return 0.0

    gain = target_lufs - measured_i
    if measured_tp is not None and measured_tp + gain > ceiling_db:
        capped = ceiling_db - measured_tp
        print(
            f"[warn] {label} at {target_lufs:.1f} LUFS would peak at "
            f"{measured_tp + gain:+.1f} dBTP; holding gain at {capped:+.1f} dB "
            f"to stay under {ceiling_db} dBTP",
            file=sys.stderr,
        )
        return capped
    return gain
