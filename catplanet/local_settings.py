#!/usr/bin/env python3
"""
local_settings.py

Loads personal settings from config/local_settings.json, which stays on your
machine and is never committed (it is listed in .gitignore).

What lives there
----------------
    reference_audio   Path to the voice clip Chatterbox clones for narration,
                      relative to the project root (e.g. data/voices/me.wav).
                      Used by batch.py. A recording of a real person's voice is
                      personal data, so the path stays out of the code.
    cats              Your cats' names, pronouns and coat descriptions. Used by
                      describe_video.py so the vision model can tell them apart.
                      An optional "personality" per cat is used by
                      generate_script.py for the jokes.
    priority_moments  Moments your audience likes best ("the cats play
                      fighting"). generate_script.py makes a matching moment
                      the main action of the script; earlier entries win.

To set up a fresh copy of the project:

    cp config/local_settings.example.json config/local_settings.json

then edit config/local_settings.json with your own values. Every setting is
optional: with no file at all, the pipeline still runs, using Chatterbox's
default voice and describing cats without names.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import paths

LOCAL_FILE = paths.LOCAL_SETTINGS
EXAMPLE_FILE = paths.LOCAL_SETTINGS_EXAMPLE

_cache: dict | None = None


def load(path: Path = LOCAL_FILE) -> dict:
    """Return the settings as a dict, or {} if the file does not exist.

    A malformed file is reported rather than silently ignored, because quietly
    dropping your cat names or voice would change the output without warning.
    """
    global _cache
    if path == LOCAL_FILE and _cache is not None:
        return _cache

    if not path.exists():
        data: dict = {}
    else:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"[warn] could not read {path.name}: {e}. Continuing without "
                f"personal settings.",
                file=sys.stderr,
            )
            data = {}
        if not isinstance(data, dict):
            print(f"[warn] {path.name} should contain a JSON object; ignoring it.",
                  file=sys.stderr)
            data = {}

    if path == LOCAL_FILE:
        _cache = data
    return data


def get(key: str, default=None):
    """One setting, or `default` if it is missing or empty."""
    value = load().get(key)
    return default if value in (None, "", []) else value
