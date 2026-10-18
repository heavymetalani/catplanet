"""
paths.py

Every folder the pipeline reads from or writes to, in one place. If you ever
want to move something, change it here and nowhere else.

    video-interpreter/
    ├── config/
    │   └── local_settings.json   your voice clip + cats (gitignored)
    ├── data/                     everything personal and generated (gitignored)
    │   ├── inbox/                drop new videos here; batch.py works through it
    │   ├── voices/               reference voice clips for Chatterbox to clone
    │   ├── outputs/<N>/          one folder per finished clip
    │   └── batch_state.json      the clip-number counter and a log of every run
    └── MUSIC GEN/                the ACE-Step music server
"""
from __future__ import annotations

from pathlib import Path

# The project root: the folder that holds catplanet/, data/, config/ ...
REPO = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO / "config"
LOCAL_SETTINGS = CONFIG_DIR / "local_settings.json"
LOCAL_SETTINGS_EXAMPLE = CONFIG_DIR / "local_settings.example.json"

DATA_DIR = REPO / "data"
INBOX_DIR = DATA_DIR / "inbox"
VOICES_DIR = DATA_DIR / "voices"
OUTPUTS_DIR = DATA_DIR / "outputs"
BATCH_STATE = DATA_DIR / "batch_state.json"

MUSIC_GEN_DIR = REPO / "MUSIC GEN"
MUSIC_SERVER_LAUNCHER = MUSIC_GEN_DIR / "start_api_server.command"


def resolve(path: str | Path) -> Path:
    """A path from a settings file or the command line, made absolute.

    Relative paths are read from the project root, so "data/voices/me.wav"
    means the same thing no matter which folder you run a command from.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p
