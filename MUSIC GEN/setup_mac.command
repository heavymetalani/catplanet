#!/usr/bin/env bash
# One-time ACE-Step 1.5 setup for an Apple Silicon Mac.
# Double-click this file in Finder, or run it from Terminal.
#
# What it does:
#   1. Installs uv (the package manager ACE-Step uses) if missing.
#   2. Runs `uv sync` to create .venv and install all Python deps (PyTorch, MLX, etc).
#   3. Pre-downloads the main ACE-Step 1.5 model checkpoints (~10GB) from Hugging Face.
#
# Re-running this script is safe — it skips work that's already done.

set -euo pipefail

# Resolve the directory this script lives in, then cd into the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/ACE-Step-1.5"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "[error] ACE-Step-1.5 repo not found at: $REPO_DIR" >&2
    echo "        Make sure setup_mac.command lives alongside the ACE-Step-1.5 folder." >&2
    exit 1
fi

cd "$REPO_DIR"

echo "============================================================"
echo " ACE-Step 1.5 — first-time setup"
echo " Repo: $REPO_DIR"
echo "============================================================"

# -------- 1. Ensure uv is installed --------
if ! command -v uv >/dev/null 2>&1; then
    # Also check the default install location, since a fresh install won't be on PATH yet.
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo
        echo "[1/3] uv not found — installing via the official script..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "[1/3] uv: $(command -v uv) ($(uv --version))"

# -------- 2. uv sync (creates .venv + installs deps) --------
echo
echo "[2/3] Installing Python dependencies (this can take several minutes the first time)..."
uv sync

# -------- 3. Pre-download checkpoints --------
echo
echo "[3/3] Downloading model checkpoints to ./checkpoints (~10GB, one-time)..."
echo "      You can Ctrl-C this step; the model will auto-download on first run instead."
uv run acestep-download || {
    echo
    echo "[warn] Pre-download failed. The model will download automatically the first time"
    echo "       you launch the UI — no action needed unless you want to retry now."
}

echo
echo "============================================================"
echo " Setup complete."
echo " Next: double-click  run.command  to launch the Gradio UI."
echo "       It opens at http://localhost:7860"
echo "============================================================"

# If launched by double-click from Finder, keep the window open so the user sees the result.
if [[ -t 0 ]]; then
    :
else
    echo
    read -n 1 -s -r -p "Press any key to close..."
    echo
fi
