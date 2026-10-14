#!/usr/bin/env bash
# Launch the ACE-Step 1.5 Gradio web UI with the MLX backend (Apple Silicon).
# Double-click this file in Finder, or run it from Terminal.
#
# Opens at http://localhost:7860 once the model finishes loading.
# Stop the server with Ctrl-C in this window.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/ACE-Step-1.5"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "[error] ACE-Step-1.5 repo not found at: $REPO_DIR" >&2
    exit 1
fi

# Make sure uv is on PATH (in case setup just ran in this session).
if ! command -v uv >/dev/null 2>&1 && [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv is not installed. Run setup_mac.command first." >&2
    exit 1
fi

if [[ ! -d "$REPO_DIR/.venv" ]]; then
    echo "[error] Python venv not found at $REPO_DIR/.venv" >&2
    echo "        Run setup_mac.command first." >&2
    exit 1
fi

cd "$REPO_DIR"
echo "Launching ACE-Step 1.5 Gradio UI (MLX backend)..."
echo "When it's ready, open:  http://localhost:7860"
echo "Press Ctrl-C to stop the server."
echo

# Delegate to the official macOS launcher script.
exec bash ./start_gradio_ui_macos.sh
