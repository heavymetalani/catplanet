#!/usr/bin/env bash
# Launch the ACE-Step 1.5 REST API server (MLX backend, Apple Silicon).
# Double-click this file in Finder, or run it from Terminal.
#
# Why this exists instead of ACE-Step-1.5/start_api_server_macos.sh:
#   That upstream script uses bash-4 syntax (${var^^}) in its git-update and
#   uv-install prompts. macOS ships bash 3.2, so it dies with
#   "bad substitution" on line 117 before it ever starts the server.
#   This launcher does the same job without those two prompts, and lives
#   OUTSIDE the ACE-Step checkout so `git pull` there stays conflict-free.
#
# Serves on http://127.0.0.1:8001  (interactive docs at /docs)
# Stop the server with Ctrl-C in this window.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/ACE-Step-1.5"

HOST="${ACESTEP_HOST:-127.0.0.1}"
PORT="${ACESTEP_PORT:-8001}"

# ==================== MLX / runtime config ====================
# Same environment the upstream macOS launcher sets.
export ACESTEP_LM_BACKEND="mlx"      # native Apple Silicon acceleration
export TOKENIZERS_PARALLELISM="false"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"

# ==================== Preflight ====================
if [[ ! -d "$REPO_DIR" ]]; then
    echo "[error] ACE-Step-1.5 not found at: $REPO_DIR" >&2
    exit 1
fi

if [[ "$(uname)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "[warn] Not Apple Silicon — unsetting MLX backend, falling back to PyTorch."
    unset ACESTEP_LM_BACKEND
fi

# uv may live in ~/.local/bin (installer default) and not be on a GUI PATH.
if ! command -v uv >/dev/null 2>&1; then
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin" "/opt/homebrew/bin"; do
        [[ -x "$candidate/uv" ]] && export PATH="$candidate:$PATH" && break
    done
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv is not installed. Run setup_mac.command first," >&2
    echo "        or: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -d "$REPO_DIR/.venv" ]]; then
    echo "[error] Python venv not found at $REPO_DIR/.venv" >&2
    echo "        Run setup_mac.command first." >&2
    exit 1
fi

# Warn early if something is already on the port, rather than failing cryptically.
if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[warn] Something is already listening on port $PORT."
    echo "       If that's an ACE-Step server you started earlier, you're set —"
    echo "       just close this window. Otherwise stop it, or set ACESTEP_PORT."
    echo
fi

# ==================== Launch ====================
cat <<EOF
============================================
  ACE-Step 1.5 API — macOS Apple Silicon (MLX)
============================================

  API:   http://${HOST}:${PORT}
  Docs:  http://${HOST}:${PORT}/docs
  Check: curl http://${HOST}:${PORT}/health

First launch loads ~5GB of weights — give it a minute or two.
Leave this window open; Ctrl-C stops the server.

EOF

cd "$REPO_DIR"
exec uv run acestep-api --host "$HOST" --port "$PORT"
