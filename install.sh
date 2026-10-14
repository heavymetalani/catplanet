#!/usr/bin/env bash
# install.sh - One-shot setup for the cat video interpreter on macOS (Apple Silicon).
# Safe to re-run; each step checks whether it's already done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { printf "\n\033[1;34m==>\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m ok\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m !!\033[0m %s\n" "$*"; }

# 1. Homebrew
if ! command -v brew >/dev/null 2>&1; then
  log "Installing Homebrew (you'll be prompted for your password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  ok "Homebrew present"
fi

# Make sure brew is on PATH for this session (Apple Silicon default)
if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# 2. Ollama + ffmpeg + python@3.12 + espeak-ng (Kokoro phoneme backend)
for pkg in ollama ffmpeg python@3.12 espeak-ng; do
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    ok "$pkg already installed"
  else
    log "Installing $pkg..."
    brew install "$pkg"
  fi
done

# 3. Ollama service
if ! pgrep -x ollama >/dev/null 2>&1; then
  log "Starting Ollama background service..."
  brew services start ollama
  sleep 3
else
  ok "Ollama service already running"
fi

# 4. Verify Ollama server reachable
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  warn "Ollama daemon not responding yet — waiting up to 15s..."
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if curl -sf http://localhost:11434/api/tags >/dev/null; then break; fi
    sleep 1
  done
fi
curl -sf http://localhost:11434/api/tags >/dev/null \
  && ok "Ollama reachable at localhost:11434" \
  || { warn "Ollama still not reachable. Try: ollama serve"; exit 1; }

# 5. Pull the models
VISION_MODEL="openbmb/minicpm-v4.5"
TEXT_MODEL="qwen2.5:14b"

for MODEL in "$VISION_MODEL" "$TEXT_MODEL"; do
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
    ok "Model $MODEL already pulled"
  else
    log "Pulling $MODEL (large download, may take several minutes)..."
    ollama pull "$MODEL"
  fi
done

# 6. Python virtualenv
# Kokoro requires Python >=3.10,<3.13 so we pin to 3.12 explicitly.
PY312="$(command -v python3.12 || echo /opt/homebrew/bin/python3.12)"
if [[ ! -x "$PY312" ]]; then
  warn "python3.12 not found. brew install python@3.12 should have provided it."
  exit 1
fi

# If an existing venv was built against a different Python version, rebuild it.
if [[ -d .venv ]]; then
  EXISTING_PY="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")"
  if [[ "$EXISTING_PY" != "3.12" ]]; then
    warn "Existing .venv uses Python $EXISTING_PY; Kokoro needs 3.12. Rebuilding..."
    rm -rf .venv
  else
    ok "Virtualenv already exists (Python 3.12)"
  fi
fi

if [[ ! -d .venv ]]; then
  log "Creating Python virtualenv in .venv using $PY312..."
  "$PY312" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
log "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Python deps installed"

# 7. Samples directory
mkdir -p samples descriptions
ok "samples/ and descriptions/ ready"

cat <<EOF

All set. Common commands (after 'source .venv/bin/activate'):

  # Full pipeline, video -> narrated voiceover audio
  python pipeline.py samples/<your_clip>.mov --narrate

  # Full pipeline, no TTS (just prints the script)
  python pipeline.py samples/<your_clip>.mov

  # Individual stages
  python describe_video.py  samples/<your_clip>.mov          # stage 1: description
  python generate_script.py --description d.txt --length 28  # stage 2: script
  python generate_voice.py  --script s.txt --out out.wav     # stage 3: TTS

Drop a cat video into ./samples/ and point the script at it.
EOF
