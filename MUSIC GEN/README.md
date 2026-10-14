# ACE-Step 1.5 on Apple Silicon

This folder holds the launchers and notes for running ACE-Step 1.5 on an M4 Mac mini. ACE-Step itself is not included in the git repository: it is a 13 GB clone of someone else's project, with its own git history and model weights. Step 0 below downloads it.

## What's here

```
MUSIC GEN/
├── ACE-Step-1.5/              # NOT in git -- downloaded in step 0 below
│   └── .env                   # copied from acestep.env.example in step 0
├── acestep.env.example        # Apple Silicon settings (MLX backend) for ACE-Step's .env
├── setup_mac.command          # ← Run first to install dependencies + download models
├── run.command                # ← Launches the Gradio UI
├── start_api_server.command   # ← Launches the REST API that pipeline.py uses for music
└── README.md                  # This file
```

The Python install and the model checkpoints aren't here yet — they download when you run `setup_mac.command`. That's because Python packages like PyTorch and MLX have to be installed natively on macOS, not from a Linux build environment.

## First-time setup

0. Download ACE-Step at the exact version this project was built and tested against, then copy in the settings. From the project root, in Terminal:

   ```bash
   cd "MUSIC GEN"
   git clone https://github.com/ace-step/ACE-Step-1.5.git
   git -C ACE-Step-1.5 checkout dce6214
   cp acestep.env.example ACE-Step-1.5/.env
   ```

   `dce6214` is the upstream commit from 18 May 2026. Newer versions will probably work, but this one is known to.

1. Open Finder and navigate to this folder.
2. Double-click `setup_mac.command`.
   - macOS may warn about an unverified developer. If so: right-click → Open → Open anyway. Or, in Terminal, run `bash setup_mac.command`.
3. Wait. The script will:
   - Install `uv` (Python package manager) if it isn't already.
   - Create a `.venv` inside `ACE-Step-1.5/` and install PyTorch, MLX, Gradio, and the rest of ACE-Step's dependencies.
   - Download the main model checkpoints (~10GB) into `ACE-Step-1.5/checkpoints/`.
4. When you see `Setup complete.`, you're ready.

Total first-run time depends on your network — typically 10–30 minutes for the model download.

## Daily use

Double-click `run.command`. It will:
- Activate the venv.
- Launch the Gradio web UI with the MLX backend (native Apple Silicon acceleration).
- Print a local URL — open `http://localhost:7860` in your browser.

To stop the server, press `Ctrl-C` in the Terminal window that `run.command` opened.

## Tuning for your Mac mini

The default config lives in `ACE-Step-1.5/.env`. Defaults are tuned for a 16GB unified-memory M4 Mac mini:

| Setting | Default | When to change |
| --- | --- | --- |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | Switch to `acestep-v15-sft` for higher quality at the cost of more steps. |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-0.6B` | On a 24GB+ Mac, upgrade to `acestep-5Hz-lm-1.7B` for better lyric/chain-of-thought planning. |
| `PORT` | `7860` | Change if another app is using port 7860. |
| `SERVER_NAME` | `127.0.0.1` | Set to `0.0.0.0` to access the UI from other devices on your LAN. |
| `LANGUAGE` | `en` | `zh`, `ja`, or `he` for non-English UI. |

After editing `.env`, just restart with `run.command` — no rebuild needed. If you change a setting you want to keep, copy it into `acestep.env.example` too, since `ACE-Step-1.5/.env` is not in git.

## Troubleshooting

**macOS blocks the .command files.** Right-click → Open → Open anyway, or run `chmod +x setup_mac.command run.command` once.

**"uv: command not found"** even after setup. The installer puts uv in `~/.local/bin`. Either restart your terminal or run `export PATH="$HOME/.local/bin:$PATH"`.

**Out of memory.** Lower the LM by setting `ACESTEP_INIT_LLM=false` in `.env` to run DiT-only — generation still works, you just lose the LLM-driven prompt rewriting.

**Want to update later.** From inside `ACE-Step-1.5/`, run `git pull` (or `bash check_update.sh`), then re-run `setup_mac.command` to update dependencies.

## More info

- Official docs: <https://github.com/ace-step/ACE-Step-1.5>
- Installation guide: <https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md>
- Tutorial (prompting, model selection, hyperparameters): <https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md>
