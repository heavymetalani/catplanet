#!/usr/bin/env bash
# phase1_quartet_test.sh — MUSIC_PLAN.md §9, phase 1.
#
# Submits ONE hand-written string-quartet caption to the ACE-Step API,
# waits for it, downloads the takes, and reports how long it took.
#
# The point of this test: find out whether ACE-Step will hold a clean
# quartet texture (no drums, no piano, no synth pads) from a caption alone,
# BEFORE any pipeline code gets written.
#
# Prereq: the API server must already be running in another terminal:
#   "MUSIC GEN/start_api_server.command"
# (Use that, not ACE-Step's own start_api_server_macos.sh — the upstream one
#  uses bash-4 syntax and dies on macOS's bash 3.2.)
#
# Usage:
#   bash phase1_quartet_test.sh                  # default: stately adagio
#   bash phase1_quartet_test.sh agitato          # try the "sports" preset
#   bash phase1_quartet_test.sh brooding         # try the "noir" preset

set -uo pipefail

SERVER="${ACESTEP_SERVER:-http://127.0.0.1:8001}"
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/music_tests"
PRESET="${1:-adagio}"
TAKES="${TAKES:-2}"
DURATION="${DURATION:-30}"

# ---------------------------------------------------------------- presets --
# Character words only. BPM and key travel in the metadata fields, never in
# the caption — see MUSIC GEN/instructions.md §4.3.
case "$PRESET" in
  adagio)   # the "attenborough" frame
    CHARACTER="stately adagio, sustained legato lines with gentle pizzicato underneath, patient and quietly dignified"
    BPM=68;  KEY="C Major" ;;
  agitato)  # the "sports" frame
    CHARACTER="agitato, rapid spiccato bowing over a driving cello ostinato, breathless and building"
    BPM=144; KEY="D Major" ;;
  brooding) # the "noir" frame
    CHARACTER="brooding and sparse, muted sul tasto violins, chromatic descending cello line, unresolved"
    BPM=76;  KEY="D Minor" ;;
  tense)    # the "reality_tv" frame
    CHARACTER="tense tremolo strings, sudden unison stabs, suspensions that refuse to resolve"
    BPM=100; KEY="A Minor" ;;
  *)
    echo "unknown preset '$PRESET' — choose: adagio | agitato | brooding | tense" >&2
    exit 1 ;;
esac

CAPTION="solo classical string quartet, two violins, viola and cello, ${CHARACTER}, chamber recording, warm close-miked acoustic, natural room decay, instrumental, no percussion, no drums, no piano, no synth, no vocals"

# ------------------------------------------------------------ preflight ----
if ! curl -sf --max-time 5 "${SERVER}/health" >/dev/null 2>&1; then
  cat >&2 <<EOF

[error] No ACE-Step server at ${SERVER}

Start it first, in a separate terminal window, and leave it running:

    "$(dirname "${BASH_SOURCE[0]}")/MUSIC GEN/start_api_server.command"

First launch loads ~5GB of model weights — wait for the line that says the
server is running on port 8001 before coming back here.

EOF
  exit 1
fi

mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "server    : ${SERVER}"
echo "preset    : ${PRESET}  (${BPM} BPM, ${KEY})"
echo "duration  : ${DURATION}s   takes: ${TAKES}"
echo
echo "caption   : ${CAPTION}"
echo

# ------------------------------------------------------------- submit -----
PAYLOAD=$(cat <<EOF
{
  "prompt":           "${CAPTION}",
  "lyrics":           "",
  "audio_duration":   ${DURATION},
  "bpm":              ${BPM},
  "key_scale":        "${KEY}",
  "time_signature":   "4",
  "inference_steps":  8,
  "infer_method":     "ode",
  "batch_size":       ${TAKES},
  "audio_format":     "wav",
  "thinking":         false,
  "use_cot_caption":  false,
  "use_cot_language": false
}
EOF
)

echo "[submit] posting to /release_task ..."
START=$SECONDS

RESP=$(curl -sf -X POST "${SERVER}/release_task" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD") || { echo "[error] submit failed" >&2; exit 2; }

TASK_ID=$(printf '%s' "$RESP" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["data"]["task_id"])' 2>/dev/null)

if [[ -z "${TASK_ID:-}" ]]; then
  echo "[error] no task_id in response:" >&2
  printf '%s\n' "$RESP" >&2
  exit 2
fi
echo "[submit] task_id = ${TASK_ID}"
echo

# --------------------------------------------------------------- poll -----
# status: 0 = running, 1 = succeeded, 2 = failed
echo "[wait] polling /query_result every 3s (Ctrl-C to give up) ..."
FILES=""
for _ in $(seq 1 200); do
  sleep 3
  Q=$(curl -sf -X POST "${SERVER}/query_result" \
        -H 'Content-Type: application/json' \
        -d "{\"task_id_list\": [\"${TASK_ID}\"]}") || continue

  read -r STATUS FILES <<<"$(printf '%s' "$Q" | python3 -c '
import json, sys
d = json.load(sys.stdin)["data"][0]
st = d.get("status", 0)
files = []
if st == 1:
    try:
        files = [x["file"] for x in json.loads(d.get("result") or "[]") if x.get("file")]
    except Exception:
        pass
print(st, " ".join(files))
' 2>/dev/null)"

  case "${STATUS:-0}" in
    1) echo "[wait] done in $((SECONDS - START))s"; break ;;
    2) echo "[error] generation failed. Full response:" >&2
       printf '%s\n' "$Q" >&2; exit 3 ;;
    *) printf '.' ;;
  esac
done
echo

if [[ -z "${FILES// }" ]]; then
  echo "[error] timed out or no audio returned" >&2
  exit 4
fi

# ----------------------------------------------------------- download -----
i=1
for f in $FILES; do
  OUT="${OUTDIR}/${STAMP}-${PRESET}-take${i}.wav"
  if curl -sf "${SERVER}${f}" -o "$OUT"; then
    echo "[saved] $OUT"
  else
    echo "[warn] could not download ${f}" >&2
  fi
  i=$((i + 1))
done

ELAPSED=$((SECONDS - START))
cat <<EOF

────────────────────────────────────────────────────────────
Total: ${ELAPSED}s for ${TAKES} take(s) of ${DURATION}s audio.

Now LISTEN to them, and answer three questions:

  1. Is it actually a string quartet?  Any drums, piano, or synth
     pads means caption-only steering is not enough — the answer
     is LoKr training (MUSIC GEN/instructions.md §3), not more
     pipeline code.
  2. Does it hold together for the full ${DURATION}s, or does it
     rush / wander / stop early?
  3. Is ${ELAPSED}s per clip acceptable? That's the per-video cost
     stage 4 will add.

Files are in: ${OUTDIR}
────────────────────────────────────────────────────────────
EOF
