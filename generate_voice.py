#!/usr/bin/env python3
"""
generate_voice.py

Stage 3 of the cat-video pipeline. Converts a voiceover script into a .wav
audio file using one of two TTS engines:

    kokoro      - 82M params, Apache 2.0, voice PRESETS (no cloning).
                  Ships with British male documentary-style voices out of the box.
                  Fastest, simplest, no reference audio required.

    chatterbox  - 500M params (Resemble AI), MIT license, voice CLONING.
                  Clones any voice from a 5-10 second reference .wav. Also has
                  explicit emotion/exaggeration control. Use this when you want
                  a specific voice (your own, a friend's, or a royalty-free sample)
                  rather than a preset.

Usage
-----
    # Kokoro (default) — pick a preset British voice, no reference audio
    python generate_voice.py --script s.txt --out out.wav
    python generate_voice.py --script s.txt --out out.wav --voice bm_lewis --speed 0.95

    # Chatterbox with default voice (no cloning)
    python generate_voice.py --script s.txt --out out.wav --engine chatterbox

    # Chatterbox cloning from a reference audio sample (5-10s of target voice)
    python generate_voice.py --script s.txt --out out.wav \\
        --engine chatterbox --reference-audio voices/my_narrator.wav \\
        --exaggeration 0.6

    # From stdin
    python generate_script.py --description d.txt --length 28 \\
        | python generate_voice.py --out out.wav --engine chatterbox

Kokoro voice presets (best fit for documentary narration)
---------------------------------------------------------
    bm_george   - warm, mature (default)
    bm_lewis    - slightly younger, more conversational
    bm_daniel   - crisper, faster cadence
    bm_fable    - storyteller-ish, good for longer narration

Chatterbox controls
-------------------
    --reference-audio PATH  A 5-10 second .wav of the voice to clone. Optional;
                            Chatterbox uses a default voice if omitted.
    --exaggeration FLOAT    0.0-1.0 (default 0.5). Higher = more emotive delivery.
                            Try 0.3 for deadpan Attenborough, 0.7 for sports energy.
    --cfg-weight FLOAT      0.0-1.0 (default 0.5). Classifier-free guidance.
                            Lower = more natural pacing, higher = more adherence to text.
    --device {mps,cpu}      Backend device (default: auto — mps if available).
    --max-chars INT         Max chars per synthesized chunk (default: 250). Long
                            scripts are split at paragraph/sentence boundaries to
                            avoid garbled output. Set to 0 to disable chunking.

Requires
--------
    pip install -r requirements.txt
    brew install espeak-ng    (Kokoro only)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

import audio_utils

# ---------------- engine-agnostic defaults ----------------
DEFAULT_ENGINE: Literal["kokoro", "chatterbox"] = "kokoro"

# ---------------- Kokoro defaults ----------------
DEFAULT_VOICE = "bm_george"
KOKORO_SAMPLE_RATE = 24_000
KOKORO_VOICE_PREFIX_TO_LANG = {
    "a": "a", "b": "b", "e": "e", "f": "f",
    "h": "h", "i": "i", "j": "j", "p": "p", "z": "z",
}

# ---------------- Chatterbox defaults ----------------
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
# Chatterbox produces garbled output on very long inputs; chunk beyond this.
# Empirically ~250 chars is safely within the reliable range.
DEFAULT_CHATTERBOX_MAX_CHARS = 250
# Silence inserted between synthesized chunks to sound natural.
SENTENCE_SILENCE_SEC = 0.15
PARAGRAPH_SILENCE_SEC = 0.40

# ---------------- pause trimming ----------------
# The attenborough style prompt asks for ellipses, em dashes and paragraph
# breaks as dramatic beats, and the TTS engines honor every one of them. Read
# aloud that lands as long dead gaps -- fine for a documentary, too slow for a
# 17-second clip. Any internal silence longer than DEFAULT_MAX_PAUSE_SEC gets
# shortened to it. Set max_pause=0 to keep the engine's own timing.
DEFAULT_MAX_PAUSE_SEC = 0.35
SILENCE_THRESHOLD_DB = -45.0

# ---------------- output level ----------------
# Neither engine normalizes: Kokoro lands around -24 LUFS, roughly 8 dB under
# a normal narration level. That is quiet on its own, and it also drags the
# music down, because the music bed is leveled relative to this file. Fixing
# the reference fixes both. Set to None (or --target-lufs 0) to leave the
# engine's own level alone.
DEFAULT_TARGET_LUFS = audio_utils.NARRATION_LUFS  # -16 LUFS


# =========================================================
# Kokoro backend
# =========================================================
def _kokoro_lang_for_voice(voice: str) -> str:
    if not voice:
        return "b"
    return KOKORO_VOICE_PREFIX_TO_LANG.get(voice[0], "b")


def synthesize_kokoro(text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
    """Run Kokoro. Returns (audio_float32, sample_rate)."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=_kokoro_lang_for_voice(voice))
    generator = pipeline(text, voice=voice, speed=speed)

    chunks: list[np.ndarray] = []
    for _graphemes, _phonemes, audio in generator:
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio. Is the text empty?")

    full = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return full, KOKORO_SAMPLE_RATE


# =========================================================
# Chunking (for Chatterbox only)
# =========================================================
#
# Chatterbox produces garbled output ("rubbish phonemes") when the input
# exceeds its reliable context length. We chunk long scripts into pieces
# that honor paragraph and sentence boundaries, synthesize each piece,
# and concatenate with natural silence between them.


def _split_by_sentence(text: str, max_chars: int) -> list[str]:
    """Split `text` at sentence terminators, packing sentences into chunks <= max_chars."""
    # Split after . ! ? ... followed by whitespace. Keep the terminator on the preceding sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sent) <= max_chars:
            current = sent
        else:
            # One sentence is itself too long — hard-split on word boundaries.
            words = sent.split()
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 <= max_chars:
                    buf = f"{buf} {w}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = w
            if buf:
                current = buf
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHATTERBOX_MAX_CHARS,
) -> list[tuple[str, str]]:
    """Split `text` into synthesizable chunks.

    Returns list of (chunk, boundary_kind) tuples, where boundary_kind is one of:
      "sentence"  - mid-paragraph break; short silence after this chunk
      "paragraph" - end of a paragraph; longer silence after this chunk
      "end"       - final chunk; no trailing silence

    Paragraphs are inferred from blank-line separators. If a paragraph is
    itself longer than max_chars, it's split at sentence boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: list[tuple[str, str]] = []

    for p_idx, para in enumerate(paragraphs):
        is_last_para = p_idx == len(paragraphs) - 1
        if len(para) <= max_chars:
            boundary = "end" if is_last_para else "paragraph"
            result.append((para, boundary))
        else:
            sub_chunks = _split_by_sentence(para, max_chars)
            for s_idx, sub in enumerate(sub_chunks):
                is_last_sub = s_idx == len(sub_chunks) - 1
                if is_last_sub:
                    boundary = "end" if is_last_para else "paragraph"
                else:
                    boundary = "sentence"
                result.append((sub, boundary))
    return result


# =========================================================
# Chatterbox backend
# =========================================================
def _chatterbox_device(preferred: str | None) -> str:
    """Resolve the torch device. Prefer MPS on Apple Silicon, fall back to CPU."""
    import torch

    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Cache the loaded Chatterbox model so re-calls don't repay the load cost.
_CHATTERBOX_MODEL = None
_CHATTERBOX_DEVICE: str | None = None


def _load_chatterbox(device: str):
    global _CHATTERBOX_MODEL, _CHATTERBOX_DEVICE
    if _CHATTERBOX_MODEL is not None and _CHATTERBOX_DEVICE == device:
        return _CHATTERBOX_MODEL
    from chatterbox.tts import ChatterboxTTS

    _CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(device=device)
    _CHATTERBOX_DEVICE = device
    return _CHATTERBOX_MODEL


def _chatterbox_wav_to_mono_np(wav) -> np.ndarray:
    """Normalize whatever Chatterbox returns into a 1-D float32 numpy array."""
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.squeeze(0) if wav.shape[0] == 1 else wav.mean(axis=0)
    return wav


def synthesize_chatterbox(
    text: str,
    reference_audio: Path | None = None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    device: str | None = None,
    max_chars_per_chunk: int = DEFAULT_CHATTERBOX_MAX_CHARS,
) -> tuple[np.ndarray, int]:
    """Run Chatterbox. Returns (audio_float32_mono, sample_rate).

    Long scripts are split into paragraph/sentence-aware chunks (<= max_chars_per_chunk)
    and synthesized one at a time, then concatenated with natural silence between
    them. This avoids the garbled output Chatterbox produces on very long inputs.
    Set max_chars_per_chunk to 0 to disable chunking entirely.
    """
    resolved_device = _chatterbox_device(device)
    model = _load_chatterbox(resolved_device)

    kwargs = {
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
    }
    if reference_audio is not None:
        if not reference_audio.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio}"
            )
        kwargs["audio_prompt_path"] = str(reference_audio)

    # Decide whether to chunk. Short scripts and disabled chunking go straight through.
    should_chunk = max_chars_per_chunk and len(text.strip()) > max_chars_per_chunk
    if not should_chunk:
        wav = model.generate(text, **kwargs)
        return _chatterbox_wav_to_mono_np(wav), int(model.sr)

    chunks = chunk_text(text, max_chars=max_chars_per_chunk)
    if not chunks:
        raise RuntimeError("No chunks produced from non-empty text — check chunker.")

    sr = int(model.sr)
    sentence_silence = np.zeros(int(SENTENCE_SILENCE_SEC * sr), dtype=np.float32)
    paragraph_silence = np.zeros(int(PARAGRAPH_SILENCE_SEC * sr), dtype=np.float32)

    pieces: list[np.ndarray] = []
    total = len(chunks)
    for i, (chunk, boundary) in enumerate(chunks, start=1):
        print(
            f"[info]   chunk {i}/{total} ({len(chunk)} chars, boundary={boundary})...",
            file=sys.stderr,
            flush=True,
        )
        wav = model.generate(chunk, **kwargs)
        pieces.append(_chatterbox_wav_to_mono_np(wav))
        if boundary == "sentence":
            pieces.append(sentence_silence)
        elif boundary == "paragraph":
            pieces.append(paragraph_silence)
        # "end" gets no trailing silence

    full = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    return full, sr


# =========================================================
# Pause trimming (engine-agnostic)
# =========================================================
def compress_silences(
    audio: np.ndarray,
    sr: int,
    max_pause: float = DEFAULT_MAX_PAUSE_SEC,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    frame_ms: float = 10.0,
) -> np.ndarray:
    """Shorten any internal silence longer than `max_pause` down to `max_pause`.

    Works on whatever the engine produced, so it fixes Kokoro's punctuation
    pauses and Chatterbox's inserted gaps alike.

    Cuts happen strictly *inside* silent regions and keep max_pause/2 of the
    original silence on each side, so the decay into the pause and the breath
    out of it survive and there is nothing audible to click.
    """
    if max_pause <= 0 or audio.size == 0:
        return audio

    hop = max(1, int(sr * frame_ms / 1000.0))
    n_frames = int(np.ceil(audio.size / hop))
    pad = n_frames * hop - audio.size
    padded = np.pad(audio, (0, pad)) if pad else audio
    frames = padded.reshape(n_frames, hop)

    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(np.max(np.abs(audio))) or 1.0
    quiet = rms < (peak * (10.0 ** (threshold_db / 20.0)))

    keep = np.ones(audio.size, dtype=bool)
    max_len = int(max_pause * sr)
    half = max_len // 2

    # Walk runs of quiet frames; trim the middle out of the long ones.
    start = None
    for i in range(n_frames + 1):
        is_quiet = quiet[i] if i < n_frames else False
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            s, e = start * hop, min(i * hop, audio.size)
            # Leading and trailing silence isn't a pause between words; leave it.
            if s > 0 and e < audio.size and (e - s) > max_len:
                keep[s + half : e - half] = False
            start = None

    return audio[keep]


# =========================================================
# Public entry point
# =========================================================
def generate_voice(
    text: str,
    output_path: Path,
    engine: str = DEFAULT_ENGINE,
    # Kokoro params
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    # Chatterbox params
    reference_audio: Path | None = None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    device: str | None = None,
    max_chars_per_chunk: int = DEFAULT_CHATTERBOX_MAX_CHARS,
    max_pause: float = DEFAULT_MAX_PAUSE_SEC,
    target_lufs: float | None = DEFAULT_TARGET_LUFS,
) -> Path:
    """Synthesize `text` to a .wav at `output_path`. Returns the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "kokoro":
        audio, sr = synthesize_kokoro(text=text, voice=voice, speed=speed)
    elif engine == "chatterbox":
        audio, sr = synthesize_chatterbox(
            text=text,
            reference_audio=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            device=device,
            max_chars_per_chunk=max_chars_per_chunk,
        )
    else:
        raise ValueError(
            f"Unknown engine '{engine}'. Choose from: kokoro, chatterbox"
        )

    if max_pause > 0:
        before = audio.size / sr
        audio = compress_silences(audio, sr, max_pause=max_pause)
        after = audio.size / sr
        if before - after > 0.05:
            print(
                f"[info] trimmed {before - after:.1f}s of pauses "
                f"({before:.1f}s -> {after:.1f}s)",
                file=sys.stderr,
            )

    sf.write(str(output_path), audio, sr)

    # Level the narration with one constant gain. Measured from the file we
    # just wrote, applied to the samples in memory, rewritten -- no dynamic
    # processing anywhere near the audio.
    if target_lufs is not None:
        measured_i, measured_tp = audio_utils.analyze(output_path)
        gain_db = audio_utils.gain_to_target(
            measured_i, measured_tp, target_lufs, label="voiceover"
        )
        if abs(gain_db) > 0.01:
            audio = (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)
            sf.write(str(output_path), audio, sr)
            print(
                f"[info] voiceover {measured_i:.1f} -> {target_lufs:.1f} LUFS "
                f"({gain_db:+.1f} dB)",
                file=sys.stderr,
            )

    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a voiceover .wav from a script (Kokoro or Chatterbox).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--script",
        type=str,
        default=None,
        help="Path to script text. Omit or pass '-' for stdin.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output .wav path.")
    ap.add_argument(
        "--engine",
        choices=["kokoro", "chatterbox"],
        default=DEFAULT_ENGINE,
        help=f"TTS engine (default: {DEFAULT_ENGINE}).",
    )

    # Kokoro-only
    ap.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Kokoro voice preset (default: {DEFAULT_VOICE}). Ignored if engine=chatterbox.",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Kokoro speed multiplier (default: 1.0). Ignored if engine=chatterbox.",
    )

    # Chatterbox-only
    ap.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help="Chatterbox reference .wav (5-10s of target voice). If omitted, uses default voice.",
    )
    ap.add_argument(
        "--exaggeration",
        type=float,
        default=DEFAULT_EXAGGERATION,
        help=f"Chatterbox emotion intensity 0-1 (default: {DEFAULT_EXAGGERATION}).",
    )
    ap.add_argument(
        "--cfg-weight",
        type=float,
        default=DEFAULT_CFG_WEIGHT,
        help=f"Chatterbox CFG weight 0-1 (default: {DEFAULT_CFG_WEIGHT}).",
    )
    ap.add_argument(
        "--device",
        choices=["mps", "cpu", "cuda"],
        default=None,
        help="Torch device for Chatterbox (default: auto-detect mps on Mac).",
    )
    ap.add_argument(
        "--target-lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help=(
            f"Normalize the voiceover to this loudness (default: "
            f"{DEFAULT_TARGET_LUFS}, the usual podcast/short-form level). "
            f"Neither engine normalizes on its own. Pass 0 to skip."
        ),
    )
    ap.add_argument(
        "--max-pause",
        type=float,
        default=DEFAULT_MAX_PAUSE_SEC,
        help=(
            f"Cap internal pauses at this many seconds "
            f"(default: {DEFAULT_MAX_PAUSE_SEC}). Applies to both engines. "
            f"Use 0 to keep the engine's own timing."
        ),
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_CHATTERBOX_MAX_CHARS,
        help=(
            f"Chatterbox: max characters per synthesized chunk "
            f"(default: {DEFAULT_CHATTERBOX_MAX_CHARS}). Longer scripts are split "
            f"at paragraph/sentence boundaries to avoid garbled output. "
            f"Set to 0 to disable chunking."
        ),
    )

    args = ap.parse_args()

    # Read script from file or stdin
    if args.script is None or args.script == "-":
        text = sys.stdin.read()
    else:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"error: script file {script_path} not found", file=sys.stderr)
            return 1
        text = script_path.read_text()

    if not text.strip():
        print("error: script is empty", file=sys.stderr)
        return 1

    word_count = len(text.split())
    if args.engine == "kokoro":
        print(
            f"[info] synthesizing ~{word_count} words via Kokoro "
            f"voice={args.voice} speed={args.speed}...",
            file=sys.stderr,
            flush=True,
        )
    else:
        ref_tag = (
            f"cloning={args.reference_audio.name}"
            if args.reference_audio
            else "voice=default"
        )
        chunk_tag = (
            f"max_chars={args.max_chars}" if args.max_chars else "chunking=off"
        )
        print(
            f"[info] synthesizing ~{word_count} words ({len(text.strip())} chars) "
            f"via Chatterbox {ref_tag} exaggeration={args.exaggeration} "
            f"cfg={args.cfg_weight} {chunk_tag}...",
            file=sys.stderr,
            flush=True,
        )

    t0 = time.perf_counter()
    try:
        out_path = generate_voice(
            text=text,
            output_path=args.out,
            engine=args.engine,
            voice=args.voice,
            speed=args.speed,
            reference_audio=args.reference_audio,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            device=args.device,
            max_chars_per_chunk=args.max_chars,
            max_pause=args.max_pause,
            target_lufs=args.target_lufs or None,
        )
    except ImportError as e:
        print(
            f"error: missing dependency: {e}\n"
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2
    except FileNotFoundError as e:
        print(
            f"error: {e}\n"
            "If this mentions espeak-ng (Kokoro), run: brew install espeak-ng",
            file=sys.stderr,
        )
        return 3
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    elapsed = time.perf_counter() - t0
    print(f"[info] wrote {out_path} in {elapsed:.2f}s", file=sys.stderr, flush=True)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
