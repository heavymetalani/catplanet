# ACE‑Step 1.5 — Prompting & LoRA Playbook
### Mac Mini M4 · 24 GB · romanized Hindi vocals · reference‑song style · Gradio GUI

Tuned to one machine: a **Mac Mini M4 with 24 GB unified memory**. Every memory, model, and parameter choice below is the committed pick for *this* device — no "it depends." Your 24 GB is shared with macOS, so treat your effective budget as ~16–18 GB and lean on the memory‑saving defaults baked into this guide.

---

## 0. The 60‑second version

1. **Launch with the macOS script** (`start_gradio_ui_macos.sh`) — it auto‑selects the MLX backend for Apple Silicon. Use the **`acestep-v15-turbo`** DiT.
2. **Train the style with LoKr, not LoRA.** LoRA proper will almost certainly OOM at 24 GB; LoKr is ~10× lighter/faster and the realistic path. **Batch size = 1**, gradient checkpointing ON.
3. **At inference, two rules dominate:** set **INT8 Quantization → None** (LoRA can't load on a quantized model), and set **Vocal Language → Hindi** (romanized lyrics otherwise sing with English phonetics). Start the planner at **0.6B** (the 1.7B "trap" is real with quant off on a shared‑memory Mac).
4. **Work the loop:** batch 2–4, fix the seed only when tuning, Repaint bad sections, Auto‑Score to screen Hindi lyric alignment.

Expect everything to run **slower than the CUDA numbers in the official docs** (their "5‑minute LoKr" and "seconds‑per‑song" figures are from a 3090/A100). On the M4 it's workable, just budget patience.

---

## 1. Mental model (30 seconds)

ACE‑Step 1.5 is two brains: a small **LM planner** (writes a blueprint — BPM/key, expanded caption, semantic "composition" codes) and a **Diffusion Transformer** that renders the audio. It's built for **iteration, not one‑click** — seed an idea, then refine with Repaint/Remix. A **style LoKr** is the right tool for "sounds like my references" because it bakes the sonic signature into the model, reusable across every future song.

On your machine: **DiT = `acestep-v15-turbo`** (8 steps, fast, ideal for the try‑listen‑adjust loop). **Backend = MLX** (automatic via the macOS script).

---

## 2. One‑time setup (M4)

```bash
# from the cloned repo
chmod +x start_gradio_ui_macos.sh        # first time only
./start_gradio_ui_macos.sh               # auto-sets MLX backend
```

Optional `.env` (survives updates):
```
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
PORT=7860
LANGUAGE=en
```

Add this to your shell before training sessions (lets MPS use more memory; **save work often**, since it removes the safety cap):
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

For the **training run only**, edit the macOS start script so the service doesn't pre‑load and hog memory: set `: "${INIT_SERVICE:=--init_service false}"`.

---

## 3. PART A — Train your style LoKr (the reusable "sounds like my references" model)

### 3.1 Reference set
- **8–20 songs**, sharing the *same sonic world* (production era, vocal texture, instrumentation, mood). Consistency beats quantity — a mixed bag yields a muddy adapter.
- **Trim each track to its most representative ~60–90 s.** Full‑length songs are the #1 OOM trigger on Apple Silicon; short clips are also enough to learn a style.
- *Rights note (not legal advice):* imitating a specific commercial artist sits in a gray area; for personal/wedding use you're fine, but prefer your own or royalty‑free material if you'll ever publish.

### 3.2 Three files per song (same base filename, one folder)
```
dataset/
├── song1.mp3              # mp3/wav/flac/ogg/opus
├── song1.lyrics.txt       # plain lyrics, no timestamps
├── song1.json             # {"caption":"...","bpm":120,"keyscale":"C major","timesignature":"4","language":"hi"}
└── ...
```
- **Lyrics:** transcribe (Whisper / ElevenLabs / Gemini; scripts in `scripts/lora_data_prepare/`), then **manually clean** — strip `[mm:ss.xx]` timestamps, fix errors. For a *style* LoKr it's learning the **sound**, so Devanagari or romanized both work; just keep `language: "hi"`. Structure tags (`[Verse]`, `[Chorus]`) are optional but help it learn shape.
- **BPM & Key:** from **Key‑BPM‑Finder** (vocalremover.org, local) → import the CSV. Don't let the LM guess these — it hallucinates them.
- **Caption (per song):** describe the consistent sonic signature in the *same vocabulary* across all songs (dimension list in §4.3). This is what teaches "the style."

### 3.3 Gradio click‑path (Training Tab)
1. **Init models.** Select **0.6B** LM only if you want it to auto‑caption; otherwise leave LM unselected (lighter).
2. **Scan** the dataset folder (auto‑detects audio, `.lyrics.txt`, `.caption.txt`, `.json`, BPM/Key `.csv`).
3. **Configure:** set a **Dataset Name**; uncheck **All Instrumental**; **Custom Activation Tag** with **Tag Position = Append** (don't *Replace* the caption); **Genre Ratio = 0**. *(Activation tags have limited effect — don't rely on a trigger word; the style is learned broadly.)*
4. **Auto‑Label** any songs missing captions (skip if you wrote your own; use **Skip Metas** if BPM/key already supplied).
5. **Preview & Edit** row‑by‑row → **Save Changes** each time. Confirm `language = Hindi` everywhere.
6. **Save Dataset** → JSON.
7. **Preprocess → tensors.** **Use tiled encode** to cap peak memory. Then **quit and relaunch Gradio to free memory** before training (relaunch **without** the LM selected).

### 3.4 Train (LoKr tab) — committed settings for 24 GB
Switch to the **Train LoKr** tab, load the tensor directory.

| Setting | Value | Why (M4‑specific) |
|---|---|---|
| Adapter | **LoKr** (LyCORIS) | LoRA proper OOMs at 24 GB; LoKr is ~10× lighter |
| `lokr_linear_dim` | 64 | plenty for a style |
| `lokr_linear_alpha` | 128 | ~2× dim |
| `lokr_factor` | −1 (auto) | let it choose |
| `lokr_weight_decompose` | true (DoRA) | default, better quality |
| Learning rate | 0.03 | LoKr default |
| **Max epochs** | **~800** (10–20 songs) | small set needs more passes |
| **Batch size** | **1** | the failure cases all used batch ≥4 — don't |
| Gradient accumulation | 4 | simulates batch 4 at **no extra memory** |
| **Gradient checkpointing** | **ON** | trades a little speed for headroom — keep on |
| Shift | 3.0 | matches the turbo target |
| Save every N epochs | 50–100 | gives you checkpoints to A/B |
| Seed | 42 | reproducibility |

**Start Training**, watch the loss curve, **Export** the adapter when done. Keep a couple of mid‑run checkpoints so you can pick the one that captures the style without overcooking it. After training, **relaunch Gradio** before generating.

### 3.5 If it still OOMs (narrow fallback, same machine)
LoKr at batch 1 with checkpointing should fit, but MPS memory is finicky. If it won't complete:
- Shorten clips further (≤45 s), drop `lokr_linear_dim` to 32, confirm **batch 1 / grad‑accum 4**, ensure the watermark env var is set.
- Switch to **Side‑Step** (CLI trainer in the repo): pick its built‑in **`vram_16gb` ("Standard 16–24 GB")** preset — it adds 8‑bit optimizer + encoder offloading designed for exactly this band.
- Last resort: preprocess locally, then run the ~minutes‑long *training step* on a rented cloud GPU and bring the small adapter file back — inference stays on your Mac.

---

## 4. PART B — Generate with your LoKr

### 4.1 Load it (Service Configuration) — committed for 24 GB
1. **INT8 Quantization → None.** Mandatory: adapters won't load on a quantized model.
2. **5Hz LM → `0.6B`.** On a shared‑memory 24 GB Mac with quant off and an adapter loaded, the 1.7B planner is a known memory trap. Start at 0.6B; only move to 1.7B if generation runs comfortably.
3. **Offload to CPU → ON.** Cheap insurance on unified memory against OOM (even though the raw tier default would disable it).
4. **DiT = `acestep-v15-turbo`**, backend **MLX**.
5. **LoRA Path** → adapter dir → **Load LoRA** → **Use LoRA = on**. (Use **Unload** to A/B against the base model.)

### 4.2 Mode & planner
- **Custom mode** = full control over Caption + Lyrics + metadata (your main workspace; the **Reference Audio** field also lives here if you ever want a per‑song reference on top of the LoKr).
- **Simple mode** = type one sentence, the LM drafts caption+lyrics+metadata — good for quick exploration, then move the keepers into Custom.
- **`Think` checkbox:** with a style adapter, **on** gives nicer auto‑arrangement; **off** gives tighter control. Try both. (With the 0.6B planner, *off* is also a fine, lighter default once your caption is solid.)

### 4.3 Caption craft
Caption is the most important text input — the music's overall portrait.

| Dimension | Examples |
|---|---|
| Genre/style | pop, R&B, lo‑fi, synthwave, folk, hip‑hop, filmi/Bollywood‑pop |
| Emotion | melancholic, uplifting, dreamy, nostalgic, intimate, euphoric |
| Instruments | acoustic guitar, piano, 808s, tabla, harmonium, strings, synth pads |
| Timbre texture | warm, bright, airy, punchy, lush, raw, polished |
| Era/production | vintage, modern trap, bedroom‑pop, studio‑polished |
| Vocals | female/male vocal, breathy, powerful, falsetto, raspy, harmonies |

- **Specific beats vague.** "intimate Hindi piano ballad with breathy female vocal and soft strings" ≫ "a sad song."
- **With the LoKr loaded, lean *lighter* on style words** — the adapter carries the signature. Use the caption to steer *this song's* mood/instruments.
- **Don't put BPM/key/tempo in the caption** — those go in the metadata fields (§4.5). Caption = musical character only.
- **No clashing fusions.** Want contrast? Write it as *evolution* in the lyrics ("soft intro → anthemic chorus"), not as a contradictory caption.

### 4.4 Romanized (Hinglish) lyrics — the core of your use case
Lyrics are the temporal script: text + structure + performance cues over time.

**Non‑negotiables for romanized Hindi:**
1. **Vocal Language = Hindi.** Latin‑script lyrics default to **English phonetics** unless you pin the language — this single setting is the difference between Hindi singing and gibberish. (Hindi is a "solid tier" language here: competent, with occasional smearing on dense consonant clusters.)
2. **Control schwa via spelling.** Romanization is ambiguous about the inherent /a/ — spell what you want sung: `pyaar` vs `pyaara`, `dil` vs `dilaa`. Double consonants (`sacchaa`) to enforce them.
3. **Fix a mispronounced word two ways:** respell it phonetically (try a couple of romanizations), **or** drop *just that word* into **Devanagari** (mixing scripts in one lyric is supported). Avoid inline IPA — understood but it breaks the singing flow.

**Structure & performance (same in Hinglish):**
- **English structure tags, Hinglish lyrics:** `[Verse 1]`, `[Pre‑Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`. Combine sparingly for delivery (`[Chorus - anthemic]`); **don't stack** many tags or it may sing them.
- **UPPERCASE = louder/more intense** (a romanization advantage Devanagari can't express).
- **Repeat vowels to elongate** (`o jaaaanaa`) — sparingly.
- **(Parentheses) = backing vocals / harmonies.**
- **Blank line between sections.** Keep **6–10 syllables per line**; keep parallel lines (e.g., first line of each verse) at similar counts so they align to the beat.
- **Keep Caption ↔ Lyrics consistent** (named instruments ↔ instrumental tags; emotion ↔ energy tags). The model is bad at contradictions.
- **Lean on your lyrics' flexibility:** when a line's syllable count fights the groove, rephrase rather than force. One core metaphor per song; avoid "AI‑flavored" imagery stacking and inconsistent rhyme.

**Mini example:**
> **Caption:** `intimate Hindi indie‑pop, breathy female vocal, fingerpicked acoustic guitar, warm soft strings, nostalgic, building to a gentle anthemic chorus`
> **Vocal Language:** Hindi · **Instrumental:** off

```
[Intro - acoustic guitar]

[Verse 1]
sham dhale yeh sheher chup hai
teri yaadon mein khoya hua
har ek pal tujhe dhoondta hai
yeh dil mera thoda sa khoya hua

[Pre-Chorus]
saansein ruki si hain
par andar toofan hai

[Chorus - anthemic]
chal jal uthein HUM
jaise raat ka taara
pal bhar ka hai par
yeh lamha hamaara

[Bridge - whispered]
kal jo bikhar jaaye sab
hum to chamke the kabhi
```

### 4.5 Metadata (optional)
Leave blank to let the LM infer (recommended), or set:

| Field | Notes |
|---|---|
| BPM | 30–300; slow 60–80, mid 90–120, fast 130–180 (extremes unstable) |
| Key Scale | `C major`, `Am`, `F# minor` (common keys most stable) |
| Time Signature | `4` (4/4) most reliable; `3`, `6` usually fine |
| Duration | 30–60 s and 2–4 min most stable; up to 600 s supported |

Anchors, not commands (ask 120, get 118–122). Don't let a manual value fight the caption.

---

## 5. The iteration loop

- **Fix the seed only when tuning** a parameter (uncheck Random Seed, enter a number) to hear that parameter's real effect; randomize seeds when exploring.
- **Batch 2–4** (skip 8 on this machine). **AutoGen** keeps new batches generating while you audition.
- **Auto‑Score** triages which takes have the cleanest Hindi lyric placement before you listen closely — useful given Hindi's occasional smearing.
- **Patch, don't restart:** **Repaint** the bad 3–90 s window (or fix a mispronounced word there); **Remix** for a same‑melody restyle; **Send To Src** to build on a take; **Apply These Settings to UI** to lock a winner.
- **Tune one thing at a time:** keep **Steps = 8** and **Shift = 3.0** for turbo; nudge **LM Temperature** (0.85 default; lower = safer); try **`sde`** for variation vs **`ode`** for determinism.

---

## 6. Copy‑paste recipes

**Recipe 1 — First test of the fresh LoKr**
> Custom · Quant **None** · LM **0.6B** · Offload **on** · Use LoRA **on** · Vocal Language **Hindi** · Think on · Batch 4 · Steps 8 · Shift 3.0 · Seed random
> Caption: *2–3 mood/instrument words only — let the LoKr carry the style.*
> Lyrics: one verse + one chorus, tagged, ~8 syllables/line. → Audition the 4, pick closest, refine.

**Recipe 2 — Dialed‑in Hinglish ballad**
> Caption: `intimate Hindi [your‑style] ballad, breathy female vocal, [your signature instruments], warm, building chorus`
> BPM 72 · Key `Am` · Duration 150 · Think on · Batch 4 · Auto‑Score on → screen by score, audition top 2, Repaint rough lines.

**Recipe 3 — Fix a mispronounced word**
> Best take → **Send To Src** → **Repaint** the 3–10 s window → respell the word (or put it in Devanagari) → regenerate just that window.

---

## 7. Quick troubleshooting (M4)

| Symptom | Fix |
|---|---|
| LoRA/LoKr won't load | Set **INT8 Quantization → None**, then load. |
| Lyrics sung with English accent | **Set Vocal Language = Hindi.** |
| One word mispronounced | Respell phonetically, or that word in Devanagari; avoid inline IPA. |
| Training OOM / freeze | **LoKr + batch 1 + grad‑accum 4 + checkpointing**; shorten clips; set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`; else Side‑Step `vram_16gb` preset. |
| Inference OOM / very slow | Drop LM to **0.6B**, keep **CPU offload on**, batch ≤4, quant must stay **None** for the adapter. |
| Rhythm/timing off | Even out syllable counts (6–10/line; parallel lines similar). |
| Style too weak | More epochs / more‑consistent reference set; lean less on caption style words. |
| Robotic / smeared vocals | Different seed/take; Repaint the section; simplify dense consonant clusters. |

---

## 8. Sources & further reading
- **Official "Ultimate Guide"** (mental models, caption/lyrics/metadata, hyperparameters): `github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md`
- **LoRA/LoKr Training Tutorial:** `…/docs/en/LoRA_Training_Tutorial.md`
- **Gradio UI Guide:** `…/docs/en/GRADIO_GUIDE.md`
- **GPU Compatibility & macOS install:** `…/docs/en/GPU_COMPATIBILITY.md`, `…/docs/en/INSTALL.md`
- **Apple‑Silicon install notes** (M‑series, MLX, the "1.7B LM trap"): freeaimusictools.com ACE‑Step Apple‑Silicon guide
- **Side‑Step** (CLI training, `vram_16gb` preset, VRAM tools): `github.com/koda-dernet/Side-Step`
- **Apple‑Silicon fork** (native MPS training, if upstream struggles): `github.com/clockworksquirrel/ace-step-apple-silicon`