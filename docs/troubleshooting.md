# Troubleshooting

## First-run sanity check

Put any short cat clip in `data/scratch/` and run:

```bash
python -m catplanet.describe_video data/scratch/<your_clip>.mp4
```

You should see a paragraph describing what the cat is doing, its posture, the surroundings and any notable moments. The first run is slower (15 to 30 seconds) while the model loads into memory. After that, a 10-second clip at 2 frames per second takes 3 to 8 seconds.

## Common problems

**"connection refused" or Ollama not running.** Start it with `brew services start ollama`, or run `ollama serve` in another terminal.

**The model download is slow.** It is 5 GB and only happens once. Let it finish.

**Out of memory.** Close other heavy apps such as Chrome or Xcode. With MiniCPM-V 4.5 quantized to int4 and about 16 frames, the model itself should stay well under 10 GB.

**The frames don't fit in the model's context window.** Either raise `--num-ctx` or send fewer frames with `--max-frames`. Each frame costs about 450 to 500 tokens. Fewer frames is the cheaper fix and usually loses little detail.

**Descriptions are too generic.** Edit the prompt in `catplanet/describe_video.py`. The more specific you are about what to look for (tail position, pupil size, interactions with objects), the more useful the description is for script-writing.

**`espeak-ng` not found.** Kokoro needs it to generate phonemes. Install it with `brew install espeak-ng` and try again.

**Kokoro's first run is slow or says "Downloading…".** The 82M model weights (about 300 MB) are downloaded from Hugging Face the first time and cached after that.

**The voice sounds robotic or clipped.** Try `--speed 0.95` or a different voice preset. Kokoro sometimes trips on long run-on sentences; shorter, punchier scripts sound more natural.

**Chatterbox MPS error, or "Metal op not implemented".** This is a known issue with some torch operations on Apple Silicon. Use `--device cpu`, which is slower but reliable.

**The cloned voice doesn't sound like the reference.** Make sure the clip is clean (no music, no overlapping speech), 5 to 10 seconds long, and spoken at a natural pace. Very short (under 3 seconds) or noisy clips clone poorly. Try raising `--cfg-weight` to 0.6 for tighter adherence. If the clip fails to load at all, convert it to `.wav` with `ffmpeg -i in.m4a out.wav`.

**Chatterbox output turns into gibberish on longer scripts.** The script is longer than Chatterbox can handle in one go. The pipeline already splits it into chunks (default `--max-chars 250`); if chunks still garble, lower it to 180. To rule out the engine entirely, switch to `--engine kokoro` for a quick check. See [voice.md](voice.md) for details.

**A clip failed in the middle of a batch.** Check the batch summary or `python batch.py --dry-run` to see where it stopped, fix the cause (often the music server), and run `python batch.py` again. It resumes from that stage. Use `python batch.py --fresh` if you would rather start the clip over. See [batch.md](batch.md#retries-pick-up-where-they-stopped).

**You see a file ending in `.partial`.** It is an unfinished file from a run that was interrupted. It is hidden in Finder, and the next run of that clip deletes it.

**`ModuleNotFoundError: No module named 'catplanet'`.** Run commands from the project root (the folder that contains `pipeline.py`). The stage modules must be run as `python -m catplanet.<stage>`, not as `python catplanet/<stage>.py`.

**The music server isn't answering.** Start it in its own terminal with `"MUSIC GEN/start_api_server.command"` and wait for `curl http://127.0.0.1:8001/health` to respond. See [music.md](music.md).
