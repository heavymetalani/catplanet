# What's next

These are possible upgrades once the basic pipeline feels good.

- **Mix and mux.** A stage 5 that combines the video, voiceover and music into a finished `.mp4`, with sidechain ducking so the music dips under the narration instead of sitting at a fixed level. This was deliberately left out for now; see [MUSIC_PLAN.md](MUSIC_PLAN.md).
- **A quartet LoKr.** If ACE-Step keeps sneaking non-string instruments into the cues, training a style adapter on real quartet recordings is the durable fix. Section 3 of `MUSIC GEN/instructions.md` has the procedure.
- **Timestamp-aware descriptions.** Change the stage 1 prompt to include timestamps, so stage 2 can write a script with suggested cut points.
- **A native video path.** Replace the Ollama frame-sampling approach with the Python `transformers` library, which exposes MiniCPM-V 4.5's 3D resampler for true video encoding. Setup is slower, but the model can perceive motion more smoothly.
