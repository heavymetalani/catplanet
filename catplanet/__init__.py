"""
catplanet -- nature-documentary narration for house cats.

The stages, in the order the pipeline runs them:

    describe_video   stage 1   video -> description         (MiniCPM-V 4.5)
    generate_script  stage 2   description -> narration     (Qwen2.5)
    generate_music   stage 2b  description -> music caption (Qwen2.5)
    generate_voice   stage 3   narration -> voiceover .wav  (Kokoro / Chatterbox)
    generate_music   stage 4   caption -> string quartet    (ACE-Step 1.5)

    pipeline         runs the stages above for one clip
    batch            runs the pipeline over every video in data/inbox/

Helpers: paths (every folder in one place), local_settings (your voice and
cats), audio_utils (loudness measurement and normalization).

Day to day you run the two scripts in the project root, `pipeline.py` and
`batch.py`. Each stage can also run on its own, e.g.
`python -m catplanet.describe_video clip.mov`.
"""
