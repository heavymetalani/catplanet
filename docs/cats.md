# Identifying your cats

Stage 1 reads the `cats` list in `config/local_settings.json` so it knows who is who. That file is not in git, so your cats' names stay on your machine. The template, `config/local_settings.example.json`, shows the format with placeholder cats:

```json
{
  "reference_audio": "data/voices/narrator.wav",
  "cats": [
    {
      "name": "Whiskers",
      "pronouns": "she/her",
      "appearance": "orange tabby with a white chest and white front paws...",
      "personality": "bold and greedy. First to the food bowl..."
    },
    {
      "name": "Shadow",
      "pronouns": "he/him",
      "appearance": "solid grey cat with no markings and green eyes...",
      "personality": "lazy and dignified. Treats every sunbeam as his own..."
    }
  ]
}
```

The vision model is told to use these names based on coat pattern, to tell the cats apart when both are in the same frame, and to fall back to "one of the cats" when it cannot tell (for example a partial view or a bad angle). The names flow through into the description. Stage 2 also gets the names and pronouns directly, and is told to call each cat by name rather than "the creature" or "the cat", so the voiceover addresses your cats by name.

## Personalities

The optional `personality` field is for the script writer, not the vision model. Stage 2 uses it to find the jokes: an anxious cat hiding from the vacuum, or a fearless one inspecting every room, gives the narration a character to play with, and the same traits come back from clip to clip like a running gag. Describe how each cat behaves, what they love and what they avoid, in a sentence or three. Stage 1 never sees this field, so the description stays a plain account of what happened.

## Editing the list

Open `config/local_settings.json`, change the names, pronouns, appearance or personality, and save. No code changes are needed. To add a cat, add another object to the `cats` array. To turn identification off, remove the `cats` list, and stage 1 falls back to a generic prompt.

## Tips for reliable identification

- Describe each coat as concretely as you can: colors, markings, tail length and size. The more distinctive the description, the more reliably the model picks the right name.
- If two cats have a similar build, lead with the single most distinctive difference, for example "white chest and paws" against "no white anywhere".
- Preview the exact prompt the model will see with `python -m catplanet.describe_video --show-prompt`.
- Point at a different file with `--cats-file path/to/other.json`. This is useful if you film a friend's cat. The file uses the same `{"cats": [...]}` shape.
