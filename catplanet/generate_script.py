#!/usr/bin/env python3
"""
generate_script.py

Stage 2 of the cat-video pipeline. Takes a detailed description of what
is happening in a clip (produced by describe_video.py) and writes a
funny short-form voiceover script in the requested style.

It works like a writers' room: first it writes down the facts (the main
action the script must cover, and what happens in order), then pitches
several comic angles, writes one draft per angle, checks every draft
against the facts and keeps the funniest one that passed (--drafts sets
how many; 1 skips the judging). Moments listed under "priority_moments" in
config/local_settings.json become the main action when a clip shows them. The script
opens with a short title for the clip, which the narrator reads first
(--no-title leaves it out). Your cats' names, pronouns and personalities
come from config/local_settings.json, the same list stage 1 uses.

Usage
-----
    # From a file
    python -m catplanet.generate_script --description data/scratch/zoomies.description.txt --length 15

    # From stdin (chain with stage 1)
    python -m catplanet.describe_video data/scratch/zoomies.mov \\
        | python -m catplanet.generate_script --length 15

    # Pick a style
    python -m catplanet.generate_script --description d.txt --length 20 --style noir

Styles
------
    attenborough  - deadpan nature-documentary narrator (default)
    sports        - excitable play-by-play sportscaster
    noir          - hardboiled 1940s detective voiceover
    reality_tv    - dramatic reality-TV confessional
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import ollama

from . import describe_video, local_settings

DEFAULT_MODEL = "qwen2.5:14b"

STYLES: dict[str, str] = {
    "attenborough": """You are a comedy writer producing a short-form video voiceover in the style
of a David Attenborough nature documentary — but applied to domestic cats,
with a deadpan, absurdly dramatic tone.

Voice & tone:
  - Dry, understated, documentary-narrator delivery.
  - Treat mundane cat behavior with epic seriousness.
  - Sprinkle in one or two absurd "scientific" observations per clip
    (fake species names, invented evolutionary explanations, etc.).
  - Never break character. Never explain the joke.

Cadence — this is critical, the output will be read aloud by a TTS engine
that follows punctuation literally:
  - Use SHORT declarative sentences. Fragments are good. No run-ons.
  - Use an ellipsis ("...") for a dramatic pause of about one beat.
    Place these BEFORE key revelations or punchlines.
  - Use an em dash (" — ") for a shorter mid-thought pause.
  - End sentences with a period so the narrator breathes.
  - Break into 2-4 short paragraphs for a clip of ~15-30 seconds;
    each paragraph break adds a longer breath.
  - Prefer concrete nouns and specific actions over adjectives.

Vocabulary Attenborough actually uses: "here we see...", "observe...",
"in an astonishing display of...", "and yet...", "having assessed...",
"what appears to be...".

Example beat structure (for a 15-second clip). [NAME] stands for whichever
cat the description names. It shows rhythm only: write your own sentences,
and never reuse any of its lines, including the last one.

    Here... we observe [NAME]. A creature of habit.

    She has chosen, once again — against all reason — to occupy
    the empty box beside the couch.

    Evolution... offers no explanation.""",

    "sports": """You are a high-energy sports play-by-play announcer calling a dramatic
moment live. Your subject: a house cat doing ordinary cat things.

Style rules:
  - Fast, breathless, building tension and payoff.
  - Use sports cliches ("and here comes the crossover", "textbook form", "he's gonna do it").
  - Treat every cat action as a decisive championship moment.
  - Short beats. Lots of exclamation.
  - Never break character.""",

    "noir": """You are a hardboiled 1940s film-noir detective narrating a case. The case
happens to be a house cat doing cat things, but you deliver it like the
city is on the line.

Style rules:
  - First person, weary, cigarette-in-voice.
  - Rain-soaked metaphors. Distrust everyone, including the cat.
  - Short sentences. Heavy pauses implied by periods.
  - Never break character.""",

    "reality_tv": """You are a reality-TV confessional writer. The "cast member" is a house cat,
and every small action is a betrayal, a power play, or a personal journey.

Style rules:
  - Petty, dramatic, over-invested in minor events.
  - Mix confessional voice ("I knew. I just KNEW.") with narrator beats.
  - Short, clipped sentences for punch.
  - Never break character.""",
}

# Stage 2 runs in four steps, the way a comedy writers' room does:
#   1. beats    - a careful, low-temperature call writes down the facts: the
#                 main action the script must cover, and what happens in order
#   2. pitch    - one call pitches a different comic angle per draft
#   3. draft    - one draft of the narration per angle, bound to the facts
#   4. judge    - one call checks every draft against the facts and ranks them
#                 by funniness; the code keeps the funniest draft that passed
# If no draft passes, one more "faithful" draft is written and checked.
# With drafts=1 there is one angle, one draft and no judging.
DEFAULT_DRAFTS = 3

# Ollama's default context can be as small as 4096 tokens. A long stage-1
# description is ~3000 tokens on its own, and the judge also reads every
# draft, so ask for room explicitly rather than have the prompt cut off.
DEFAULT_NUM_CTX = 8192

# The rule every step repeats, because "make it funnier" pulls the model
# towards inventing bigger events.
NO_UPGRADES = ("Never upgrade what a cat does into a bigger stunt: lying on her back is "
               "not a backflip, a glance is not a leap, a walk is not a sprint.")

BEATS_SYSTEM_PROMPT = """You are the script supervisor on a show that narrates house-cat videos.
You write down what actually happens in a clip, plainly and accurately,
so the writers can't drift from it. You add nothing."""

BEATS_PROMPT_TEMPLATE = """{cast_block}
What happens in this {length}-second clip:
--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---
{priority_block}
Write the beat sheet for this clip:
  - "beats": 3 to 6 short sentences, in the order they happen, each saying who does what, using the cats' names. Keep each action exactly as the description gives it. {no_upgrades} Leave out guesses ("possibly", "seems", "perhaps").
  - "priority": {priority_rule}
  - "main_action": one sentence naming the event the narration must be about. {main_action_rule}

Reply as JSON: {{"beats": ["...", "..."], "priority": 0, "main_action": "..."}}"""

# Inserted when config/local_settings.json lists priority moments.
PRIORITY_BLOCK_TEMPLATE = """
Moments that do best with our audience:
{moments}
"""
PRIORITY_RULE = ("the number of the moment in that list that this clip clearly shows, or 0 if "
                 "none does. Count a moment only if the description plainly describes it. If "
                 "the clip shows more than one, give the lowest number.")
MAIN_ACTION_RULE_PRIORITY = ("If the clip shows one of the listed moments, the main action is "
                             "that moment; otherwise it is the biggest or funniest thing that happens.")
MAIN_ACTION_RULE_PLAIN = "It is the biggest or funniest thing that happens."

# The facts every later step is bound to.
FACTS_BLOCK_TEMPLATE = """
--- FACTS ---
Main action: {main_action}
What happens, in order:
{beats}
--- END FACTS ---
"""

PITCH_SYSTEM_PROMPT = """You are a comedy writer in the writers' room of a show that narrates
house-cat videos. You don't write the narration yourself: you pitch the
joke that a script will be built around."""

# One comic lens per angle, so the drafts are different jokes rather than
# three takes on the same one. More drafts than lenses cycles round them.
COMIC_LENSES = [
    "Raise the stakes: treat a small moment as a matter of life and death, history or science.",
    "Character comedy: play the cats' personalities against each other or against the situation.",
    "Misdirection: build the moment up as something grand, then deflate it at the end with the narrator's commentary on what really happened. The twist is in the words, not a new event.",
]

PITCH_PROMPT_TEMPLATE = """The style of the show:
--- STYLE ---
{style_prompt}
--- END STYLE ---
{cast_block}
What happens in this {length}-second clip:
--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---
{facts_block}
Pitch {n} comic angles for this clip, one for each lens below, in this order:
{lenses}

Each angle is two or three sentences pitched to the other writers, not narration: for example "<cat> thinks they are ..., but ...". Say who the hero is, what they want or what is at stake, and the punchline the narration builds to.
  - Build every angle on the main action.{facts_rule}
  - The joke is in how the narrator frames what really happens. A joke can exaggerate why the cats do things, never what they do: don't invent actions or events. {no_upgrades}
  - Use the cats' personalities where the action fits them.
  - Don't borrow lines from the style's example; every punchline should be new.

Reply as JSON: {{"angles": ["...", "..."]}}"""

USER_PROMPT_TEMPLATE = """Below is a detailed description of a {length}-second short-form cat video.
{cast_block}
--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---
{facts_block}{angle_block}
Write the voiceover narration for this clip.

Constraints:
  - Target length: approximately {word_count} words (matches roughly {length} seconds at a lively short-form pace).
  - Output ONLY the spoken narration words. No stage directions, no bracketed cues, no timestamps, no section headers, no explanation.
  - Call each cat by the name the description gives them, and use their pronouns. Say each named cat's name at least once. Never call a named cat "the creature", "this cat" or "the feline".
  - If the description only says "one of the cats", leave that cat unnamed rather than guessing.
{facts_constraints}  - Let the cats' personalities drive the jokes where the action fits. Show them through what the cat does; never list traits.
  - Exaggerate why the cats do things, never what they do: don't invent actions or events. {no_upgrades}
  - Keep the voice, jokes, cadence and punctuation your instructions ask for. Don't just retell the description.
{title_rule}
Begin."""

# Draft constraints that only make sense once there is a beat sheet.
FACTS_CONSTRAINTS = """  - Say the main action plainly at least once, in words a viewer can match to the clip (if they wrestle, say they wrestle). The narration is about the main action.
  - Narrate only actions that are in the facts, in their order.
"""
# Without a beat sheet, fall back to the description's own order.
NO_FACTS_CONSTRAINTS = """  - If the description mentions multiple distinct moments, weave the narration to hit them in order.
"""

# Inserted into USER_PROMPT_TEMPLATE when the draft has a pitched angle.
ANGLE_BLOCK_TEMPLATE = """
--- COMIC ANGLE ---
{angle}
--- END COMIC ANGLE ---

Build the narration around this comic angle, and end on its punchline. Use the description for what happens, not for wording.
"""

# Used instead of an angle for the fallback draft, when every angled draft
# failed the judge's checks.
FAITHFUL_BLOCK = """
Stick closely to the facts above. Find the joke in the narrator's commentary on what really happens, not in anything new.
"""

# Added to the constraints when a title is wanted. The title is spoken first,
# so it is written in the narrator's voice, not as a caption.
TITLE_RULE = """  - Start with a title for the clip on its own line, in exactly this form:
        TITLE: <a short, funny title of 2 to 6 words, in the narrator's voice>
    Invent the title yourself, like the name of an episode in your style; don't copy a heading from the description.
    Then a blank line, then the narration. The title is read aloud before the narration.
"""

# Words held back from the narration target for the spoken title, so the
# voiceover still fits the clip.
TITLE_WORD_ALLOWANCE = 5

JUDGE_SYSTEM_PROMPT = """You are the head writer of a comedy show that narrates house-cat videos.
You check the drafts your writers hand in against what really happens in
the clip, then rank them by which will get the biggest laugh when read
aloud over it."""

JUDGE_PROMPT_TEMPLATE = """The style for this clip:
--- STYLE ---
{style_prompt}
--- END STYLE ---
{cast_block}
What happens in the {length}-second clip:
--- DESCRIPTION ---
{description}
--- END DESCRIPTION ---
{facts_block}
The drafts:

{drafts}

First, check every draft against {reference}:
  - "shows_main_action": true if the draft plainly says the main action happens, false if it skips it or only hints at it.
  - "invented": every physical action or event the draft says happens that is not in {reference}, including an action upgraded into a bigger stunt (a cat lying on her back described as a backflip). Only list things a viewer could check against the video. Never list the title, or what a cat thinks, decides, wants or plans, or the narrator's commentary, comparisons and made-up history: "the cat decides the blanket is her headquarters" is commentary, not an invented action. Empty if there are none.

Then rank all the drafts from funniest to least funny, judging:
  1. Is it actually funny, with a clear joke and a payoff, rather than a retelling of the description?
  2. Does it stay in the style's voice, call the cats by their names, and run roughly {word_count} words?
  3. Does it make up its own punchline, rather than reusing a line from the style's example?

Reply as JSON: {{"checks": [{{"draft": 1, "shows_main_action": true, "invented": []}}], "ranking": [1, 2, 3], "reason": "<one or two sentences on the funniest draft that passed the checks>"}}"""


@dataclass
class BeatSheet:
    """The facts every draft is bound to."""
    main_action: str
    beats: list[str]
    priority: str | None = None     # the priority moment the clip shows, if any

    def block(self) -> str:
        return FACTS_BLOCK_TEMPLATE.format(
            main_action=self.main_action,
            beats="\n".join(f"  - {b}" for b in self.beats),
        )


@dataclass
class DraftCheck:
    """The judge's check of one draft against the facts."""
    shows_main_action: bool
    invented: list[str]

    @property
    def passed(self) -> bool:
        return self.shows_main_action and not self.invented

    def summary(self) -> str:
        parts = ["shows the main action" if self.shows_main_action
                 else "SKIPS the main action"]
        parts.append("invents: " + "; ".join(self.invented) if self.invented
                     else "invents nothing")
        return ("passed" if self.passed else "failed") + " - " + ", ".join(parts)


@dataclass
class ScriptResult:
    """Stage 2's output: the chosen script, plus how it was chosen."""
    script: str
    angles: list[str | None]
    drafts: list[str]
    winner: int          # index into drafts
    reason: str
    beats: BeatSheet | None = None
    checks: list[DraftCheck | None] = field(default_factory=list)
    ranking: list[int] = field(default_factory=list)   # indexes, funniest first

    def report(self) -> str:
        """The facts, every angle and draft, the checks and the pick, for
        <name>.script_drafts.txt."""
        parts = []
        if self.beats:
            parts.append("=== Facts ===")
            if self.beats.priority:
                parts.append(f"Priority moment: {self.beats.priority}")
            parts.append(self.beats.block().strip()
                         .removeprefix("--- FACTS ---").removesuffix("--- END FACTS ---")
                         .strip() + "\n")
        for i, (angle, draft) in enumerate(zip(self.angles, self.drafts)):
            mark = "   <- picked" if i == self.winner else ""
            label = "Draft" if angle is not FAITHFUL else "Faithful draft"
            parts.append(f"=== {label} {i + 1}{mark} ===")
            if angle and angle is not FAITHFUL:
                parts.append(f"Angle: {angle}\n")
            parts.append(draft + "\n")
            check = self.checks[i] if i < len(self.checks) else None
            if check:
                parts.append(f"Check: {check.summary()}\n")
        if self.ranking:
            parts.append("Funniest first: " + ", ".join(str(i + 1) for i in self.ranking) + "\n")
        if len(self.drafts) > 1:
            parts.append(f"=== Why draft {self.winner + 1} ===\n{self.reason}\n")
        return "\n".join(parts)


# Marks the fallback draft in ScriptResult.angles.
FAITHFUL = "(faithful draft: no angle, facts only)"


def build_cast_block(cats: list[dict] | None = None) -> str:
    """Names, pronouns and personalities of the cats, from config/local_settings.json.

    The description already uses the names, but listing them here as well keeps
    the pronouns right and stops the writer drifting back to "the creature".
    The optional "personality" field gives the writer something to joke about;
    stage 1 never sees it, so the description stays a plain account of the
    clip. Empty string if no cats are configured.
    """
    if cats is None:
        cats = describe_video.load_cats()
    if not cats:
        return ""
    lines = ["", "The cats in this household (the description uses these names):"]
    for cat in cats:
        line = f"  - {cat['name']} ({cat['pronouns']})"
        if cat.get("personality"):
            line += f": {cat['personality']}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def load_priority_moments() -> list[str]:
    """The moments your audience likes best, from config/local_settings.json.

    When a clip shows one of them, it becomes the main action the script must
    cover. Earlier entries win when a clip shows several.
    """
    moments = local_settings.get("priority_moments", [])
    if not isinstance(moments, list):
        return []
    return [m.strip() for m in moments if isinstance(m, str) and m.strip()]


def prompt_signature(style: str, title: bool = True, drafts: int = DEFAULT_DRAFTS) -> str:
    """Everything in the prompts except the description, for the pipeline's
    settings fingerprint: editing a prompt, your cats or your priority
    moments redoes the script."""
    return "\n".join([
        build_system_prompt(style),
        NO_UPGRADES,
        BEATS_SYSTEM_PROMPT,
        BEATS_PROMPT_TEMPLATE,
        PRIORITY_BLOCK_TEMPLATE,
        PRIORITY_RULE,
        MAIN_ACTION_RULE_PRIORITY,
        MAIN_ACTION_RULE_PLAIN,
        *load_priority_moments(),
        FACTS_BLOCK_TEMPLATE,
        PITCH_SYSTEM_PROMPT,
        PITCH_PROMPT_TEMPLATE,
        *COMIC_LENSES,
        *EXAMPLE_LINES.get(style, []),
        USER_PROMPT_TEMPLATE,
        FACTS_CONSTRAINTS,
        NO_FACTS_CONSTRAINTS,
        ANGLE_BLOCK_TEMPLATE,
        FAITHFUL_BLOCK,
        TITLE_RULE if title else "",
        JUDGE_SYSTEM_PROMPT if drafts > 1 else "",
        JUDGE_PROMPT_TEMPLATE if drafts > 1 else "",
        build_cast_block(),
        str(drafts),
    ])


_TITLE_LINE = re.compile(r"^[\s*#_]*title[\s*_]*:\s*(.+)$", re.IGNORECASE)
_LABEL_LINE = re.compile(r"^[\s*#_]*(narration|voiceover|script)[\s*_]*:\s*$", re.IGNORECASE)


def split_title(text: str) -> tuple[str | None, str]:
    """Pull the "TITLE: ..." line off the model's output.

    Returns (title, narration). The title is None if the model left it out.
    """
    lines = text.strip().splitlines()
    title = None
    if lines:
        m = _TITLE_LINE.match(lines[0])
        if m:
            title = m.group(1).strip().strip('*_"\'“”‘’ ').rstrip(".").strip()
            lines = lines[1:]
    # Drop blank lines and a stray "Narration:" label before the narration.
    while lines and (not lines[0].strip() or _LABEL_LINE.match(lines[0])):
        lines = lines[1:]
    return (title or None), "\n".join(lines).strip()


def build_system_prompt(style: str) -> str:
    if style not in STYLES:
        raise ValueError(
            f"Unknown style '{style}'. Choose from: {', '.join(STYLES)}"
        )
    return STYLES[style]


def _ask_json(model: str, system: str, user: str, schema: dict,
              temperature: float, num_ctx: int) -> dict:
    """One chat call constrained to `schema`. Raises ValueError on a bad reply."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema,
        options={"temperature": temperature, "top_p": 0.95, "num_ctx": num_ctx},
    )
    try:
        data = json.loads(response["message"]["content"])
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return data


def write_beats(
    description: str,
    video_length: int,
    model: str,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> BeatSheet | None:
    """Step 1: write down the facts. Returns None if the reply is unusable twice."""
    schema = {
        "type": "object",
        "properties": {
            "beats": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "integer"},
            "main_action": {"type": "string"},
        },
        "required": ["beats", "priority", "main_action"],
    }
    moments = load_priority_moments()
    priority_block = PRIORITY_BLOCK_TEMPLATE.format(
        moments="\n".join(f"  {i + 1}. {m}" for i, m in enumerate(moments))
    ) if moments else ""
    user_prompt = BEATS_PROMPT_TEMPLATE.format(
        cast_block=build_cast_block(),
        length=video_length,
        description=description.strip(),
        priority_block=priority_block,
        no_upgrades=NO_UPGRADES,
        priority_rule=PRIORITY_RULE if moments else "always 0.",
        main_action_rule=MAIN_ACTION_RULE_PRIORITY if moments else MAIN_ACTION_RULE_PLAIN,
    )
    for attempt in (1, 2):
        try:
            # Low temperature: this is bookkeeping, not comedy.
            data = _ask_json(model, BEATS_SYSTEM_PROMPT, user_prompt, schema, 0.2, num_ctx)
            beats = [b.strip() for b in data.get("beats", [])
                     if isinstance(b, str) and b.strip()]
            main_action = str(data.get("main_action", "")).strip()
            if not beats or not main_action:
                raise ValueError("no beats or no main action")
            number = int(data.get("priority") or 0)
            priority = moments[number - 1] if 1 <= number <= len(moments) else None
            return BeatSheet(main_action, beats, priority)
        except (ValueError, TypeError) as e:
            print(f"[warn] beat sheet attempt {attempt} was unusable ({e}).",
                  file=sys.stderr)
    print("[warn] drafting from the description alone, without a beat sheet.",
          file=sys.stderr)
    return None


def pitch_angles(
    description: str,
    video_length: int,
    style: str,
    model: str,
    n: int,
    temperature: float = 0.9,
    num_ctx: int = DEFAULT_NUM_CTX,
    beats: BeatSheet | None = None,
) -> list[str]:
    """Step 2: pitch `n` different comic angles. Returns [] if the reply is unusable."""
    schema = {
        "type": "object",
        "properties": {"angles": {"type": "array", "items": {"type": "string"}}},
        "required": ["angles"],
    }
    lenses = "\n".join(f"  {i + 1}. {COMIC_LENSES[i % len(COMIC_LENSES)]}"
                       for i in range(n))
    user_prompt = PITCH_PROMPT_TEMPLATE.format(
        style_prompt=build_system_prompt(style),
        length=video_length,
        description=description.strip(),
        cast_block=build_cast_block(),
        facts_block=beats.block() if beats else "",
        facts_rule=" Stick to the facts." if beats else "",
        no_upgrades=NO_UPGRADES,
        n=n,
        lenses=lenses,
    )
    try:
        data = _ask_json(model, PITCH_SYSTEM_PROMPT, user_prompt, schema,
                         temperature, num_ctx)
    except ValueError as e:
        print(f"[warn] the angle pitch was unusable ({e}); drafting without angles.",
              file=sys.stderr)
        return []
    angles = [a.strip() for a in data.get("angles", [])
              if isinstance(a, str) and a.strip()]
    return angles[:n]


def write_draft(
    description: str,
    video_length: int,
    style: str,
    model: str,
    angle: str | None = None,
    words_per_second: float = 2.5,
    temperature: float = 0.9,
    title: bool = True,
    num_ctx: int = DEFAULT_NUM_CTX,
    beats: BeatSheet | None = None,
) -> str:
    """Step 3: write one draft of the narration, built around `angle` if given.

    Pass angle=FAITHFUL for the fallback draft that sticks to the facts.
    With `title`, the first paragraph is a short spoken title for the clip.
    """
    if angle is FAITHFUL:
        angle_block = FAITHFUL_BLOCK
    elif angle:
        angle_block = ANGLE_BLOCK_TEMPLATE.format(angle=angle)
    else:
        angle_block = ""
    system_prompt = build_system_prompt(style)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        length=video_length,
        description=description.strip(),
        word_count=target_words(video_length, words_per_second, title),
        cast_block=build_cast_block(),
        facts_block=beats.block() if beats else "",
        facts_constraints=FACTS_CONSTRAINTS if beats else NO_FACTS_CONSTRAINTS,
        no_upgrades=NO_UPGRADES,
        angle_block=angle_block,
        title_rule=TITLE_RULE if title else "",
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "top_p": 0.95,
            "num_ctx": num_ctx,
        },
    )
    text = response["message"]["content"].strip()
    if not title:
        return text

    heading, narration = split_title(text)
    if heading is None:
        print("[warn] the model left out the title; using the narration alone.",
              file=sys.stderr)
        return narration
    # A period and a paragraph break give the narrator a breath after the title.
    end = "" if heading[-1] in "!?" else "."
    return f"{heading}{end}\n\n{narration}"


def judge_drafts(
    description: str,
    drafts: list[str],
    video_length: int,
    style: str,
    model: str,
    words_per_second: float = 2.5,
    title: bool = True,
    num_ctx: int = DEFAULT_NUM_CTX,
    beats: BeatSheet | None = None,
) -> tuple[list[DraftCheck | None], list[int], str]:
    """Step 4: check every draft against the facts and rank them by funniness.

    Returns (checks, ranking, reason): one check per draft (None where the
    judge skipped one), draft indexes funniest first, and the judge's reason.
    Raises ValueError if the verdict is unusable.
    """
    schema = {
        "type": "object",
        "properties": {
            "checks": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "draft": {"type": "integer"},
                    "shows_main_action": {"type": "boolean"},
                    "invented": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["draft", "shows_main_action", "invented"],
            }},
            "ranking": {"type": "array", "items": {"type": "integer"}},
            "reason": {"type": "string"},
        },
        "required": ["checks", "ranking", "reason"],
    }
    numbered = "\n\n".join(
        f"--- DRAFT {i + 1} ---\n{d}\n--- END DRAFT {i + 1} ---"
        for i, d in enumerate(drafts)
    )
    user_prompt = JUDGE_PROMPT_TEMPLATE.format(
        style_prompt=build_system_prompt(style),
        cast_block=build_cast_block(),
        length=video_length,
        description=description.strip(),
        facts_block=beats.block() if beats else "",
        reference="the facts" if beats else "the description",
        drafts=numbered,
        word_count=target_words(video_length, words_per_second, title),
    )
    # Low temperature: the verdict should be a judgement, not a dice roll.
    data = _ask_json(model, JUDGE_SYSTEM_PROMPT, user_prompt, schema, 0.2, num_ctx)

    checks: list[DraftCheck | None] = [None] * len(drafts)
    for c in data.get("checks", []):
        try:
            i = int(c["draft"]) - 1
            if 0 <= i < len(drafts) and checks[i] is None:
                invented = [str(x).strip() for x in c.get("invented", []) if str(x).strip()]
                checks[i] = DraftCheck(bool(c["shows_main_action"]), invented)
        except (KeyError, TypeError, ValueError):
            continue
    if all(c is None for c in checks):
        raise ValueError("no usable checks")
    for i, c in enumerate(checks):
        line = reused_example_line(drafts[i], style)
        if c is not None and line:
            c.invented.append(f'reuses the style example\'s line "{line}"')

    ranking: list[int] = []
    for r in data.get("ranking", []):
        try:
            i = int(r) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(drafts) and i not in ranking:
            ranking.append(i)
    # Drafts the judge left out of the ranking go last, in order.
    ranking += [i for i in range(len(drafts)) if i not in ranking]
    return checks, ranking, str(data.get("reason", "")).strip()


# Lines from a style's example that the writer keeps copying. A draft that
# reuses one fails its check, because the example is meant to show rhythm,
# not supply the punchline.
EXAMPLE_LINES: dict[str, list[str]] = {
    "attenborough": ["offers no explanation"],
}


def reused_example_line(text: str, style: str) -> str | None:
    """The example line `text` copies, if any."""
    low = text.lower()
    return next((line for line in EXAMPLE_LINES.get(style, []) if line in low), None)


def _passes(check: DraftCheck | None, beats: BeatSheet | None) -> bool:
    """Whether a draft may win. Without a beat sheet there is no main action
    to check, so only invented actions count against it."""
    if check is None:
        return False
    if beats is None:
        return not check.invented
    return check.passed


def target_words(video_length: int, words_per_second: float, title: bool) -> int:
    """How many words the narration should aim for."""
    word_count = max(12, int(video_length * words_per_second))
    if title:
        word_count = max(12, word_count - TITLE_WORD_ALLOWANCE)
    return word_count


def generate_script(
    description: str,
    video_length: int,
    style: str,
    model: str,
    words_per_second: float = 2.5,
    temperature: float = 0.9,
    title: bool = True,
    drafts: int = DEFAULT_DRAFTS,
    num_ctx: int = DEFAULT_NUM_CTX,
    log=None,
) -> ScriptResult:
    """Write the beat sheet, pitch angles, write `drafts` drafts, and keep the
    funniest draft that sticks to the facts.

    `log`, if given, is called with a short progress line before each call.
    """
    if not description.strip():
        # Without a description the model invents a scene from nothing.
        raise ValueError("the description is empty, so there is nothing to narrate")
    say = log or (lambda msg: None)
    drafts = max(1, drafts)

    say("writing the beat sheet")
    beats = write_beats(description, video_length, model, num_ctx)
    if beats:
        say(f"main action: {beats.main_action}"
            + (f" (priority moment: {beats.priority})" if beats.priority else ""))

    say(f"pitching {drafts} comic angle{'s' if drafts > 1 else ''}")
    angles: list[str | None] = list(pitch_angles(
        description, video_length, style, model, drafts, temperature, num_ctx, beats))
    if angles and len(angles) < drafts:
        print(f"[warn] got {len(angles)} of {drafts} angles; the rest are "
              f"drafted without one.", file=sys.stderr)
    angles += [None] * (drafts - len(angles))

    def draft(angle, temp):
        return write_draft(description, video_length, style, model, angle,
                           words_per_second, temp, title, num_ctx, beats)

    texts = []
    for i, angle in enumerate(angles):
        say(f"writing draft {i + 1} of {drafts}")
        texts.append(draft(angle, temperature))

    if drafts == 1:
        return ScriptResult(texts[0], angles, texts, 0, "Only one draft was written.",
                            beats=beats)

    def judge(candidates):
        return judge_drafts(description, candidates, video_length, style, model,
                            words_per_second, title, num_ctx, beats)

    say(f"checking and ranking {drafts} drafts")
    try:
        checks, ranking, reason = judge(texts)
    except (ValueError, KeyError, TypeError) as e:
        print(f"[warn] the judge's verdict was unusable ({e}); keeping draft 1.",
              file=sys.stderr)
        return ScriptResult(texts[0], angles, texts, 0,
                            "The judge's verdict was unusable, so draft 1 was kept.",
                            beats=beats)

    # The code, not the judge, makes the final call: the funniest draft that
    # passed its checks wins, however much the judge liked a failing one.
    passing = [i for i in ranking if _passes(checks[i], beats)]
    if passing:
        winner = passing[0]
        if winner != ranking[0]:
            reason = (f"Draft {ranking[0] + 1} was ranked funniest but failed its check "
                      f"({checks[ranking[0]].summary() if checks[ranking[0]] else 'not checked'}), "
                      f"so the funniest draft that passed was kept.")
        return ScriptResult(texts[winner], angles, texts, winner, reason,
                            beats, checks, ranking)

    # Every draft failed: write one that sticks to the facts and check it.
    say("every draft failed its check; writing a faithful draft")
    angles.append(FAITHFUL)
    texts.append(draft(FAITHFUL, min(temperature, 0.7)))
    checks.append(None)
    try:
        extra_checks, _, _ = judge([texts[-1]])
        checks[-1] = extra_checks[0]
    except (ValueError, KeyError, TypeError) as e:
        print(f"[warn] the faithful draft's check was unusable ({e}).", file=sys.stderr)
    if _passes(checks[-1], beats):
        winner = len(texts) - 1
        reason = "Every angled draft failed its check, so the faithful draft was kept."
    else:
        winner = ranking[0]
        reason = ("Every draft failed its check, including the faithful one, so the "
                  "draft ranked funniest was kept. Check it against the clip.")
        print(f"[warn] {reason}", file=sys.stderr)
    return ScriptResult(texts[winner], angles, texts, winner, reason,
                        beats, checks, ranking)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a funny voiceover script from a video description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--description",
        type=str,
        default=None,
        help="Path to a file with the description. Omit or pass '-' to read from stdin.",
    )
    ap.add_argument(
        "--length",
        type=int,
        required=True,
        help="Video length in seconds (drives target word count).",
    )
    ap.add_argument(
        "--style",
        choices=sorted(STYLES.keys()),
        default="attenborough",
        help="Voiceover style preset (default: attenborough).",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama text model (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--wps",
        type=float,
        default=2.5,
        help="Target words per second of video (default: 2.5).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9). Higher = wilder.",
    )
    ap.add_argument(
        "--no-title",
        action="store_true",
        help="Leave out the spoken title at the start of the script.",
    )
    ap.add_argument(
        "--drafts",
        type=int,
        default=DEFAULT_DRAFTS,
        help=(f"How many drafts to write before keeping the funniest "
              f"(default: {DEFAULT_DRAFTS}). 1 writes a single draft."),
    )
    ap.add_argument(
        "--drafts-out",
        type=Path,
        default=None,
        help="Also save every angle and draft, and why the winner won, to this file.",
    )
    args = ap.parse_args()

    # Read description from file or stdin
    if args.description is None or args.description == "-":
        description = sys.stdin.read()
    else:
        desc_path = Path(args.description)
        if not desc_path.exists():
            print(f"error: description file {desc_path} not found", file=sys.stderr)
            return 1
        description = desc_path.read_text()

    if not description.strip():
        print("error: description is empty", file=sys.stderr)
        return 1

    print(
        f"[info] generating {args.style} script for {args.length}s clip via {args.model}...",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        result = generate_script(
            description=description,
            video_length=args.length,
            style=args.style,
            model=args.model,
            words_per_second=args.wps,
            temperature=args.temperature,
            title=not args.no_title,
            drafts=args.drafts,
            log=lambda msg: print(f"[info] {msg}...", file=sys.stderr, flush=True),
        )
    except ollama.ResponseError as e:
        msg = str(e)
        if "not found" in msg.lower():
            print(
                f"\nerror: model '{args.model}' is not pulled.\n"
                f"Run: ollama pull {args.model}",
                file=sys.stderr,
            )
        else:
            print(f"ollama error: {msg}", file=sys.stderr)
        return 2
    except ConnectionError:
        print(
            "error: could not connect to Ollama at http://localhost:11434\n"
            "Start it with: brew services start ollama   (or: ollama serve)",
            file=sys.stderr,
        )
        return 3

    elapsed = time.perf_counter() - t0
    print(f"[info] generated in {elapsed:.2f}s", file=sys.stderr, flush=True)
    if len(result.drafts) > 1:
        print(f"[info] kept draft {result.winner + 1} of {len(result.drafts)}: "
              f"{result.reason}", file=sys.stderr)
    if args.drafts_out is not None:
        args.drafts_out.write_text(result.report())
        print(f"[info] saved every draft to {args.drafts_out}", file=sys.stderr)
    print(result.script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
