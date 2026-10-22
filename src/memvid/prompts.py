"""All prompts in one place, so they can be reviewed, versioned and cited in the report.

Every prompt starts with `### TASK: <name>` and is divided into `### <SECTION>` blocks.
The structure keeps prompts readable and lets the deterministic heuristic backend (used in
smoke tests) parse them. PROMPT_VERSION is stored with every cached model output; change it
whenever a template changes, so stale caches are detected.
"""

from __future__ import annotations

PROMPT_VERSION = "1"

CATEGORIES = [
    "person", "tool", "container", "food", "material", "furniture", "device", "vehicle",
    "animal", "clothing", "text", "place", "other",
]


def sections(prompt: str) -> dict[str, str]:
    """Parse '### NAME' sections (used by the heuristic backend)."""
    out: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in prompt.splitlines():
        if line.startswith("### "):
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current, buf = line[4:].strip(), []
        else:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def task_of(prompt: str) -> str | None:
    for name in sections(prompt):
        if name.startswith("TASK:"):
            return name.split(":", 1)[1].strip()
    return None


# ----------------------------------------------------------------------------- perception
PERCEIVE = """### TASK: perceive
You see {n_frames} frames sampled uniformly, in temporal order, from a {duration:.0f}-second segment of a longer video, plus the speech transcribed in that segment. Describe ONLY this segment.

Rules:
- Report only what is visible in the frames or said in the transcript. Do not guess or infer what happens outside this segment.
- entities: each distinct person, object, animal, substance or text that takes part in what happens. "name" is a short noun phrase with distinguishing features ("red mug", "man in blue shirt", "pine board"). "category" is one of: {categories}. "attributes" maps property names to short concrete values, e.g. {{"color": "white", "state": "half painted", "count": "4", "location": "on workbench", "contents": "water"}}. Omit unknown attributes.
- events: what happens, one present-tense sentence each, using entity names exactly as in "entities"; "salience" is 3 for the main activity of the video, 2 for notable actions, 1 for background.
- relations: [subject, predicate, object] facts between entities ("man", "holds", "drill"), names exactly as in "entities".
- speech_summary: what is said, in 1-3 sentences, keeping names, numbers and stated reasons; "" if there is no speech.

### CHUNK
id: {chunk_id}; time: {t_start} to {t_end}

### TRANSCRIPT
{transcript}

### OUTPUT
Reply with ONE JSON object and nothing else, in this form:
{{"entities": [{{"name": "red mug", "category": "container", "attributes": {{"color": "red", "contents": "coffee"}}}}],
 "events": [{{"description": "man pours coffee into red mug", "entities": ["man", "red mug"], "salience": 2}}],
 "relations": [["man", "holds", "red mug"]],
 "speech_summary": ""}}
"""

# ----------------------------------------------------------------------------- memory
ADJUDICATE = """### TASK: adjudicate
Two descriptions come from different moments of the same video. Decide whether they refer to the SAME physical entity (the same individual object or person, not merely the same kind). Attribute differences may be real changes over time (an object can be painted, filled, moved or opened).

### CANDIDATE A
{a}

### CANDIDATE B
{b}

### OUTPUT
Reply with ONE JSON object: {{"same": true}} or {{"same": false}}
"""

COMPRESS = """### TASK: compress
The events below happened in a video between {t_start} and {t_end}. Replace them with at most {max_items} coarser events that keep what a later question might ask about: who did what to which entity, counts, colors, states and changes of state, and spoken facts. Write entity ids in square brackets exactly as given (e.g. [E3]). Drop repetition.

### EVENTS
{events}

### OUTPUT
Reply with ONE JSON object:
{{"summaries": [{{"description": "...", "entity_ids": ["E3"], "salience": 2}}]}}
"""

SUMMARIZE = """### TASK: summarize
You maintain a running summary of a long video that is watched segment by segment. Update the summary with the new observation. The summary must stay under {max_words} words. Keep times (hh:mm:ss), entities with their attributes, changes of state and spoken facts; when space runs out, shorten the least important older details first.

### CURRENT SUMMARY
{summary}

### NEW OBSERVATION
{observation}

### LIMIT
{max_words} words

### OUTPUT
Reply with ONE JSON object: {{"summary": "..."}}
"""

# ----------------------------------------------------------------------------- answering
ANSWER_FORMATS = {
    "short_text": "Answer in at most 8 words.",
    "number": "Answer with a single number (digits).",
    "yes_no": "Answer with yes or no.",
    "multiple_choice": "Answer with the letter of the correct option only.",
}

ANSWER = """### TASK: answer
Answer the question about a video using ONLY the memory records below. Times are hh:mm:ss from the start of the video. Cite the ids of the records that support the answer. If the records do not contain the answer, give your best guess and set "confidence" to "low".
{format_instruction}

### MEMORY
{memory}

### QUESTION
{question}
{options_block}
### OUTPUT
Reply with ONE JSON object:
{{"answer": "...", "cited_record_ids": ["V3", "S1"], "confidence": "low"}}
"""

JUDGE = """### TASK: judge
Decide whether the candidate answer is correct given the reference answer. Wording may differ; a different entity, number, color, time or state is incorrect.

### QUESTION
{question}

### REFERENCE
{reference}

### CANDIDATE
{candidate}

### OUTPUT
Reply with ONE JSON object: {{"correct": true}} or {{"correct": false}}
"""
