"""Small VLM captions followed by text-model schema extraction (no heuristics)."""

from __future__ import annotations

import hashlib
import json
import re

from PIL import Image

from memvid.io import read_jsonl
from memvid.models.base import GenerativeModel
from memvid.perception.run import resize_to_pixels
from memvid.prompts import sections
from memvid.schemas import Chunk, FrameRecord

CAPTION_VERSION = "1"


def caption_frames(cfg, vlm):
    path = cfg.derived_dir / "visual_captions.jsonl"
    signature = hashlib.sha256(
        json.dumps(
            {
                "version": CAPTION_VERSION,
                "model": vlm.describe(),
                "frames": hashlib.sha256((cfg.derived_dir / "frames/index.jsonl").read_bytes()).hexdigest(),
                "pixels": cfg.perception.max_pixels,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    done = {}
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if rows and rows[0].get("signature") == signature:
            done = {row["chunk_id"]: row for row in rows[1:]}
    if not done:
        path.write_text(json.dumps({"signature": signature}) + "\n", encoding="utf-8")
    frames = read_jsonl(cfg.derived_dir / "frames/index.jsonl", FrameRecord)
    chunks = read_jsonl(cfg.derived_dir / "chunks.jsonl", Chunk)
    with path.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            if chunk.chunk_id in done:
                continue
            selected = sorted((r for r in frames if r.chunk_id == chunk.chunk_id), key=lambda r: r.frame_index)
            if not selected:
                raise ValueError(f"No visual frames for chunk {chunk.chunk_id}")
            images = []
            for frame in selected:
                with Image.open(cfg.derived_dir / frame.path) as im:
                    images.append(resize_to_pixels(im.convert("RGB"), cfg.perception.max_pixels))
            text = vlm.generate(
                "Describe the people, objects, setting and visible actions in these video frames. "
                "The frames are in chronological order. State only visible facts. "
                "Use short plain sentences, not JSON.",
                images=images,
                max_new_tokens=192,
            )
            if not text.strip():
                raise RuntimeError(f"Empty visual caption for chunk {chunk.chunk_id}")
            f.write(json.dumps({"chunk_id": chunk.chunk_id, "text": text, "frame_count": len(selected)}) + "\n")
            f.flush()
            print(f"Visual caption {chunk.chunk_id + 1}/{len(chunks)}: {text}", flush=True)


class CaptionGroundedModel(GenerativeModel):
    def __init__(self, cfg, llm):
        self.llm = llm
        path = cfg.derived_dir / "visual_captions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.captions = {r["chunk_id"]: r["text"] for r in rows[1:]}
        self.signature = rows[0]["signature"]
        self.name = "caption_grounded/" + llm.name

    def describe(self):
        return {
            "name": self.name,
            "text_model": self.llm.describe(),
            "visual_captions_signature": self.signature,
            "adapter_version": 2,
            "max_entities": 3,
            "max_events": 2,
        }

    def generate(self, prompt, *, images=None, json_schema=None, max_new_tokens=None):
        match = re.search(r"### CHUNK\s+id:\s*(\d+)", prompt)
        if not match:
            raise ValueError("Missing chunk identity in perception prompt")
        caption = self.captions[int(match.group(1))]
        source = sections(prompt)
        grounded = (
            "Extract supported facts from this video segment. Return only JSON, no explanation.\n"
            "Use only facts in VISUAL and SPEECH below. Do not invent objects, names or actions.\n"
            "Keys: entities, events, relations, speech_summary.\n"
            "entities: at most 3 objects with fields name (short string), category (string), attributes (object).\n"
            "events: at most 2 objects with description (short string), "
            "entities (list of entity names), salience (1-3).\n"
            "relations: a list of [subject, predicate, object] using entity names, or [].\n"
            "speech_summary: one short sentence preserving names, numbers and reasons from SPEECH.\n"
            "When a fact is absent, omit it. Use empty arrays rather than inventing facts.\n"
            "VISUAL:\n" + caption + "\nSPEECH:\n" + source.get("TRANSCRIPT", "(no speech)") + "\n"
            'JSON keys to fill: {"entities":[],"events":[],"relations":[],"speech_summary":""}'
        )
        marker = "Your previous answer could not be used:\n"
        if marker in prompt:
            error = prompt.split(marker, 1)[1].split("Previous answer:", 1)[0]
            grounded += "\nCorrect this JSON/schema error: " + error[:1000]
        return self.llm.generate(grounded, json_schema=json_schema, max_new_tokens=max_new_tokens)
