"""Deterministic rule-based stand-in for the generative models.

It reads the structured prompts of memvid.prompts and returns syntactically valid
answers built from the prompt's own text (it never looks at images). It exists to run
the whole pipeline end to end without a GPU (the smoke test, demo development).
Results produced with it are meaningless and must never be reported; runs record the
backend name, and every result table built from such runs is marked as not valid.
"""

from __future__ import annotations

import json
import re
from typing import Any

from memvid.models.base import GenerativeModel
from memvid.prompts import sections, task_of
from memvid.textutils import STOPWORDS, content_words

_ID = re.compile(r"\[([A-Z]\d+)\]")


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(content_words(a)), set(content_words(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class HeuristicModel(GenerativeModel):
    name = "heuristic"

    def generate(self, prompt, *, images=None, json_schema=None, max_new_tokens=None) -> str:
        task = task_of(prompt)
        sec = sections(prompt)
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            return json.dumps({"error": f"unknown task {task}"})
        if task == "perceive":
            return json.dumps(self._task_perceive(sec, images or []))
        return json.dumps(handler(sec))

    # -------------------------------------------------------------- tasks
    @staticmethod
    def _dominant_color(images) -> str | None:
        if not images:
            return None
        r, g, b = (sum(c) / len(c) for c in zip(*images[0].convert("RGB").resize((8, 8)).getdata(), strict=True))
        names = {"red": (200, 40, 40), "green": (40, 160, 60), "blue": (40, 60, 200),
                 "yellow": (220, 210, 50), "white": (235, 235, 235), "black": (20, 20, 20),
                 "gray": (128, 128, 128), "brown": (130, 80, 40), "purple": (130, 50, 160)}
        return min(names, key=lambda n: sum((x - y) ** 2 for x, y in zip((r, g, b), names[n], strict=True)))

    @classmethod
    def _task_perceive(cls, sec: dict[str, str], images=()) -> dict[str, Any]:
        text = sec.get("TRANSCRIPT", "")
        if text.strip() == "(no speech)":
            text = ""
        nouns: list[str] = []
        for m in re.finditer(r"\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)?)", text.lower()):
            phrase = " ".join(w for w in m.group(1).split() if w not in STOPWORDS)
            if phrase and phrase not in nouns:
                nouns.append(phrase)
        nouns = nouns[:5]
        first = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else ""
        events = [{"description": first[:200], "entities": nouns[:2], "salience": 2}] if first else []
        entities = [{"name": n, "category": "other", "attributes": {}} for n in nouns]
        color = cls._dominant_color(list(images))
        if color:
            entities.append({"name": "scene", "category": "place", "attributes": {"color": color}})
            events.append({"description": f"the scene is mostly {color}", "entities": ["scene"],
                           "salience": 1})
        return {
            "entities": entities,
            "events": events,
            "relations": [],
            "speech_summary": text[:300],
        }

    @staticmethod
    def _task_adjudicate(sec: dict[str, str]) -> dict[str, Any]:
        def name(block: str) -> str:
            m = re.search(r"name:\s*(.+)", block)
            return m.group(1) if m else block

        return {"same": _jaccard(name(sec.get("CANDIDATE A", "")), name(sec.get("CANDIDATE B", ""))) >= 0.5}

    @staticmethod
    def _task_compress(sec: dict[str, str]) -> dict[str, Any]:
        lines = [ln for ln in sec.get("EVENTS", "").splitlines() if ln.strip()]
        descs, ids, sal = [], [], 1
        for ln in lines:
            body = re.sub(r"^\[[A-Z]\d+\]\s*", "", ln)
            body = re.sub(r"^\S+\s*-\s*\S+\s*", "", body)
            body = re.sub(r"^(?:event|summary|speech):\s*", "", body)
            m = re.search(r"\(salience (\d)\)", body)
            if m:
                sal = max(sal, int(m.group(1)))
                body = body.replace(m.group(0), "")
            descs.append(body.strip())
            ids += [i for i in re.findall(r"E\d+", " ".join(re.findall(r"\{([^}]*)\}", body)))
                    if i not in ids]
            ids += [i for i in _ID.findall(body) if i.startswith("E") and i not in ids]
            body = re.sub(r"\s*\{[^}]*\}", "", body)
        return {"summaries": [{"description": "; ".join(descs[:3])[:300], "entity_ids": ids,
                               "salience": sal}]}

    @staticmethod
    def _task_summarize(sec: dict[str, str]) -> dict[str, Any]:
        limit = int(re.findall(r"\d+", sec.get("LIMIT", "200"))[0])
        current = sec.get("CURRENT SUMMARY", "")
        current = "" if current == "(empty)" else current
        words = (current + " " + sec.get("NEW OBSERVATION", "")).split()
        return {"summary": " ".join(words[-limit:])}

    @staticmethod
    def _task_answer(sec: dict[str, str]) -> dict[str, Any]:
        q = sec.get("QUESTION", "")
        best, best_score = None, -1.0
        for ln in sec.get("MEMORY", "").splitlines():
            m = re.match(r"\[([A-Z]\d+)\]\s*(.*)", ln.strip())
            if not m:
                continue
            score = len(set(content_words(q)) & set(content_words(m.group(2))))
            if score > best_score:
                best, best_score = m, score
        if best is None:
            return {"answer": "unknown", "cited_record_ids": [], "confidence": "low"}
        body = re.sub(r"\d{2}:\d{2}:\d{2}(?:-\d{2}:\d{2}:\d{2})?", " ", best.group(2))
        body = re.sub(r"^\s*(?:event|summary|speech|note|change|relation|entity)\b:?", " ", body)
        body = re.sub(r"\{[^}]*\}|\[[A-Z]\d+\]", " ", body)
        words = [w for w in content_words(body) if w not in set(content_words(q))]
        return {"answer": " ".join(words[:6]) or "unknown",
                "cited_record_ids": [best.group(1)], "confidence": "low"}

    @staticmethod
    def _task_judge(sec: dict[str, str]) -> dict[str, Any]:
        ref = set(content_words(sec.get("REFERENCE", "")))
        cand = set(content_words(sec.get("CANDIDATE", "")))
        return {"correct": bool(ref) and ref <= cand}

    def describe(self) -> dict[str, Any]:
        return {"name": "heuristic", "backend": "heuristic", "warning": "not a real model"}
