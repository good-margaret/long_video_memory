"""Tests for v1.5.1: the entity/relation share cap of C3, the summary span bound, per-file
question freezing (questions_v2.yaml next to a frozen questions.yaml) and the judge-cache key."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from memvid.config import ProjectConfig, StructuredConfig
from memvid.errors import MemvidError
from memvid.evaluation.metrics import JudgeCache
from memvid.memory.records import Entity, Event, Relation
from memvid.memory.store import MemoryStore
from memvid.memory.structured import StructuredMemory
from memvid.models.embed import ApproxTokenCounter, HashingEmbedder
from memvid.perception.schema import ChunkObservation, ObsEntity, ObsEvent
from memvid.questions.manage import freeze_questions, load_questions
from memvid.questions.schema import Question

OBJ = ("drill saw hammer boat sled rope tent cup knife map axe pot crab bucket net auger "
       "rifle goggles glove hat jacket bag cooler box tote bin lid shovel truck").split()
COL = "red green blue yellow white black brown grey orange purple".split()


def _obs(i: int) -> ChunkObservation:
    """A chunk with many new objects and relations, as the real perception produces."""
    objs = [f"{COL[(i + k) % len(COL)]} {OBJ[(3 * i + k) % len(OBJ)]} {i}" for k in range(5)]
    return ChunkObservation(
        chunk_id=i, t_start=20.0 * i, t_end=20.0 * (i + 1),
        entities=[ObsEntity(name="man", category="person")]
        + [ObsEntity(name=o, category="tool", attributes={"color": o.split()[0]}) for o in objs],
        events=[ObsEvent(description=f"man handles the {objs[0]} and the {objs[1]} at minute {i}",
                         entities=["man", objs[0], objs[1]], salience=2),
                ObsEvent(description=f"the {objs[2]} lies next to the {objs[3]} in scene {i}",
                         entities=[objs[2], objs[3]], salience=1)],
        relations=[["man", "holds", objs[0]], ["man", "uses", objs[1]], [objs[2], "next to", objs[3]],
                   [objs[3], "on", objs[4]], ["man", "looks at", objs[4]]],
        speech_summary=f"he describes the {objs[0]} number {i}")


def _run(n: int, budget: int, **cfg) -> StructuredMemory:
    store = MemoryStore(HashingEmbedder(), ApproxTokenCounter())
    mem = StructuredMemory(store, budget, StructuredConfig(**cfg), llm=None)
    for i in range(n):
        mem.process(_obs(i))
    return mem


def _tokens(mem: StructuredMemory, kinds) -> int:
    return sum(mem.store.tokens_of(r.id) for r in mem.store.records.values() if isinstance(r, kinds))


class StructureCapTest(unittest.TestCase):
    def test_share_is_capped_and_budget_holds(self) -> None:
        mem = _run(80, 3000)
        self.assertLessEqual(max(s["tokens"] for s in mem.sizes), 3000)
        self.assertLessEqual(_tokens(mem, (Entity, Relation)), 0.40 * 3000)
        self.assertGreater(mem.counters.get("structure_cap_evictions", 0), 0)

    def test_larger_budget_keeps_more_events(self) -> None:
        """The defect of v1.5.0: doubling the budget bought relations, not events."""
        small, large = _run(80, 3000), _run(80, 6000)
        self.assertGreater(_tokens(large, Event), _tokens(small, Event))
        uncapped = _run(80, 6000, structure_max_share=1.0)
        self.assertGreater(_tokens(large, Event), _tokens(uncapped, Event))
        self.assertGreater(_tokens(uncapped, (Entity, Relation)), _tokens(large, (Entity, Relation)))

    def test_summary_span_is_bounded(self) -> None:
        mem = _run(120, 1500)
        spans = [r.t_end - r.t_start for r in mem.store.records.values() if isinstance(r, Event)]
        limit = mem.cfg.compression_window_s * 2 ** mem.cfg.max_compression_level
        self.assertEqual(mem.cfg.max_compression_level, 2)
        self.assertLessEqual(max(spans), limit)


def _question(qid: str, answer: str = "white") -> dict:
    return {"id": qid, "question": f"What color was it ({qid})?", "answer": answer,
            "answer_type": "short_text", "type": "attribute_recall",
            "gt_intervals": [["00:00:00", "00:00:20"]], "modality": "visual"}


class QuestionSetFreezeTest(unittest.TestCase):
    def _cfg(self, root: Path, filename: str) -> ProjectConfig:
        cfg = ProjectConfig.model_validate({
            "video": {"video_id": "vid", "filename": "vid.mp4"},
            "paths": {"derived_dir": str(root / "derived"), "annotations_dir": str(root / "ann")},
            "questions": {"filename": filename}})
        cfg._root = root
        return cfg

    def test_v2_frozen_next_to_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann = root / "ann" / "vid"
            ann.mkdir(parents=True)
            v1 = {"video_id": "vid", "annotator": "test", "questions": [_question("q1"), _question("q2"), _question("q3")]}
            v2 = {"video_id": "vid", "annotator": "test", "questions": [_question("q1", "black"), _question("q3")]}
            (ann / "questions.yaml").write_text(yaml.safe_dump(v1))
            (ann / "questions_v2.yaml").write_text(yaml.safe_dump(v2))
            (ann / "split.json").write_text(json.dumps({"q1": "test", "q2": "dev", "q3": "test"}))

            c1, c2 = self._cfg(root, "questions.yaml"), self._cfg(root, "questions_v2.yaml")
            self.assertEqual(freeze_questions(c1)["n_questions"], 3)
            info = freeze_questions(c2)  # v2 drops q2: allowed, split.json covers it
            self.assertEqual((info["n_questions"], info["n_test"]), (2, 2))
            self.assertEqual([q.answer for q in load_questions(c2, "test", require_frozen=True)],
                             ["black", "white"])
            self.assertEqual(len(load_questions(c1, "test", require_frozen=True)), 2)

            v2["questions"][0]["answer"] = "grey"  # editing a frozen set is refused
            (ann / "questions_v2.yaml").write_text(yaml.safe_dump(v2))
            with self.assertRaises(MemvidError):
                load_questions(c2, "test", require_frozen=True)
            with self.assertRaises(MemvidError):
                freeze_questions(c2)
            load_questions(c1, "test", require_frozen=True)  # v1 is untouched


class JudgeKeyTest(unittest.TestCase):
    def test_reference_change_invalidates_verdict(self) -> None:
        base = dict(id="q56", question="What color was the glove?", answer_type="short_text",
                    type="attribute_recall", gt_intervals=[(0.0, 20.0)], modality="visual")
        old = Question(answer="a white glove", **base)
        new = Question(answer="a black glove", **base)
        self.assertNotEqual(JudgeCache.judge_key(old, "black glove"),
                            JudgeCache.judge_key(new, "black glove"))
        self.assertEqual(JudgeCache.judge_key(new, "Black glove."),
                         JudgeCache.judge_key(new, "black glove"))


if __name__ == "__main__":
    unittest.main()
