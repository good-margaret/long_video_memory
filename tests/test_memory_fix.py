"""Tests for the memory fixes: hierarchical compression, entity priority, the transcript
speech channel, per-condition budgets and the answer guard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memvid.config import ProjectConfig, StructuredConfig
from memvid.experiment import memory_inputs, planned_runs
from memvid.memory.answer import answer_question
from memvid.memory.records import Entity, Event
from memvid.memory.retrieval import Retrieved
from memvid.memory.store import MemoryStore, replay
from memvid.memory.structured import StructuredMemory
from memvid.models.base import GenerativeModel
from memvid.models.embed import ApproxTokenCounter, HashingEmbedder
from memvid.perception.schema import ChunkObservation, ObsEntity, ObsEvent
from memvid.questions.schema import Question

WORDS = ("red green blue yellow board drill saw hammer boat sled dog moose truck snow ice "
         "crab bird gun map sign door table bed cup knife rope tent fire lake hill road").split()


def _obs(i: int, transcript: str = "") -> ChunkObservation:
    w = [WORDS[(i * 7 + k) % len(WORDS)] for k in range(6)]
    return ChunkObservation(
        chunk_id=i, t_start=20.0 * i, t_end=20.0 * (i + 1),
        entities=[ObsEntity(name="man", category="person"),
                  ObsEntity(name=f"{w[0]} {w[1]}", category="tool")],
        events=[ObsEvent(description=f"man uses the {w[0]} {w[1]} near the {w[2]} {i}",
                         entities=["man", f"{w[0]} {w[1]}"], salience=2),
                ObsEvent(description=f"the {w[3]} {w[4]} passes the {w[5]} at minute {i}",
                         entities=[], salience=1)],
        speech_summary=f"he talks about the {w[2]} and the {w[5]} number {i}",
        transcript=transcript)


def _run(n: int, budget: int, **cfg) -> StructuredMemory:
    store = MemoryStore(HashingEmbedder(), ApproxTokenCounter())
    mem = StructuredMemory(store, budget, StructuredConfig(**cfg), llm=None)
    for i in range(n):
        mem.process(_obs(i))
    return mem


class HierarchicalCompressionTest(unittest.TestCase):
    def test_budget_and_levels(self) -> None:
        mem = _run(60, 900)
        sizes = [s["tokens"] for s in mem.sizes]
        self.assertLessEqual(max(sizes), 900)
        recount = sum(mem.store.counter.count(r.line()) for r in mem.store.records.values())
        self.assertEqual(mem.store.tokens(), recount)
        levels = [r.level for r in mem.store.records.values() if isinstance(r, Event)]
        self.assertLessEqual(max(levels), mem.cfg.max_compression_level + 1)
        self.assertGreater(mem.counters.get("recompressions", 0), 0)
        self.assertEqual(set(replay(mem.store.ops)), set(mem.store.records))

    def test_recompression_keeps_more_of_the_past(self) -> None:
        """With re-compression, old windows survive as summaries instead of being evicted."""
        deep, flat = _run(60, 900), _run(60, 900, max_compression_level=0)

        def oldest_event_chunk(mem):
            return min(min(r.source_chunks) for r in mem.store.records.values()
                       if isinstance(r, Event))

        self.assertLess(deep.counters.get("evicted_event", 0), flat.counters.get("evicted_event", 0))
        self.assertLessEqual(oldest_event_chunk(deep), oldest_event_chunk(flat))

    def test_summaries_keep_provenance(self) -> None:
        mem = _run(40, 900)
        summarized = [r for r in mem.store.records.values()
                      if isinstance(r, Event) and r.summarized]
        self.assertTrue(summarized)
        for r in summarized:  # provenance is the union of the merged events' chunks
            self.assertTrue(all(r.t_start <= 20.0 * c + 1e-6 and 20.0 * c < r.t_end
                                for c in r.source_chunks))
        # two events of one chunk may be merged, but most summaries span several chunks
        self.assertTrue(any(len(r.source_chunks) >= 2 for r in summarized))


class CompressionPromptCapTest(unittest.TestCase):
    def test_prompt_input_is_capped(self) -> None:
        """Each compression call sees at most compression_max_input_tokens of records."""
        seen = []

        class Recorder(StructuredMemory):
            def _summarize(self, group, t0, t1):
                seen.append(sum(self.store.tokens_of(r.id) for r in group))
                return super()._summarize(group, t0, t1)

        store = MemoryStore(HashingEmbedder(), ApproxTokenCounter())
        mem = Recorder(store, 900, StructuredConfig(compression_max_input_tokens=150), llm=None)
        for i in range(60):
            mem.process(_obs(i, transcript="word " * 60))
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 150 + max(store.tokens_of(r) for r in store.records))
        self.assertLessEqual(max(s["tokens"] for s in mem.sizes), 900)


class EntityPriorityTest(unittest.TestCase):
    def test_unreferenced_entity_ranks_below_referenced_event(self) -> None:
        mem = _run(1, 10_000)
        ent = Entity(id="E99", source_chunks=[0], name="lonely cup", first_seen=0, last_seen=20)
        ev = Event(id="V99", source_chunks=[0], t_start=0, t_end=20, description="x",
                   entity_ids=["E1"], salience=2)
        p_lonely = mem.priority(ent, {}, 0.0)
        p_used = mem.priority(ent, {"E99": 2}, 0.0)
        p_event = mem.priority(ev, {}, 0.0)
        self.assertLess(p_lonely, p_event)
        self.assertGreater(p_used, p_event)


class SpeechInputTest(unittest.TestCase):
    def _cfg(self, root: Path, mode: str) -> ProjectConfig:
        d = root / "derived" / "vid"
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"chunk_id": i, "t_start": 20.0 * i, "t_end": 20.0 * (i + 1),
                 "transcript_text": f"the price was {i} dollars said Drew"} for i in range(3)]
        (d / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return ProjectConfig.model_validate({
            "video": {"video_id": "vid", "filename": "vid.mp4"},
            "paths": {"derived_dir": str(root / "derived")},
            "memory_input": {"speech": mode, "max_transcript_chars": 20}})

    def test_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            obs = [_obs(i) for i in range(3)]
            both = memory_inputs(self._cfg(Path(tmp), "both"), obs)
            self.assertTrue(both[1].speech_summary and both[1].transcript.startswith("the price was 1"))
            self.assertLessEqual(len(both[1].transcript), 24)  # cut at a word + " ..."
            self.assertIn('Said: "', both[1].to_text())
            only = memory_inputs(self._cfg(Path(tmp), "transcript"), obs)
            self.assertEqual(only[0].speech_summary, "")
            summ = memory_inputs(self._cfg(Path(tmp), "summary"), obs)
            self.assertEqual(summ[0].transcript, "")
            self.assertEqual(obs[0].transcript, "")  # inputs are copied, not modified

    def test_c3_stores_transcript_as_speech(self) -> None:
        store = MemoryStore(HashingEmbedder(), ApproxTokenCounter())
        mem = StructuredMemory(store, None, StructuredConfig(), llm=None)
        mem.process(_obs(0, transcript="Drew says the wool costs 100 dollars an ounce"))
        speech = [r.description for r in store.records.values()
                  if isinstance(r, Event) and r.modality == "speech"]
        self.assertTrue(any("100 dollars" in s for s in speech))
        self.assertEqual(len(speech), 2)  # the summary and the raw transcript


class PlannedRunsTest(unittest.TestCase):
    def test_condition_budgets(self) -> None:
        cfg = ProjectConfig.model_validate({
            "video": {"video_id": "vid", "filename": "vid.mp4"},
            "experiment": {"conditions": ["C1", "C2", "C3", "C4"], "ablations": ["no_resolution"],
                           "condition_budgets": {"C1": ["B1"]}}})
        runs = planned_runs(cfg)
        self.assertEqual([b for c, b in runs if c == "C1"], ["B1"])
        self.assertEqual([b for c, b in runs if c == "C2"], ["B1", "B2", "B3"])
        self.assertIn(("C4", "full"), runs)
        self.assertEqual(cfg.budgets.values, {"B1": 0.10, "B2": 0.25, "B3": 0.50})


class _Fixed(GenerativeModel):
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt, *, images=None, json_schema=None, max_new_tokens=None):
        return self.text


class AnswerGuardTest(unittest.TestCase):
    def _q(self, kind: str) -> Question:
        extra = {"options": ["a cup", "a drill"], "answer": "B"} if kind == "multiple_choice" \
            else {"answer": "white"}
        return Question(id="q1", question="What did he hold?", answer_type=kind, type="past_event",
                        gt_intervals=[(0.0, 20.0)], modality="visual", **extra)

    def test_confidence_word_is_not_an_answer(self) -> None:
        ret = Retrieved(records=[], ranked_ids=[], scores={}, tokens=0, matched_entities=[], cues=[])
        llm = _Fixed('{"answer": "low", "cited_record_ids": [], "confidence": "low"}')
        self.assertEqual(answer_question(self._q("multiple_choice"), ret, llm, {}).answer, "")
        llm = _Fixed('{"answer": "B", "cited_record_ids": [], "confidence": "low"}')
        self.assertEqual(answer_question(self._q("multiple_choice"), ret, llm, {}).answer, "B")


if __name__ == "__main__":
    unittest.main()
