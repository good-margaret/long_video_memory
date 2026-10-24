from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from memvid.config import ASRConfig, load_config
from memvid.evaluation.metrics import JudgeCache, _specific_evidence_record, interval_distance, summarize_rows
from memvid.episodic import fuse_chunk_rankings
from memvid.memory.records import Entity, Event
from memvid.preprocess.asr import import_srt, load_pcm_wav
from memvid.perception.run import build_rescue_prompt
from memvid.perception.schema import ChunkObservation, ModelObservation
from memvid.schemas import Chunk


class ObservationValidationTest(unittest.TestCase):
    def test_rejects_empty_model_observation(self) -> None:
        with self.assertRaises(ValidationError):
            ModelObservation.model_validate(
                {"entities": [], "events": [], "relations": [], "speech_summary": ""}
            )

    def test_accepts_spoken_information_without_visual_entities(self) -> None:
        observation = ModelObservation.model_validate(
            {
                "entities": [],
                "events": [],
                "relations": [],
                "speech_summary": "The speaker introduces the experiment.",
            }
        )
        self.assertTrue(observation.speech_summary)

    def test_chunk_records_transcript_fallback(self) -> None:
        observation = ChunkObservation(
            chunk_id=2,
            t_start=40.0,
            t_end=60.0,
            speech_summary="Welcome to the city of Nome.",
            fallback="transcript_after_vlm_failure",
        )
        self.assertTrue(observation.ok)
        self.assertEqual(observation.fallback, "transcript_after_vlm_failure")

    def test_visual_rescue_prompt_requires_spoken_and_visible_content(self) -> None:
        chunk = Chunk(
            chunk_id=2,
            t_start=40.0,
            t_end=60.0,
            transcript_text="Welcome to the city of Nome.",
        )
        prompt = build_rescue_prompt(chunk, n_frames=4, max_chars=2000)
        self.assertIn("recovery request", prompt)
        self.assertIn("Welcome to the city of Nome.", prompt)
        self.assertIn("clearly visible content", prompt)


class SubtitleImportTest(unittest.TestCase):
    def test_imports_timed_segments_and_strips_tags(self) -> None:
        content = """1
00:00:01,000 --> 00:00:02,500
<i>Hello</i> world

2
00:00:03.000 --> 00:00:04.000
Second line
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.srt"
            path.write_text(content, encoding="utf-8")
            segments, info = import_srt(
                path, ASRConfig(source="srt", subtitle_path=str(path), language="en")
            )
        self.assertEqual([s.text for s in segments], ["Hello world", "Second line"])
        self.assertEqual((segments[0].start, segments[0].end), (1.0, 2.5))
        self.assertEqual(info.source, "srt")

    def test_loads_pcm_wav_without_pyav(self) -> None:
        samples = np.array([0, 16384, -16384, 32767], dtype="<i2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(samples.tobytes())
            loaded = load_pcm_wav(path)
        np.testing.assert_allclose(loaded, samples.astype(np.float32) / 32768.0)


class MetricTest(unittest.TestCase):
    def test_interval_distance(self) -> None:
        self.assertEqual(interval_distance([(0, 10)], [(5, 12)]), 0)
        self.assertEqual(interval_distance([(0, 10)], [(15, 20)]), 5)
        self.assertIsNone(interval_distance([], [(15, 20)]))

    def test_broad_provenance_is_not_specific_evidence(self) -> None:
        entity = Entity(id="E1", source_chunks=[1, 2], name="person", first_seen=0, last_seen=40)
        summary = Event(id="V1", source_chunks=[1, 2], t_start=0, t_end=40,
                        description="coarse summary", summarized=True)
        event = Event(id="V2", source_chunks=[2], t_start=20, t_end=40,
                      description="concrete event")
        self.assertFalse(_specific_evidence_record(entity))
        self.assertFalse(_specific_evidence_record(summary))
        self.assertTrue(_specific_evidence_record(event))

    def test_judge_cache_key_is_versioned_but_manual_key_is_stable(self) -> None:
        self.assertTrue(JudgeCache.key("q1", "An axe").startswith("strict-v2::"))
        self.assertEqual(JudgeCache.manual_key("q1", "An axe"), "q1::axe")

    def test_hybrid_rank_fusion_rewards_agreement(self) -> None:
        archive = [(4, 0.9), (2, 0.8), (8, 0.7)]
        c3 = [(8, 0.2), (5, 0.1), (4, 0.05)]
        ranked = fuse_chunk_rankings(archive, c3, "hybrid")
        self.assertEqual(ranked[0][0], 4)
        self.assertEqual({chunk_id for chunk_id, _ in ranked}, {2, 4, 5, 8})

    def test_summary_groups_accuracy_by_modality(self) -> None:
        rows = [
            {"correct": True, "grounded": True, "g_precision": 1.0, "tiou": 1.0,
             "retrieved": True, "recall_at_1": True, "recall_at_5": True,
             "recall_at_10": True, "reciprocal_rank": 1.0, "temporal_error_s": 0.0,
             "temporal_hit_20s": True, "temporal_hit_60s": True, "survived": True,
             "type": "past_event", "modality": "visual"},
            {"correct": False, "grounded": False, "g_precision": 0.0, "tiou": 0.0,
             "retrieved": False, "recall_at_1": False, "recall_at_5": False,
             "recall_at_10": False, "reciprocal_rank": 0.0, "temporal_error_s": 90.0,
             "temporal_hit_20s": False, "temporal_hit_60s": False, "survived": False,
             "type": "past_event", "modality": "speech"},
        ]
        summary = summarize_rows(rows, n_boot=100, seed=0)
        self.assertEqual(summary["by_modality"]["visual"]["mean"], 1.0)
        self.assertEqual(summary["by_modality"]["speech"]["mean"], 0.0)


class ExperimentConfigTest(unittest.TestCase):
    def test_all_experiment_configs_load(self) -> None:
        for name in ("f4_asr", "f8_asr", "f4_srt", "f8_srt"):
            cfg = load_config(f"configs/experiment_{name}.yaml")
            self.assertEqual(cfg.video.video_id, "KbahC-QCKU8")


if __name__ == "__main__":
    unittest.main()
