from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from memvid.config import ASRConfig, load_config
from memvid.evaluation.metrics import interval_distance, summarize_rows
from memvid.preprocess.asr import import_srt, load_pcm_wav
from memvid.perception.schema import ModelObservation


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
