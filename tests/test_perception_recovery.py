import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from memvid.config import ProjectConfig
from memvid.errors import PipelineStateError
from memvid.io import read_json, write_jsonl
from memvid.perception import run as P
from memvid.perception.schema import ChunkObservation, ModelObservation
from memvid.prompts import PROMPT_VERSION
from memvid.schemas import Chunk, FrameRecord


@pytest.fixture
def checkpoint(tmp_path):
    cfg = ProjectConfig(video={"video_id": "test", "filename": "test.mp4"}, paths={"derived_dir": str(tmp_path)})
    d = cfg.derived_dir
    d.mkdir(parents=True)
    chunks = [Chunk(chunk_id=i, t_start=i * 20, t_end=(i + 1) * 20, transcript_text="Speech") for i in range(3)]
    frames = [
        FrameRecord(chunk_id=i, frame_index=0, t_target=i * 20 + 1, path="frame.jpg", width=64, height=64)
        for i in range(3)
    ]
    write_jsonl(d / "chunks.jsonl", chunks)
    (d / "frames").mkdir()
    write_jsonl(d / "frames/index.jsonl", frames)
    Image.new("RGB", (64, 64), "white").save(d / "frame.jpg")
    model = SimpleNamespace(describe=lambda: {"name": "test-model"})
    old = [
        ChunkObservation(
            chunk_id=0, t_start=0, t_end=20, speech_summary="fallback", fallback="transcript_after_vlm_failure"
        ),
        ChunkObservation(chunk_id=1, t_start=20, t_end=40, speech_summary="keep unchanged"),
        ChunkObservation(chunk_id=2, t_start=40, t_end=60, ok=False, error="invalid JSON"),
    ]
    signature = {
        "prompt_version": PROMPT_VERSION,
        "model": model.describe(),
        "perception": cfg.perception.model_dump(mode="json"),
    }
    partial = d / "observations.partial.jsonl"
    partial.write_text(
        json.dumps({"signature": signature}) + "\n" + "\n".join(o.model_dump_json() for o in old) + "\n",
        encoding="utf-8",
    )
    return cfg, model, old, partial


def test_selective_retry_keeps_successful_rows_and_updates_manifest(checkpoint):
    cfg, model, old, partial = checkpoint

    def perceive(cfg, model, chunk, frames, *, recovery=False):
        assert recovery
        return ChunkObservation(
            chunk_id=chunk.chunk_id,
            t_start=chunk.t_start,
            t_end=chunk.t_end,
            speech_summary="recovered",
            rescue_used=True,
        )

    with patch.object(P, "perceive_chunk", side_effect=perceive) as generate:
        obs = P.run_perception(cfg, model, retry_failed=True, require_matching_resume=True, progress=False)
    assert [call.args[2].chunk_id for call in generate.call_args_list] == [0, 2]
    assert obs[1].model_dump() == old[1].model_dump()
    assert not any(P.needs_visual_recovery(o) for o in obs)
    assert len(partial.read_text().splitlines()) == 6  # header + 3 originals + 2 replacements
    info = read_json(cfg.derived_dir / "perception_info.json")
    assert info["selective_recovery"]["retried_chunk_ids"] == [0, 2]
    assert info["selective_recovery"]["retained_chunk_count"] == 1
    backup = cfg.derived_dir / info["selective_recovery"]["backup_dir"]
    assert len((backup / "observations.partial.jsonl").read_text().splitlines()) == 4
    from memvid.perception.run import load_observations

    assert len(load_observations(cfg)) == 3  # final file matches its updated manifest hash


def test_default_cli_resume_behavior_is_unchanged(checkpoint):
    cfg, model, old, _ = checkpoint
    with patch.object(P, "perceive_chunk") as generate:
        obs = P.run_perception(cfg, model, progress=False)
    generate.assert_not_called()
    assert obs == old


def test_mismatch_refuses_full_restart_and_preserves_checkpoint(checkpoint):
    cfg, model, _, partial = checkpoint
    original = partial.read_bytes()
    cfg.perception.max_pixels = 128 * 128
    with patch.object(P, "perceive_chunk") as generate, pytest.raises(PipelineStateError, match="Refusing"):
        P.run_perception(cfg, model, retry_failed=True, require_matching_resume=True, progress=False)
    generate.assert_not_called()
    assert partial.read_bytes() == original


def test_failed_retry_is_not_accepted_and_next_run_only_retries_remaining(checkpoint):
    cfg, model, old, _ = checkpoint

    def first(cfg, model, chunk, frames, *, recovery=False):
        return (
            old[2]
            if chunk.chunk_id == 2
            else ChunkObservation(chunk_id=0, t_start=0, t_end=20, speech_summary="recovered")
        )

    with patch.object(P, "perceive_chunk", side_effect=first):
        obs = P.run_perception(cfg, model, retry_failed=True, require_matching_resume=True, progress=False)
    assert [o.chunk_id for o in obs if P.needs_visual_recovery(o)] == [2]
    with patch.object(P, "perceive_chunk", return_value=old[2]) as generate:
        P.run_perception(cfg, model, retry_failed=True, require_matching_resume=True, progress=False)
    assert [call.args[2].chunk_id for call in generate.call_args_list] == [2]


def test_missing_partial_refuses_accidental_full_run(checkpoint):
    cfg, model, old, partial = checkpoint
    write_jsonl(cfg.derived_dir / "observations.jsonl", old)
    partial.unlink()
    with pytest.raises(PipelineStateError, match="checkpoint"):
        P.run_perception(cfg, model, retry_failed=True, require_matching_resume=True, progress=False)


def test_empty_partial_cannot_overwrite_completed_results(checkpoint):
    cfg, model, old, partial = checkpoint
    write_jsonl(cfg.derived_dir / "observations.jsonl", old)
    partial.write_text("", encoding="utf-8")
    with pytest.raises(PipelineStateError, match="checkpoint"):
        P.run_perception(cfg, model, retry_failed=True, require_matching_resume=True, progress=False)
    assert partial.read_bytes() == b""


def test_recovery_uses_same_frames_and_larger_budget(checkpoint):
    cfg, model, _, _ = checkpoint
    chunk = Chunk(chunk_id=0, t_start=0, t_end=20, transcript_text="Speech")
    frame = FrameRecord(chunk_id=0, frame_index=0, t_target=1, path="frame.jpg", width=64, height=64)
    result = SimpleNamespace(
        ok=True, value=ModelObservation(entities=[], events=[], relations=[], speech_summary="Model speech"), attempts=1
    )
    with patch.object(P, "generate_validated", return_value=result) as generate:
        obs = P.perceive_chunk(cfg, model, chunk, [frame], recovery=True)
    assert obs.rescue_used and obs.ok
    assert generate.call_args.kwargs["max_new_tokens"] >= 2048
    assert len(generate.call_args.kwargs["images"]) == 1
    assert cfg.models.vlm.max_new_tokens == 1024
    assert "red mug" not in generate.call_args.args[1]
