"""No model downloads: verify temporal isolation, failure handling and clip support."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from memvid.config import ProjectConfig
from memvid.demo.notebook import DemoSettings, NotebookDemo, config_dict, merge_support
from memvid.memory.persist import save_memory
from memvid.memory.records import Entity
from memvid.models.embed import ApproxTokenCounter, HashingEmbedder


class Reply:
    def __init__(self, text):
        self.text = text

    def generate(self, *args, **kwargs):
        return self.text


@pytest.fixture
def demo(tmp_path):
    obj = NotebookDemo.__new__(NotebookDemo)
    obj.session = tmp_path
    obj.settings = DemoSettings("unused", str(tmp_path))
    obj.run_path = tmp_path / "run"
    obj.prepared = tmp_path / "prepared.mp4"
    obj.cfg = ProjectConfig(
        video={"video_id": "demo", "filename": "prepared.mp4"}, models={"embedder": "hashing", "tokenizer": "approx"}
    )
    # Match the public operation-log schema explicitly.
    old = Entity(id="e1", source_chunks=[0], name="box", first_seen=0, last_seen=20, attributes={"color": "red"})
    new = old.model_copy(update={"source_chunks": [0, 1], "last_seen": 40, "attributes": {"color": "blue"}})
    ops = [
        {
            "seq": i,
            "chunk_id": i,
            "op": "upsert",
            "label": "update",
            "record_id": "e1",
            "record": r.model_dump(mode="json"),
            "reason": "test",
        }
        for i, r in enumerate([old, new])
    ]
    save_memory(
        obj.run_path / "memory.sqlite",
        meta={"condition": "C3", "budget_tokens": 2000},
        records={"e1": new},
        ops=ops,
        resolution=[],
        sizes=[{"chunk_id": i, "tokens": 20, "n_records": 1} for i in range(2)],
        bounds={0: (0.0, 20.0), 1: (20.0, 40.0)},
    )
    return obj


def test_snapshot_has_no_future_updates(demo):
    _, records, _ = demo.snapshot(0)
    assert records["e1"].attributes["color"] == "red"
    assert records["e1"].source_chunks == [0]
    assert demo.snapshot(1)[1]["e1"].attributes["color"] == "blue"
    with pytest.raises(ValueError):
        demo.snapshot(99)


def test_changes_show_before_and_after(demo):
    row = demo.changes(1)[0]
    assert "red" in row[-2] and "blue" in row[-1]


def test_replay_and_history_preserve_deletions(demo):
    saved, _, _ = demo.snapshot(1)
    operations = [
        *saved.ops,
        {
            "seq": 2,
            "chunk_id": 2,
            "op": "delete",
            "label": "evict",
            "record_id": "e1",
            "record": None,
            "reason": "budget",
        },
    ]
    save_memory(
        demo.run_path / "memory.sqlite",
        meta=saved.meta,
        records={},
        ops=operations,
        resolution=[],
        sizes=[*saved.sizes, {"chunk_id": 2, "tokens": 0, "n_records": 0}],
        bounds={**saved.bounds, 2: (40, 60)},
    )
    assert demo.snapshot(2)[1] == {}
    row = demo.changes(2)[0]
    assert "blue" in row[-2] and row[-1] == "" and row[1] == "delete"


@pytest.mark.parametrize(
    "reply,status",
    [
        ('{"answer":"red","cited_record_ids":["e1"],"confidence":"high"}', "answered"),
        ('{"answer":"blue","cited_record_ids":["future"],"confidence":"high"}', "invalid_citations"),
        ('{"answer":"","cited_record_ids":[],"confidence":"low"}', "insufficient_evidence"),
        ("not json", "model_error"),
    ],
)
def test_answer_status_and_visible_bounds(demo, reply, status):
    with (
        patch("memvid.demo.notebook.get_embedder", return_value=HashingEmbedder()),
        patch("memvid.demo.notebook.get_token_counter", return_value=ApproxTokenCounter()),
        patch("memvid.demo.notebook.get_llm", return_value=Reply(reply)),
        patch.object(demo, "clips", return_value=[]) as clips,
    ):
        res = demo.ask("What color is the box?", 0)
    assert res["status"] == status
    assert all(r.source_chunks == [0] for r in res["retrieved"])
    if status == "answered":
        assert res["answer"].cited_intervals == [(0.0, 20.0)]
    else:
        clips.assert_not_called()


def test_empty_memory_skips_llm(demo):
    with (
        patch.object(demo, "snapshot", return_value=(SimpleNamespace(bounds={0: (0, 20)}), {}, {})),
        patch("memvid.demo.notebook.get_embedder", return_value=HashingEmbedder()),
        patch("memvid.demo.notebook.get_token_counter", return_value=ApproxTokenCounter()),
        patch("memvid.demo.notebook.get_llm") as llm,
    ):
        assert demo.ask("What happened?", 0)["status"] == "insufficient_evidence"
        llm.assert_not_called()


def test_separate_support_and_future_padding(demo):
    demo.prepared.write_bytes(b"identity")
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"clip")

    with patch("memvid.demo.notebook.run", side_effect=fake_run):
        clips = demo.clips([(0, 3), (15, 20), (30, 40)], upto_chunk=0)
    assert len(clips) == 2
    assert clips[-1]["interval"][1] == 20
    assert len(calls) == 2


def test_merge_disjoint_intervals():
    assert merge_support([(0, 5), (6, 8), (100, 110)]) == [[0.0, 8.0], [100.0, 110.0]]


def test_long_evidence_is_lazy_and_keeps_all_time(demo):
    with (
        patch.object(demo, "snapshot", return_value=(SimpleNamespace(bounds={0: (0, 130)}), {}, {})),
        patch.object(demo, "_export_clip") as export,
    ):
        clips = demo.clips([(0, 130)], upto_chunk=0, lazy=True)
        assert len(clips) == 5
        assert sum(b - a for a, b in [c["interval"] for c in clips]) == 130
        assert all(c["path"] is None and c["interval"][1] - c["interval"][0] <= 30 for c in clips)
        export.assert_not_called()


def test_profiles(tmp_path):
    s = DemoSettings("v", str(tmp_path))
    local = config_dict(s, tmp_path / "v", tmp_path)
    assert local["asr"]["source"] == "whisper"
    assert local["models"]["vlm"]["device_map"] == "cpu"
    assert local["frames"]["frames_per_chunk"] == 8
    s.profile = "kaggle"
    big = config_dict(s, tmp_path / "v", tmp_path)
    assert big["models"]["vlm"]["device_map"] == "balanced"
    assert big["models"]["llm"]["dtype"] == "float16"


def test_open_relocates_session_without_original_inputs(demo):
    demo.prepared.write_bytes(b"media")
    folder = demo.session / "runs/demo/C3_DEMO"
    folder.mkdir(parents=True)
    shutil.copy2(demo.run_path / "memory.sqlite", folder / "memory.sqlite")
    (demo.session / "session.json").write_text(
        json.dumps({"settings": {"video_path": "missing-original.mp4", "output_dir": "old-machine-output"}}),
        encoding="utf-8",
    )
    (demo.session / "demo.yaml").write_text(
        yaml.safe_dump(config_dict(demo.settings, Path("/old-machine/prepared.mp4"), Path("/old-machine"))),
        encoding="utf-8",
    )
    restored = NotebookDemo.open(demo.session)
    assert restored.cfg.video_path == demo.prepared
    assert restored.snapshot(0)[1]["e1"].attributes["color"] == "red"


def test_widget_question_follows_selected_snapshot(demo):
    widgets = pytest.importorskip("ipywidgets")
    shown = []
    result = {
        "status": "insufficient_evidence",
        "answer": None,
        "retrieved": [],
        "clips": [],
        "scores": {},
        "retrieved_tokens": 0,
    }
    with (
        patch("IPython.display.display", side_effect=lambda obj: shown.append(obj)),
        patch.object(demo, "ask", return_value=result) as ask,
    ):
        demo.display()
        root = next(x for x in shown if isinstance(x, widgets.VBox))
        slider = root.children[0]
        question = next(x for x in root.children if isinstance(x, widgets.Textarea))
        button = next(x for x in root.children if isinstance(x, widgets.Button))
        slider.value = 0
        question.value = "What color was the box?"
        button.click()
        ask.assert_called_once_with(question.value, 0)


def test_local_perception_does_not_mix_caption_chunks(tmp_path):
    from memvid.demo.local_perception import CaptionGroundedModel

    (tmp_path / "visual_captions.jsonl").write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"signature": "test"},
                {"chunk_id": 0, "text": "A red box"},
                {"chunk_id": 1, "text": "A blue car"},
            ]
        ),
        encoding="utf-8",
    )
    model = SimpleNamespace(name="test-text", generate=lambda prompt, **kwargs: prompt)
    adapter = CaptionGroundedModel(SimpleNamespace(derived_dir=tmp_path), model)
    prompt = adapter.generate(
        "### CHUNK\nid: 0; time: 0 to 20\n### TRANSCRIPT\n(no speech)\n### OUTPUT\nExample: red mug filled with coffee"
    )
    assert "A red box" in prompt and "A blue car" not in prompt
    assert "red mug" not in prompt and "coffee" not in prompt
