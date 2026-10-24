"""Notebook orchestration and widgets; reuse the research pipeline and C3 records.

All timestamps in memory refer to prepared.mp4. source_start_s maps them back to
the original input. Models run in separate processes so stage RAM is reclaimed.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from memvid._proc import run
from memvid.config import load_config
from memvid.demo import backend as B
from memvid.io import read_json, write_json
from memvid.memory.answer import adhoc_question, answer_question
from memvid.memory.persist import load_memory
from memvid.memory.retrieval import retrieve
from memvid.memory.store import MemoryStore, parse_record, replay
from memvid.models.base import GenerativeModel
from memvid.models.registry import get_embedder, get_llm, get_token_counter, release_models
from memvid.prompts import sections
from memvid.video.probe import media_duration, probe_video


@dataclass
class DemoSettings:
    video_path: str
    output_dir: str
    audio_path: str = ""
    audio_offset_s: float = 0.0  # positive means delay audio relative to video
    source_start_s: float = 0.0
    duration_s: float | None = 180.0
    profile: str = "local_small"
    budget_tokens: int = 2000

    def validate(self):
        if self.profile not in {"local_small", "kaggle"}:
            raise ValueError("profile must be local_small or kaggle")
        if self.source_start_s < 0 or (self.duration_s is not None and self.duration_s <= 0):
            raise ValueError("Invalid source time range")
        if self.budget_tokens < 50:
            raise ValueError("Memory budget must be at least 50 tokens")
        for p in [self.video_path, *([self.audio_path] if self.audio_path else [])]:
            if not Path(p).expanduser().is_file():
                raise FileNotFoundError(p)
        if Path(self.output_dir).resolve() in {Path(p).resolve() for p in (self.video_path, self.audio_path) if p}:
            raise ValueError("Output must be a directory separate from source files")


class GroundedAnswerModel(GenerativeModel):
    """Demo-only abstention prompt, keeping research/evaluation prompts unchanged."""

    def __init__(self, model):
        self.model = model

    def generate(self, prompt, *, images=None, json_schema=None, max_new_tokens=None):
        blocks = sections(prompt)
        ids = re.findall(r"^\[([^\]]+)\]", blocks.get("MEMORY", ""), re.MULTILINE)
        strict = (
            "Answer the question using only MEMORY. Answer in at most 8 words.\n"
            "For a supported answer cited_record_ids must contain at least one supporting record ID.\n"
            "Allowed record IDs: " + ", ".join(ids) + ". Do not invent IDs.\n"
            "If MEMORY does not establish the answer, return an empty answer and empty cited_record_ids.\n"
            "Return one JSON object with keys answer (string), cited_record_ids (list of strings), "
            "confidence (low, medium or high). No explanations.\n"
            "MEMORY:\n"
            + blocks.get("MEMORY", "")
            + "\nQUESTION:\n"
            + blocks.get("QUESTION", "")
            + '\nOUTPUT JSON (fill these keys):\n{"answer":"","cited_record_ids":[],"confidence":"low"}\n'
            "Respond with the filled JSON object only."
        )
        if "Previous answer:\n" in prompt:
            previous = prompt.split("Previous answer:\n", 1)[1].split("Reply again", 1)[0]
            strict += "\nYour previous reply was not valid JSON. Convert it to the required JSON: " + previous[:500]
        return self.model.generate(strict, json_schema=json_schema, max_new_tokens=max_new_tokens)


def _identity(path):
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def config_dict(settings: DemoSettings, prepared: Path, session: Path):
    local = settings.profile == "local_small"

    def model(name, tokens):
        spec = {
            "backend": "transformers",
            "model_id": name,
            "dtype": "float32" if local else "float16",
            "device_map": "cpu" if local else "balanced",
            "max_new_tokens": tokens,
        }
        if "SmolVLM" in name:
            spec.update(image_max_side=384, image_splitting=False)
        return spec

    return {
        "video": {"video_id": "demo", "filename": str(prepared)},
        "paths": {
            "raw_dir": str(session),
            "derived_dir": str(session / "derived"),
            "runs_dir": str(session / "runs"),
            "annotations_dir": str(session / "annotations"),
        },
        "asr": {
            "source": "whisper",
            "model_size": "tiny.en" if local else "large-v3-turbo",
            "device": "cpu" if local else "cuda",
            "compute_type": "int8" if local else "float16",
            "language": "en",
            "temperature_fallback": False,
        },
        "chunking": {"chunk_seconds": 20},
        "frames": {"frames_per_chunk": 8, "max_side": 384 if local else 768},
        "perception": {"max_pixels": (256 if local else 448) ** 2, "repair_attempts": 1},
        "models": {
            "vlm": model(
                "HuggingFaceTB/SmolVLM-500M-Instruct" if local else "Qwen/Qwen3-VL-4B-Instruct", 512 if local else 1024
            ),
            "llm": model("Qwen/Qwen3-0.6B" if local else "Qwen/Qwen3-VL-4B-Instruct", 512),
            "tokenizer": "hf",
            "embedder": "sentence-transformers",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedder_device": "cpu",
        },
        "budgets": {"mode": "tokens", "values": {"DEMO": settings.budget_tokens}},
        "experiment": {"conditions": ["C3"], "ablations": []},
    }


def merge_support(intervals, *, gap_s=2.0):
    merged = []
    for a, b in sorted(set((float(a), float(b)) for a, b in intervals)):
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + gap_s:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


class NotebookDemo:
    def __init__(self, settings: DemoSettings):
        self.settings = settings
        settings.validate()
        payload = {
            "version": 1,
            "settings": asdict(settings),
            "video": _identity(settings.video_path),
            "audio": _identity(settings.audio_path) if settings.audio_path else None,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        self.session = Path(settings.output_dir).expanduser().resolve() / key
        self.session.mkdir(parents=True, exist_ok=True)
        self.config_path = self.session / "demo.yaml"
        self.prepared = self.session / "prepared.mp4"
        self.cfg = None
        self.run_path = None
        self.log_path = self.session / "processing.log"
        write_json(self.session / "session.json", payload)

    @classmethod
    def open(cls, session_dir):
        """Open saved artifacts without requiring the original media or rerunning models."""
        obj = cls.__new__(cls)
        obj.session = Path(session_dir).expanduser().resolve()
        obj.settings = DemoSettings(**read_json(obj.session / "session.json")["settings"])
        obj.config_path = obj.session / "demo.yaml"
        obj.prepared = obj.session / "prepared.mp4"
        obj.log_path = obj.session / "processing.log"
        obj.cfg = load_config(obj.config_path)
        # A session folder can be copied to Kaggle or another local directory.
        obj.cfg.paths.raw_dir = str(obj.session)
        obj.cfg.paths.derived_dir = str(obj.session / "derived")
        obj.cfg.paths.runs_dir = str(obj.session / "runs")
        obj.cfg.video.filename = str(obj.prepared)
        obj.run_path = obj.cfg.video_runs_dir / "C3_DEMO"
        load_memory(obj.run_path / "memory.sqlite")
        if not obj.prepared.is_file():
            raise FileNotFoundError("prepared.mp4 is required for supporting clips")
        return obj

    def prepare(self):
        """Normalize to browser-playable H264/AAC with a single shared timeline."""
        if not all(os.environ.get(f"MEMVID_{name}") for name in ("FFMPEG", "FFPROBE")):
            from memvid.demo.bootstrap import setup_media_tools

            setup_media_tools(self.session.parent / ".cache" / "media-tools")
        for name in ("ffmpeg", "ffprobe"):
            run([name, "-version"])
        s = self.settings
        info = probe_video(Path(s.video_path), compute_hash=False)
        if s.source_start_s >= info.duration_s:
            raise ValueError("Start time is beyond video duration")
        duration = min(s.duration_s or info.duration_s, info.duration_s - s.source_start_s)
        if s.audio_path:
            audio_duration = media_duration(s.audio_path)
            if s.source_start_s + duration > audio_duration + s.audio_offset_s + 0.5:
                print("Warning: audio is shorter than the selected video; the end will be padded with silence.")
        elif not info.has_audio:
            raise ValueError("Video has no audio. Supply a separate audio_path; Whisper requires speech audio.")
        if not self.prepared.exists():
            tmp = self.session / "prepared.partial.mp4"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2"]
            # Some Windows builds select a hardware-only native AV1 decoder.
            # libaom is a software decoder and also runs on old GPUs/CPU hosts.
            if info.video_codec == "av1":
                decoders = run(["ffmpeg", "-hide_banner", "-decoders"]).stdout
                cmd += ["-c:v", "libdav1d" if "libdav1d" in decoders else "libaom-av1"]
            cmd += ["-i", s.video_path]
            if s.audio_path:
                cmd += ["-i", s.audio_path]
            audio_index = 1 if s.audio_path else 0
            # Audio sample at video time t comes from audio time t-offset.
            if s.audio_offset_s >= 0:
                af = f"adelay={s.audio_offset_s * 1000:.3f}:all=1,apad"
            else:
                af = f"atrim=start={-s.audio_offset_s:.6f},asetpts=PTS-STARTPTS,apad"
            cmd += [
                "-filter_complex",
                f"[{audio_index}:a:0]{af}[audio]",
                "-map",
                "0:v:0",
                "-map",
                "[audio]",
                "-ss",
                str(s.source_start_s),
                "-t",
                str(duration),
                "-vf",
                "scale='min(1280,iw)':-2",
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-threads",
                "2",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-aac_coder",
                "fast",
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                str(tmp),
            ]
            run(cmd)
            tmp.replace(self.prepared)
        prepared_info = probe_video(self.prepared, compute_hash=False)
        if not prepared_info.has_audio or abs(prepared_info.duration_s - duration) > 0.6:
            raise ValueError("Prepared media failed duration/audio validation")
        self.config_path.write_text(
            yaml.safe_dump(config_dict(s, self.prepared, self.session), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.cfg = load_config(self.config_path)
        self.run_path = self.cfg.video_runs_dir / "C3_DEMO"
        return self.prepared

    def _stage(self, name):
        marker = self.session / f"{name}.complete"
        from memvid.prompts import PROMPT_VERSION

        signature = hashlib.sha256(
            json.dumps(
                {
                    "config": self.cfg.model_dump(mode="json"),
                    "media": _identity(self.prepared),
                    "prompt": PROMPT_VERSION,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == signature:
            print(f"{name}: cached stage found")
            return
        print(f"{name}: running; log: {self.log_path}", flush=True)
        # A subprocess also frees Whisper/CUDA allocations reliably on Windows.
        with self.log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "memvid.demo.notebook", name, str(self.config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                while True:
                    try:
                        code = proc.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if lines:
                            print(lines[-1][-240:], flush=True)
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
        if code:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"{name} failed. See {self.log_path}\n{tail}")
        marker.write_text(signature + "\n", encoding="utf-8")
        print(f"{name}: complete", flush=True)

    def build(self):
        if self.cfg is None:
            self.prepare()
        stages = ["preprocess"]
        if self.settings.profile == "local_small":
            stages.append("caption")
        for stage in [*stages, "perceive", "memory"]:
            self._stage(stage)
        B._load.cache_clear()
        return self.run_path

    def snapshot(self, chunk):
        saved = load_memory(self.run_path / "memory.sqlite")
        if chunk not in saved.bounds:
            raise ValueError("Unknown snapshot")
        records = replay(saved.ops, upto_chunk=chunk)
        size = next(s for s in saved.sizes if s["chunk_id"] == chunk)
        return saved, records, size

    def _export_clip(self, a, b):
        key = hashlib.sha256(f"{_identity(self.prepared)}:{a}:{b}".encode()).hexdigest()[:16]
        out = self.session / "clips" / f"{key}.mp4"
        if not out.exists():
            out.parent.mkdir(exist_ok=True)
            tmp = out.with_suffix(".partial.mp4")
            run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-threads",
                    "2",
                    "-ss",
                    str(a),
                    "-i",
                    str(self.prepared),
                    "-t",
                    str(b - a),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-aac_coder",
                    "fast",
                    "-movflags",
                    "+faststart",
                    str(tmp),
                ]
            )
            tmp.replace(out)
        return str(out)

    def preview(self):
        """A short embedded preview avoids embedding the whole long video in notebook JSON."""
        return self._export_clip(0.0, min(10.0, media_duration(self.prepared)))

    def materialize_clip(self, item):
        return {**item, "path": self._export_clip(*item["interval"])}

    def clips(self, intervals, *, upto_chunk, lazy=False):
        """Preserve all support, split long spans, and optionally defer encoding to selection."""
        saved, _, _ = self.snapshot(upto_chunk)
        visible_end = saved.bounds[upto_chunk][1]
        clipped = [(max(0, a - 1), min(visible_end, b + 1)) for a, b in intervals]
        paths = []
        for start, end in merge_support(clipped):
            a = start
            while a < end:
                b = min(end, a + 30.0)
                paths.append(
                    {
                        "path": None if lazy else self._export_clip(a, b),
                        "interval": [a, b],
                        "source_interval": [a + self.settings.source_start_s, b + self.settings.source_start_s],
                    }
                )
                a = b
        return paths

    def changes(self, chunk):
        saved, _, _ = self.snapshot(chunk)
        before = replay(saved.ops, upto_chunk=chunk - 1)
        rows = []
        for op in saved.ops:
            if op["chunk_id"] != chunk:
                continue
            rid = op["record_id"]
            old = before.get(rid)
            new = parse_record(op["record"]) if op["op"] == "upsert" else None
            rows.append([op["seq"], op["op"], rid, op["reason"], old.text() if old else "", new.text() if new else ""])
            if new:
                before[rid] = new
            else:
                before.pop(rid, None)
        return rows

    def ask(self, question, chunk):
        if not question.strip():
            raise ValueError("Enter a question")
        saved, records, _ = self.snapshot(chunk)
        bounds = {c: iv for c, iv in saved.bounds.items() if c <= chunk}
        try:
            store = MemoryStore.from_records(records, get_embedder(self.cfg), get_token_counter(self.cfg))
            ret = retrieve(store, question, self.cfg.retrieval, bounds[chunk][1])
            if not ret.records:
                return {
                    "status": "insufficient_evidence",
                    "answer": None,
                    "retrieved": [],
                    "clips": [],
                    "scores": {},
                    "retrieved_tokens": 0,
                }
            # Demo-only instruction: keep evaluation prompts unchanged.
            q = adhoc_question(
                question + "\nUse only the supplied memory. If it does not establish the answer, "
                "return an empty answer and no cited_record_ids. Do not guess."
            )
            ans = answer_question(q, ret, GroundedAnswerModel(get_llm(self.cfg)), bounds)
            status = (
                "model_error"
                if not ans.ok
                else "invalid_citations"
                if ans.invalid_citations
                else "answered"
                if ans.answer and ans.cited_ids
                else "insufficient_evidence"
            )
            clips = self.clips(ans.cited_intervals, upto_chunk=chunk, lazy=True) if status == "answered" else []
            result = {
                "status": status,
                "answer": ans,
                "retrieved": ret.records,
                "clips": clips,
                "scores": ret.scores,
                "retrieved_tokens": ret.tokens,
            }
            with (self.session / "questions.jsonl").open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "question": question,
                            "chunk": chunk,
                            "status": status,
                            "answer": ans.model_dump(mode="json"),
                            "clips": clips,
                        }
                    )
                    + "\n"
                )
            return result
        finally:
            release_models()

    def display(self):
        import ipywidgets as w
        import pandas as pd
        from IPython.display import HTML, Video, clear_output, display

        saved = load_memory(self.run_path / "memory.sqlite")
        ids = sorted(saved.bounds)
        if not ids:
            raise ValueError("No processed chunks available")
        display(
            HTML(
                "<details><summary>Record ID legend</summary>"
                "<p>E: entity; V: event (including speech and compressed summaries); "
                "S: state change; R: relation; N: note. Numbers are per-type creation counters, "
                "not timestamps. V is not visual-only; S does not mean speech.</p></details>"
            )
        )
        slider = w.SelectionSlider(options=ids, value=ids[-1], description="Chunk", continuous_update=False)
        kind = w.Dropdown(
            options=["all", "entity", "event", "state_change", "relation", "note"],
            value="all",
            description="Record type",
        )
        memory_out, answer_out, record_out = w.Output(), w.Output(), w.Output()
        question = w.Textarea(description="Question", placeholder="Ask in English", layout=w.Layout(width="95%"))
        go = w.Button(description="Answer from memory", button_style="primary")
        source = w.Dropdown(description="Record", options=[])
        open_record = w.Button(description="Open source")

        def show_video(item):
            item = self.materialize_clip(item)
            display(
                HTML(
                    f"<p>Original source time: {item['source_interval'][0]:.1f}–{item['source_interval'][1]:.1f} s</p>"
                )
            )
            display(Video(item["path"], embed=True, html_attributes='controls preload="metadata"'))

        def show_clips(items):
            if not items:
                return
            selector = w.Dropdown(
                options=[
                    (f"{i + 1}: {v['source_interval'][0]:.1f}–{v['source_interval'][1]:.1f} s", i)
                    for i, v in enumerate(items)
                ],
                description="Clip",
            )
            player = w.Output()

            def play(*_):
                with player:
                    clear_output(wait=True)
                    try:
                        show_video(items[selector.value])
                    except Exception as exc:
                        print(f"Clip error: {exc}")

            selector.observe(play, names="value")
            display(w.VBox([selector, player]))
            play()

        def refresh(*_):
            with answer_out:
                clear_output()
            with record_out:
                clear_output()
            with memory_out:
                clear_output(wait=True)
                state, records, size = self.snapshot(slider.value)
                end = state.bounds[slider.value][1]
                display(
                    HTML(
                        f"<b>Memory after {end:.1f} s: {size['tokens']} / "
                        f"{state.meta.get('budget_tokens')} tokens; {len(records)} records</b>"
                    )
                )
                visible = [r for r in records.values() if kind.value == "all" or r.kind == kind.value]
                display(pd.DataFrame(B.record_rows(visible), columns=B.RECORD_HEADERS))
                caption_file = self.cfg.derived_dir / "visual_captions.jsonl"
                if caption_file.is_file():
                    for line in caption_file.read_text(encoding="utf-8").splitlines()[1:]:
                        caption = json.loads(line)
                        if caption["chunk_id"] == slider.value:
                            display(
                                HTML(
                                    "<details><summary>Raw SmolVLM frame description</summary><pre>"
                                    + html.escape(caption["text"])
                                    + "</pre></details>"
                                )
                            )
                            break
                observation_file = self.cfg.derived_dir / "observations.jsonl"
                if observation_file.is_file():
                    observation = next(
                        (
                            json.loads(line)
                            for line in observation_file.read_text(encoding="utf-8").splitlines()
                            if json.loads(line)["chunk_id"] == slider.value
                        ),
                        None,
                    )
                    if observation:
                        print(
                            "Perception provenance:",
                            {key: observation.get(key) for key in ("ok", "fallback", "rescue_used", "error")},
                        )
                        display(
                            HTML(
                                "<details><summary>Observation before memory insertion</summary><pre>"
                                + html.escape(json.dumps(observation, ensure_ascii=False, indent=2))
                                + "</pre></details>"
                            )
                        )
                display(HTML("<b>Changes at this step</b>"))
                display(
                    pd.DataFrame(
                        self.changes(slider.value), columns=["seq", "operation", "record", "reason", "before", "after"]
                    )
                )
                source.options = [(f"{r.id}: {r.kind}", r.id) for r in records.values()]
                display(pd.DataFrame(state.sizes).set_index("chunk_id").loc[: slider.value])

        def answer(_):
            go.disabled = True
            with answer_out:
                clear_output(wait=True)
                print("Retrieving records and generating an answer…")
                try:
                    result = self.ask(question.value, slider.value)
                    print("Status:", result["status"])
                    ans = result["answer"]
                    if result["status"] == "answered":
                        display(HTML(f"<p><b>{html.escape(ans.answer)}</b></p>"))
                    if ans:
                        print(
                            "Confidence:",
                            ans.confidence,
                            "Citations:",
                            ans.cited_ids,
                            "Invalid:",
                            ans.invalid_citations,
                            "Error:",
                            ans.error,
                        )
                    table = pd.DataFrame(B.record_rows(result["retrieved"]), columns=B.RECORD_HEADERS)
                    table["retrieval score"] = [result["scores"].get(r.id) for r in result["retrieved"]]
                    table["cited"] = [bool(ans and r.id in ans.cited_ids) for r in result["retrieved"]]
                    display(table)
                    print("Retrieved tokens:", result["retrieved_tokens"])
                    print("Clips show cited-record provenance; they do not guarantee that the answer is correct.")
                    show_clips(result["clips"])
                except Exception as exc:
                    print(f"Error: {exc}")
                finally:
                    go.disabled = False

        def open_source(_):
            with record_out:
                clear_output(wait=True)
                try:
                    state, records, _ = self.snapshot(slider.value)
                    r = records.get(source.value)
                    if r:
                        show_clips(
                            self.clips(
                                [state.bounds[c] for c in r.source_chunks if c <= slider.value],
                                upto_chunk=slider.value,
                                lazy=True,
                            )
                        )
                except Exception as exc:
                    print(f"Error: {exc}")

        slider.observe(refresh, names="value")
        kind.observe(refresh, names="value")
        go.on_click(answer)
        open_record.on_click(open_source)
        display(w.VBox([slider, kind, memory_out, w.HBox([source, open_record]), record_out, question, go, answer_out]))
        refresh()


def _worker(stage, config):
    from memvid.pipeline import Pipeline

    cfg = load_config(config)
    if stage == "preprocess":
        Pipeline(cfg).preprocess()
    elif stage == "caption":
        from memvid.demo.local_perception import caption_frames
        from memvid.models.registry import get_vlm

        caption_frames(cfg, get_vlm(cfg))
    elif stage == "perceive":
        from memvid.models.registry import get_vlm
        from memvid.perception.run import run_perception

        if "SmolVLM" in cfg.models.vlm.model_id:
            from memvid.demo.local_perception import CaptionGroundedModel

            model = CaptionGroundedModel(cfg, get_llm(cfg))
        else:
            model = get_vlm(cfg)
        obs = run_perception(cfg, model)
        bad = [o.chunk_id for o in obs if not o.ok or o.fallback == "transcript_after_vlm_failure"]
        if bad:
            raise RuntimeError(
                f"Visual perception failed for chunks {bad}; inspect observations and processing.log. "
                "No memory was built from transcript-only fallbacks."
            )
    elif stage == "memory":
        from memvid.experiment import build_memory

        build_memory(cfg, "C3", "DEMO")
    else:
        raise ValueError(stage)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    _worker(*sys.argv[1:])
