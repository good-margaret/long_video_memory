"""Demo logic, independent of the UI toolkit (usable without Gradio).

* memory_at(run, chunk)      records after a chunk, by replaying the operation log
* ops_at(run, chunk)         the operations performed while processing that chunk
* timeline_figure(...)       records drawn on the video time axis
* ask(run, question)         live retrieval + answering on the final memory
* clip(start, end)           supporting clip cut with ffmpeg (cached)
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from memvid._proc import run as run_cmd
from memvid.config import ProjectConfig
from memvid.experiment import list_runs
from memvid.memory.answer import adhoc_question, answer_question
from memvid.memory.persist import SavedMemory, load_memory
from memvid.memory.records import RecordT, record_span
from memvid.memory.retrieval import retrieve
from memvid.memory.store import MemoryStore, replay
from memvid.memory.systems import parse_condition
from memvid.models.registry import get_embedder, get_llm, get_token_counter
from memvid.pipeline import video_duration
from memvid.timeutils import format_timestamp

KIND_COLORS = {"entity": "#2ca02c", "event": "#1f77b4", "state_change": "#d62728",
               "relation": "#9467bd", "note": "#7f7f7f"}


@lru_cache(maxsize=16)
def _load(path: str) -> SavedMemory:
    return load_memory(Path(path) / "memory.sqlite")


def runs(cfg: ProjectConfig) -> list[dict[str, Any]]:
    return list_runs(cfg)


def n_chunks(run_path: str) -> int:
    return len(_load(run_path).bounds)


def chunk_bounds(run_path: str, chunk: int) -> tuple[float, float]:
    return _load(run_path).bounds.get(chunk, (0.0, 0.0))


def record_rows(records: list[RecordT]) -> list[list[Any]]:
    rows = []
    for r in records:
        a, b = record_span(r)
        rows.append([r.id, r.kind, format_timestamp(a, decimals=0), format_timestamp(b, decimals=0),
                     r.text(), " ".join(map(str, r.source_chunks))])
    return rows


RECORD_HEADERS = ["id", "kind", "start", "end", "content", "source chunks"]
OP_HEADERS = ["seq", "op", "label", "record", "reason"]


def memory_at(run_path: str, chunk: int) -> dict[str, list[RecordT]]:
    state = replay(_load(run_path).ops, upto_chunk=chunk)
    out: dict[str, list[RecordT]] = {}
    for r in state.values():
        out.setdefault(r.kind, []).append(r)
    return out


def ops_at(run_path: str, chunk: int) -> list[list[Any]]:
    return [[o["seq"], o["op"], o["label"], o["record_id"], o["reason"]]
            for o in _load(run_path).ops if o["chunk_id"] == chunk]


def timeline_figure(run_path: str, chunk: int, duration: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = memory_at(run_path, chunk)
    kinds = [k for k in ("entity", "event", "state_change", "relation", "note") if k in groups]
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.35 * sum(len(groups[k]) for k in kinds) / 4 + 1.5))
    y = 0
    ticks, labels = [], []
    for k in kinds:
        for r in sorted(groups[k], key=lambda r: record_span(r)[0]):
            a, b = record_span(r)
            ax.barh(y, max(b - a, duration / 400), left=a, height=0.8, color=KIND_COLORS[k],
                    alpha=0.5 if getattr(r, "compressed", False) else 0.9)
            y += 1
        ticks.append(y - len(groups[k]) / 2 - 0.5)
        labels.append(f"{k} ({len(groups[k])})")
    ax.set_yticks(ticks, labels)
    ax.set_xlim(0, duration)
    ax.axvline(chunk_bounds(run_path, chunk)[1], color="k", lw=0.8, ls="--")
    ax.set_xlabel("video time (s)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def ask(cfg: ProjectConfig, run_path: str, question: str) -> dict[str, Any]:
    saved = _load(run_path)
    base, _ = parse_condition(saved.meta["condition"])
    store = MemoryStore.from_records(saved.records, get_embedder(cfg), get_token_counter(cfg))
    q = adhoc_question(question)
    ret = retrieve(store, question, cfg.retrieval, video_duration(cfg), whole_memory=(base == "C1"))
    ans = answer_question(q, ret, get_llm(cfg), saved.bounds)
    return {"answer": ans, "retrieved": ret.records, "cited": set(ans.cited_ids)}


def clip(cfg: ProjectConfig, start: float, end: float, pad: float = 2.0) -> Path:
    a = max(0.0, start - pad)
    b = min(video_duration(cfg), end + pad)
    key = hashlib.sha1(f"{cfg.video.video_id}:{a:.2f}:{b:.2f}".encode()).hexdigest()[:12]
    out = cfg.video_runs_dir / "clips" / f"clip_{key}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        # re-encode (not -c copy) so the cut is exact rather than snapped to keyframes
        run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.3f}",
                 "-to", f"{b:.3f}", "-i", str(cfg.video_path), "-c:v", "libx264",
                 "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart", str(out)])
    return out


def clip_for_record(cfg: ProjectConfig, run_path: str, record_id: str) -> Path:
    saved = _load(run_path)
    r = saved.records[record_id]
    ivs = sorted(saved.bounds[c] for c in r.source_chunks)
    start, end = ivs[0][0], ivs[-1][1]
    if end - start > 120:  # long support (merged or compressed record): show its first chunk
        end = ivs[0][1]
    return clip(cfg, start, end)
