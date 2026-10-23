"""Run perception over all chunks, audit it, freeze it.

Outputs (data/derived/<video_id>/):
  observations.jsonl          one ChunkObservation per chunk, in chunk order
  observations.partial.jsonl  progress file; an interrupted run resumes from it
  perception_info.json        model, prompt version, pixel budget, failures, timing
  perception_audit/           audit sheet + contact sheets for manual checking
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from memvid._progress import progress_iter
from memvid.config import ProjectConfig
from memvid.errors import MemvidError, PipelineStateError
from memvid.io import iter_jsonl, read_jsonl, write_json, write_jsonl
from memvid.manifest import load_manifest, register_file, save_manifest
from memvid.models.base import GenerativeModel
from memvid.models.jsonutil import generate_validated
from memvid.perception.schema import ChunkObservation, ModelObservation
from memvid.prompts import CATEGORIES, PERCEIVE, PROMPT_VERSION
from memvid.schemas import Chunk, FrameRecord
from memvid.timeutils import format_timestamp

log = logging.getLogger(__name__)


def observations_path(cfg: ProjectConfig) -> Path:
    return cfg.derived_dir / "observations.jsonl"


def resize_to_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)


def build_prompt(chunk: Chunk, n_frames: int, max_chars: int) -> str:
    transcript = chunk.transcript_text.strip()[:max_chars] or "(no speech)"
    return PERCEIVE.format(
        n_frames=n_frames, duration=chunk.duration, categories=", ".join(CATEGORIES),
        chunk_id=chunk.chunk_id, t_start=format_timestamp(chunk.t_start, decimals=0),
        t_end=format_timestamp(chunk.t_end, decimals=0), transcript=transcript,
    )


def _select(frames: list[FrameRecord], k: int | None) -> list[FrameRecord]:
    frames = sorted(frames, key=lambda f: f.frame_index)
    if k is None or k >= len(frames):
        return frames
    idx = [round(i * (len(frames) - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [len(frames) // 2]
    return [frames[i] for i in dict.fromkeys(idx)]


def image_workload_stats(cfg: ProjectConfig, frames: list[FrameRecord]) -> dict[str, Any]:
    """Deterministic input-size accounting for visual-budget experiments."""
    by_chunk: dict[int, list[FrameRecord]] = {}
    for frame in frames:
        by_chunk.setdefault(frame.chunk_id, []).append(frame)
    selected = [f for chunk in by_chunk.values() for f in _select(chunk, cfg.perception.frames_per_chunk)]
    pixels = []
    for frame in selected:
        original = frame.width * frame.height
        pixels.append(min(original, cfg.perception.max_pixels))
    return {
        "selected_frames": len(selected),
        "selected_pixels": int(sum(pixels)),
        "mean_selected_frames_per_chunk": round(len(selected) / max(1, len(by_chunk)), 3),
        "mean_pixels_per_frame": round(sum(pixels) / max(1, len(pixels)), 1),
    }


def _peak_cuda_memory() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / (1024**2), 1)
            for i in range(torch.cuda.device_count())
        }
    except Exception:
        return None


def perceive_chunk(cfg: ProjectConfig, vlm: GenerativeModel, chunk: Chunk,
                   frames: list[FrameRecord]) -> ChunkObservation:
    pc = cfg.perception
    chosen = _select(frames, pc.frames_per_chunk)
    images = [resize_to_pixels(Image.open(cfg.derived_dir / f.path).convert("RGB"), pc.max_pixels)
              for f in chosen]
    prompt = build_prompt(chunk, len(images), pc.max_transcript_chars)
    res = generate_validated(vlm, prompt, ModelObservation, images=images,
                             max_new_tokens=cfg.models.vlm.max_new_tokens,
                             repair_attempts=pc.repair_attempts)
    if res.ok:
        return ChunkObservation.from_model(chunk.chunk_id, chunk.t_start, chunk.t_end, res.value,
                                           res.attempts)
    return ChunkObservation(chunk_id=chunk.chunk_id, t_start=chunk.t_start, t_end=chunk.t_end,
                            ok=False, error=res.error, attempts=res.attempts)


def run_perception(cfg: ProjectConfig, vlm: GenerativeModel, *, force: bool = False,
                   progress: bool = True, limit: int | None = None) -> list[ChunkObservation]:
    d = cfg.derived_dir
    m = load_manifest(d, cfg.video.video_id)
    if "perception" in m.frozen_stages and not force:
        raise MemvidError("perception outputs are frozen; use --force only to invalidate results")
    if "preprocess" not in m.frozen_stages:
        log.warning("Stage 1 outputs are not frozen yet; freeze them before reporting results")
    chunks_p, frames_p = d / "chunks.jsonl", d / "frames" / "index.jsonl"
    if not chunks_p.is_file() or not frames_p.is_file():
        raise PipelineStateError("chunks/frames missing: run `memvid preprocess` first")
    chunks: list[Chunk] = read_jsonl(chunks_p, Chunk)
    frames: list[FrameRecord] = read_jsonl(frames_p, FrameRecord)
    by_chunk: dict[int, list[FrameRecord]] = {}
    for f in frames:
        by_chunk.setdefault(f.chunk_id, []).append(f)

    signature = {"prompt_version": PROMPT_VERSION, "model": vlm.describe(),
                 "perception": cfg.perception.model_dump(mode="json")}
    try:
        import torch

        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass
    partial = d / "observations.partial.jsonl"
    done: dict[int, ChunkObservation] = {}
    if partial.is_file() and not force:
        rows = list(iter_jsonl(partial))
        if rows and rows[0].get("signature") == signature:
            for r in rows[1:]:
                o = ChunkObservation.model_validate(r)
                done[o.chunk_id] = o
            log.info("resuming: %d chunks already perceived", len(done))
    if not done:
        partial.write_text(json.dumps({"signature": signature}) + "\n", encoding="utf-8")

    todo = [c for c in chunks if c.chunk_id not in done]
    if limit is not None:
        todo = todo[:limit]
    t0 = time.perf_counter()
    with open(partial, "a", encoding="utf-8") as fh:
        for c in progress_iter(todo, desc="perception", enabled=progress):
            if c.chunk_id not in by_chunk:
                raise PipelineStateError(f"no frames for chunk {c.chunk_id}")
            obs = perceive_chunk(cfg, vlm, c, by_chunk[c.chunk_id])
            done[obs.chunk_id] = obs
            fh.write(obs.model_dump_json() + "\n")
            fh.flush()
    wall = time.perf_counter() - t0
    if len(done) < len(chunks):
        log.warning("partial run: %d/%d chunks; rerun to continue", len(done), len(chunks))
        return [done[k] for k in sorted(done)]

    obs = [done[c.chunk_id] for c in chunks]
    write_jsonl(observations_path(cfg), obs)
    stats = observation_stats(obs)
    workload = image_workload_stats(cfg, frames)
    info = {
        **signature,
        "wall_time_s_this_run": round(wall, 1),
        "mean_wall_time_s_per_processed_chunk": round(wall / max(1, len(todo)), 3),
        "peak_cuda_memory_mb": _peak_cuda_memory(),
        "workload": workload,
        "stats": stats,
    }
    write_json(d / "perception_info.json", info)
    m = load_manifest(d, cfg.video.video_id)
    h = cfg.section_hash("perception", "models")
    register_file(m, d, "observations", observations_path(cfg), step="perceive", config_hash=h)
    register_file(m, d, "perception_info", d / "perception_info.json", step="perceive", config_hash=h)
    m.stats["perception"] = {**stats, "model": vlm.describe().get("name"),
                             "prompt_version": PROMPT_VERSION,
                             "max_pixels": cfg.perception.max_pixels,
                             **workload,
                             "wall_time_s": round(wall, 1),
                             "peak_cuda_memory_mb": info["peak_cuda_memory_mb"]}
    save_manifest(d, m)
    return obs


def observation_stats(obs: list[ChunkObservation]) -> dict[str, Any]:
    n = len(obs)
    ok = [o for o in obs if o.ok]
    return {
        "n_chunks": n,
        "parse_failures": n - len(ok),
        "parse_failure_rate": round((n - len(ok)) / n, 4) if n else 0.0,
        "repaired": sum(1 for o in ok if o.attempts > 1),
        "mean_entities": round(sum(len(o.entities) for o in ok) / max(1, len(ok)), 2),
        "mean_events": round(sum(len(o.events) for o in ok) / max(1, len(ok)), 2),
        "mean_relations": round(sum(len(o.relations) for o in ok) / max(1, len(ok)), 2),
        "chunks_with_speech_summary": sum(1 for o in ok if o.speech_summary),
        "dropped_refs": sum(o.dropped_refs for o in obs),
        "categories": dict(Counter(e.category for o in ok for e in o.entities).most_common()),
    }


def load_observations(cfg: ProjectConfig, *, require_frozen: bool = False) -> list[ChunkObservation]:
    p = observations_path(cfg)
    if not p.is_file():
        raise PipelineStateError("observations.jsonl missing: run `memvid perceive` first")
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    if require_frozen and "perception" not in m.frozen_stages:
        raise PipelineStateError("observations are not frozen: run `memvid perception-freeze`")
    rec = m.artifacts.get("observations")
    if rec is not None:
        from memvid.io import sha256_file

        if sha256_file(p) != rec.sha256:
            raise MemvidError("observations.jsonl differs from the manifest hash")
    return read_jsonl(p, ChunkObservation)


# ----------------------------------------------------------------------------- audit
AUDIT_FIELDS = ["chunk_id", "time", "contact_sheet", "n_entities", "entities", "n_events", "events",
                "false_entities", "missed_salient_events", "notes"]


def contact_sheet(cfg: ProjectConfig, frames: list[FrameRecord], out: Path, cols: int = 4) -> None:
    ims = [Image.open(cfg.derived_dir / f.path).convert("RGB") for f in sorted(frames, key=lambda f: f.frame_index)]
    w = max(i.width for i in ims)
    h = max(i.height for i in ims)
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (im, f) in enumerate(zip(ims, sorted(frames, key=lambda f: f.frame_index), strict=True)):
        x, y = (k % cols) * w, (k // cols) * (h + 18)
        sheet.paste(im, (x, y + 18))
        draw.text((x + 4, y + 3), format_timestamp(f.t_actual or f.t_target), fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def make_perception_audit(cfg: ProjectConfig) -> Path:
    obs = load_observations(cfg)
    out = cfg.derived_dir / "perception_audit"
    sheet = out / "perception_audit.csv"
    if sheet.exists():
        raise MemvidError(f"{sheet} exists (it may hold your labels); delete it to regenerate")
    frames: list[FrameRecord] = read_jsonl(cfg.derived_dir / "frames" / "index.jsonl", FrameRecord)
    rng = random.Random(cfg.seed + 2)
    sample = sorted(rng.sample(obs, min(cfg.perception.audit_size, len(obs))), key=lambda o: o.chunk_id)
    rows = []
    for o in sample:
        img = out / f"chunk_{o.chunk_id:04d}.jpg"
        contact_sheet(cfg, [f for f in frames if f.chunk_id == o.chunk_id], img)
        rows.append({
            "chunk_id": o.chunk_id,
            "time": f"{format_timestamp(o.t_start)}-{format_timestamp(o.t_end)}",
            "contact_sheet": img.name,
            "n_entities": len(o.entities),
            "entities": " | ".join(e.name for e in o.entities),
            "n_events": len(o.events),
            "events": " | ".join(e.description for e in o.events),
            "false_entities": "", "missed_salient_events": "", "notes": "",
        })
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "INSTRUCTIONS.txt").write_text(
        "For each row, open the contact sheet (and the video at that time if needed) and fill:\n"
        "  false_entities         number of listed entities that are not in the segment\n"
        "  missed_salient_events  number of clearly visible or spoken salient events missing\n"
        "Then run: memvid perception-audit-summary -c <config>\n", encoding="utf-8")
    return sheet


def summarize_perception_audit(cfg: ProjectConfig) -> dict[str, Any]:
    sheet = cfg.derived_dir / "perception_audit" / "perception_audit.csv"
    if not sheet.is_file():
        raise PipelineStateError("run `memvid perception-audit` first")
    with open(sheet, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    try:
        false = [int(r["false_entities"]) for r in rows]
        missed = [int(r["missed_salient_events"]) for r in rows]
    except ValueError as exc:
        raise MemvidError("perception_audit.csv is incomplete or non-integer") from exc
    n_ent = sum(int(r["n_entities"]) for r in rows)
    n_ev = sum(int(r["n_events"]) for r in rows)
    fer = sum(false) / n_ent if n_ent else 0.0
    mer = sum(missed) / (n_ev + sum(missed)) if (n_ev + sum(missed)) else 0.0
    summary = {
        "n_chunks": len(rows), "n_entities": n_ent, "false_entities": sum(false),
        "false_entity_rate": round(fer, 3),
        "missed_salient_events": sum(missed), "missed_event_rate": round(mer, 3),
        "max_false_entity_rate": cfg.perception.max_false_entity_rate,
        "accepted": fer <= cfg.perception.max_false_entity_rate,
    }
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    m.stats["perception_audit"] = summary
    save_manifest(cfg.derived_dir, m)
    return summary


def freeze_perception(cfg: ProjectConfig) -> None:
    from memvid.manifest import verify_manifest

    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    if "observations" not in m.artifacts:
        raise PipelineStateError("no observations to freeze")
    problems = [p for p in verify_manifest(m, cfg.derived_dir, {"frames": "*.jpg"})
                if p.startswith(("observations", "perception_info"))]
    if problems:
        raise MemvidError("cannot freeze: " + "; ".join(problems))
    if not m.stats.get("perception_audit", {}).get("accepted"):
        log.warning("perception audit not accepted (or not done); freezing anyway as requested")
    if "perception" not in m.frozen_stages:
        m.frozen_stages.append("perception")
    save_manifest(cfg.derived_dir, m)
