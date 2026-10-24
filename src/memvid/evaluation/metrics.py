"""Per-question evaluation of one run (accuracy, grounding, retrieval, survival).

Definitions (report, Section "Problem formulation"):
  correct       answer scored against the reference (scoring.py)
  grounded      I(cited) intersects G(q)                         (C2-C4 only)
  g_precision   |I(cited) n G(q)| / |I(cited)|  (0 without citations; penalises citing
                long, vague records whose support covers much more than the evidence)
  tiou          |I(cited) n G(q)| / |I(cited) u G(q)|            (C2-C4 only)
  retrieved     I(retrieved) intersects G(q)                     (C2-C4 only)
  survived      some non-entity record of the final memory has support intersecting
                G(q) (C2-C4). A provenance-based PROXY: it shows that material from the
                evidence chunks is still stored, not that the specific fact is. Entity
                records are excluded because their provenance spans every appearance.
For C1 (one record whose provenance is every chunk) these are trivially true and are
reported as missing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from memvid.config import ProjectConfig
from memvid.evaluation.scoring import normalize_text, score_answer
from memvid.evaluation.stats import bootstrap_ci
from memvid.experiment import load_answers
from memvid.io import read_json, write_json, write_jsonl
from memvid.memory.persist import load_memory
from memvid.memory.records import Entity, Event, RecordT, record_overlaps
from memvid.memory.systems import parse_condition
from memvid.models.base import GenerativeModel
from memvid.pipeline import video_duration
from memvid.questions.manage import load_questions
from memvid.timeutils import merge_intervals, overlap_length, tiou, total_length


def interval_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float | None:
    """Minimum gap in seconds; zero means that the interval unions overlap."""
    aa, bb = merge_intervals(a), merge_intervals(b)
    if not aa or not bb:
        return None
    return min(max(0.0, c - y, x - d) for x, y in aa for c, d in bb)


class JudgeCache:
    """Caches LLM-judge verdicts and applies manual overrides
    (data/annotations/<video_id>/manual_scores.csv: qid, answer, correct)."""

    def __init__(self, cfg: ProjectConfig):
        self.path = cfg.video_runs_dir / "judge_cache.json"
        self.cache: dict[str, bool] = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.manual: dict[str, bool] = {}
        mpath = cfg.annotations_dir / "manual_scores.csv"
        if mpath.is_file():
            with open(mpath, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    correct = r["correct"].strip().lower() in ("y", "yes", "1", "true")
                    self.manual[self.manual_key(r["qid"], r["answer"])] = correct

    @staticmethod
    def key(qid: str, answer: str) -> str:
        # Include the rubric version so stale verdicts are not silently reused after the
        # judge prompt changes.
        return f"strict-v2::{qid}::{normalize_text(answer)}"

    @staticmethod
    def manual_key(qid: str, answer: str) -> str:
        return f"{qid}::{normalize_text(answer)}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))


def _specific_evidence_record(record: RecordT) -> bool:
    """Whether temporal provenance still refers to the record's own concrete fact.

    Entity provenance grows across every appearance and compressed summaries inherit the
    union of their inputs, so neither can support fact-level retrieval metrics reliably.
    """
    return not isinstance(record, Entity) and not (
        isinstance(record, Event) and record.summarized
    )


def _support(records: list[RecordT], bounds: dict[int, tuple[float, float]]) -> list[tuple[float, float]]:
    return merge_intervals([
        bounds[c]
        for record in records
        for c in record.source_chunks
        if c in bounds
    ])


def evaluate_run(cfg: ProjectConfig, rdir: Path, split: str, *, judge: GenerativeModel | None = None,
                 cache: JudgeCache | None = None) -> list[dict[str, Any]]:
    info = read_json(rdir / "run_info.json")
    base, ablation = parse_condition(info["condition"])
    saved = load_memory(rdir / "memory.sqlite")
    answers = {a.qid: a for a in load_answers(rdir, split)}
    duration = video_duration(cfg)
    cache = cache or JudgeCache(cfg)
    rows = []
    for q in load_questions(cfg, split):
        a = answers.get(q.id)
        if a is None:
            raise ValueError(f"{rdir.name}: no answer for {q.id}")
        key = JudgeCache.key(q.id, a.answer)
        manual_key = JudgeCache.manual_key(q.id, a.answer)
        if manual_key in cache.manual:
            correct, method = cache.manual[manual_key], "manual"
        else:
            s = score_answer(a.answer, q.answer, q.answer_type, accepted=q.accepted_answers,
                             options=q.options, rel_tol=cfg.questions.numeric_rel_tol, judge=None,
                             question=q.question)
            correct, method = s.correct, s.method
            if not correct and method == "none" and judge is not None:
                if key not in cache.cache:
                    cache.cache[key] = score_answer(a.answer, q.answer, q.answer_type,
                                                    accepted=q.accepted_answers, judge=judge,
                                                    question=q.question).correct
                correct, method = cache.cache[key], "judge"
        gt = list(q.gt_intervals)
        per_record = base != "C1"
        ranked_ids = a.retrieval_ranked_ids or a.retrieved_ids
        provenance_ranks = [
            rank for rank, rid in enumerate(ranked_ids, start=1)
            if rid in saved.records and record_overlaps(saved.records[rid], saved.bounds, gt)
        ]
        relevant_ranks = [
            rank for rank, rid in enumerate(ranked_ids, start=1)
            if rid in saved.records
            and _specific_evidence_record(saved.records[rid])
            and record_overlaps(saved.records[rid], saved.bounds, gt)
        ]
        first_rank = min(relevant_ranks, default=None)
        retrieved_records = [saved.records[rid] for rid in a.retrieved_ids if rid in saved.records]
        cited_records = [saved.records[rid] for rid in a.cited_ids if rid in saved.records]
        specific_retrieved = [r for r in retrieved_records if _specific_evidence_record(r)]
        specific_cited = [r for r in cited_records if _specific_evidence_record(r)]
        strict_retrieved_intervals = _support(specific_retrieved, saved.bounds)
        strict_cited_intervals = _support(specific_cited, saved.bounds)
        temporal_error = interval_distance(strict_retrieved_intervals, gt) if per_record else None
        rows.append({
            "run": rdir.name, "condition": info["condition"], "base": base, "ablation": ablation,
            "budget": info["budget_name"], "budget_tokens": info["budget_tokens"],
            "qid": q.id, "type": q.type, "answer_type": q.answer_type, "modality": q.modality,
            "age_s": q.evidence_age(duration), "reference": q.answer, "answer": a.answer,
            "correct": bool(correct), "method": method, "confidence": a.confidence,
            "n_cited": len(a.cited_ids), "invalid_citations": len(a.invalid_citations),
            "retrieved_tokens": a.retrieved_tokens, "answer_ok": a.ok,
            "grounded": (overlap_length(strict_cited_intervals, gt) > 0) if per_record else None,
            "g_precision": (overlap_length(strict_cited_intervals, gt) /
                            total_length(strict_cited_intervals)
                            if strict_cited_intervals else 0.0) if per_record else None,
            "tiou": tiou(strict_cited_intervals, gt) if per_record else None,
            "retrieved": (overlap_length(strict_retrieved_intervals, gt) > 0) if per_record else None,
            "recall_at_1": (any(rank <= 1 for rank in relevant_ranks)) if per_record else None,
            "recall_at_5": (any(rank <= 5 for rank in relevant_ranks)) if per_record else None,
            "recall_at_10": (any(rank <= 10 for rank in relevant_ranks)) if per_record else None,
            "first_relevant_rank": first_rank if per_record else None,
            "reciprocal_rank": (1.0 / first_rank if first_rank else 0.0) if per_record else None,
            "temporal_error_s": temporal_error,
            "temporal_hit_20s": (temporal_error <= 20) if temporal_error is not None else None,
            "temporal_hit_60s": (temporal_error <= 60) if temporal_error is not None else None,
            "survived": any(record_overlaps(r, saved.bounds, gt) for r in saved.records.values()
                            if _specific_evidence_record(r)) if per_record else None,
            # broad provenance diagnostics are retained under explicit names; they are not
            # evidence that the answer-bearing fact survived compression
            "provenance_retrieved": (
                overlap_length(a.retrieved_intervals, gt) > 0
            ) if per_record else None,
            "provenance_recall_at_5": (
                any(rank <= 5 for rank in provenance_ranks)
            ) if per_record else None,
            "provenance_survived": any(
                record_overlaps(r, saved.bounds, gt)
                for r in saved.records.values()
                if not isinstance(r, Entity)
            ) if per_record else None,
            "heuristic_models": info.get("heuristic_models", False),
        })
    cache.save()
    write_jsonl(rdir / f"eval_{split}.jsonl", rows)
    write_json(rdir / f"eval_{split}_summary.json", summarize_rows(rows, cfg.analysis.n_bootstrap,
                                                                 cfg.seed))
    return rows


def _mean_ci(vals: list[float], n_boot: int, seed: int) -> dict[str, float] | None:
    if not vals:
        return None
    m, lo, hi = bootstrap_ci(vals, n=n_boot, seed=seed)
    return {"mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(vals)}


def summarize_rows(rows: list[dict[str, Any]], n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    def col(name, subset=None):
        return [float(r[name]) for r in (subset or rows) if r.get(name) is not None]

    out: dict[str, Any] = {
        "accuracy": _mean_ci(col("correct"), n_boot, seed),
        "grounded": _mean_ci(col("grounded"), n_boot, seed),
        "g_precision": _mean_ci(col("g_precision"), n_boot, seed),
        "tiou": _mean_ci(col("tiou"), n_boot, seed),
        "retrieved": _mean_ci(col("retrieved"), n_boot, seed),
        "recall_at_1": _mean_ci(col("recall_at_1"), n_boot, seed),
        "recall_at_5": _mean_ci(col("recall_at_5"), n_boot, seed),
        "recall_at_10": _mean_ci(col("recall_at_10"), n_boot, seed),
        "mrr": _mean_ci(col("reciprocal_rank"), n_boot, seed),
        "temporal_error_s": _mean_ci(col("temporal_error_s"), n_boot, seed),
        "temporal_hit_20s": _mean_ci(col("temporal_hit_20s"), n_boot, seed),
        "temporal_hit_60s": _mean_ci(col("temporal_hit_60s"), n_boot, seed),
        "survived": _mean_ci(col("survived"), n_boot, seed),
        "provenance_retrieved": _mean_ci(col("provenance_retrieved"), n_boot, seed),
        "provenance_recall_at_5": _mean_ci(col("provenance_recall_at_5"), n_boot, seed),
        "provenance_survived": _mean_ci(col("provenance_survived"), n_boot, seed),
        "by_type": {},
        "by_modality": {},
    }
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        out["by_type"][t] = _mean_ci([float(r["correct"]) for r in sub], n_boot, seed)
    for modality in sorted({r["modality"] for r in rows}):
        sub = [r for r in rows if r["modality"] == modality]
        out["by_modality"][modality] = _mean_ci(
            [float(r["correct"]) for r in sub], n_boot, seed
        )
    return out
