"""Resumable query-adaptive episodic-memory experiment for one GPU worker."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from memvid.config import load_config
from memvid.episodic import (
    EpisodicRetriever,
    answer_from_selection,
    attach_visual_evidence,
    critique_evidence,
    inspect_visual_evidence,
)
from memvid.evaluation.metrics import summarize_rows
from memvid.evaluation.scoring import score_answer
from memvid.io import iter_jsonl, sha256_file, write_json, write_jsonl
from memvid.models.registry import get_embedder, get_llm, get_token_counter, get_vlm
from memvid.pipeline import Pipeline
from memvid.questions.manage import load_question_file, load_split

LOG = logging.getLogger("episodic_experiment")
DEV_METHODS = ("archive", "c3", "hybrid")


def experiment_questions(cfg, split: str):
    """Load the versioned question file without coupling it to an older perception manifest.

    Perception and memory are question-independent.  The exact question and split hashes are
    logged for this experiment, so a later question revision cannot be mixed silently.
    """
    questions = load_question_file(cfg.questions_path).questions
    assignment = load_split(cfg)
    missing = [q.id for q in questions if q.id not in assignment]
    if missing:
        raise RuntimeError(f"questions missing from split.json: {missing}")
    return questions if split == "all" else [q for q in questions if assignment[q.id] == split]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--memory-run", required=True, type=Path)
    p.add_argument("--output", default="runs/episodic_rehydration", type=Path)
    p.add_argument("--phase", choices=("dev", "select", "test", "aggregate"), required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--answer-budget", type=int, default=1500)
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--visual-chunks", type=int, default=2)
    p.add_argument("--frames-per-chunk", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", force=True)


def completed(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {(row["condition"], row["qid"]) for row in iter_jsonl(path)}


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def eta(start: float, done: int, total: int) -> str:
    if done <= 0:
        return "unknown"
    remaining = (time.perf_counter() - start) / done * max(0, total - done)
    return f"{remaining / 60:.1f} min"


def deterministic_score(question, answer, judge=None) -> tuple[bool, str]:
    result = score_answer(answer, question.answer, question.answer_type,
                          accepted=question.accepted_answers, options=question.options,
                          judge=judge, question=question.question)
    return result.correct, result.method


def build_retriever(args, cfg):
    return EpisodicRetriever(
        cfg, args.memory_run, get_embedder(cfg), get_token_counter(cfg),
        answer_budget_tokens=args.answer_budget, max_chunks=args.max_chunks,
    )


def run_dev(args, cfg) -> None:
    out = args.output / f"dev_shard{args.shard_index}.jsonl"
    if args.force and out.exists():
        out.unlink()
    done = completed(out)
    questions = experiment_questions(cfg, "dev")
    questions = [q for i, q in enumerate(questions) if i % args.num_shards == args.shard_index]
    tasks = [(method, q) for method in DEV_METHODS for q in questions
             if (method, q.id) not in done]
    LOG.info("DEV START shard=%d/%d pending=%d methods=%s", args.shard_index,
             args.num_shards, len(tasks), ",".join(DEV_METHODS))
    retriever = build_retriever(args, cfg)
    llm = get_llm(cfg)
    start = time.perf_counter()
    for index, (method, question) in enumerate(tasks, start=1):
        t0 = time.perf_counter()
        selection = retriever.select(question.question, method)
        answer = answer_from_selection(question, selection, retriever, llm)
        correct, scoring = deterministic_score(question, answer.answer, judge=llm)
        append(out, {
            "phase": "dev", "condition": method, "qid": question.id,
            "answer": answer.model_dump(mode="json"), "correct": correct,
            "scoring": scoring, "chunk_ids": selection.ranked_chunk_ids,
            "tokens": selection.tokens, "wall_time_s": round(time.perf_counter() - t0, 2),
        })
        LOG.info("DEV %d/%d method=%s qid=%s correct=%s chunks=%s step=%.1fs ETA=%s",
                 index, len(tasks), method, question.id, correct,
                 selection.ranked_chunk_ids, time.perf_counter() - t0,
                 eta(start, index, len(tasks)))
    LOG.info("DEV COMPLETE shard=%d rows=%d elapsed=%.1fmin", args.shard_index,
             len(tasks), (time.perf_counter() - start) / 60)


def select_method(args) -> None:
    rows = [row for path in sorted(args.output.glob("dev_shard*.jsonl")) for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError("no dev shard outputs")
    summary = {}
    for method in DEV_METHODS:
        group = [r for r in rows if r["condition"] == method]
        summary[method] = {
            "n": len(group),
            "accuracy": sum(bool(r["correct"]) for r in group) / max(1, len(group)),
            "mean_tokens": sum(r["tokens"] for r in group) / max(1, len(group)),
        }
    best = max(DEV_METHODS, key=lambda m: (summary[m]["accuracy"], -summary[m]["mean_tokens"], m))
    payload = {"selected_method": best, "dev": summary}
    write_json(args.output / "selection.json", payload)
    LOG.info("DEV SELECTION %s", json.dumps(payload, indent=2))


def run_test(args, cfg) -> None:
    selection_path = args.output / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError("selection.json missing; run --phase select")
    method = json.loads(selection_path.read_text(encoding="utf-8"))["selected_method"]
    out = args.output / f"test_shard{args.shard_index}.jsonl"
    if args.force and out.exists():
        out.unlink()
    done = completed(out)
    questions = experiment_questions(cfg, "test")
    questions = [q for i, q in enumerate(questions) if i % args.num_shards == args.shard_index]
    conditions = (f"{method}_text", f"{method}_visual_critic")
    # Interleave cheap text-only and full visual+critic conditions so the running ETA
    # reflects the real workload after the first question pair.
    tasks = [(condition, q) for q in questions for condition in conditions
             if (condition, q.id) not in done]
    LOG.info("TEST START shard=%d/%d selected=%s pending=%d", args.shard_index,
             args.num_shards, method, len(tasks))
    retriever = build_retriever(args, cfg)
    llm, vlm = get_llm(cfg), get_vlm(cfg)
    start = time.perf_counter()
    for index, (condition, question) in enumerate(tasks, start=1):
        t0 = time.perf_counter()
        selection = retriever.select(question.question, method)
        visual, visual_chunks = [], []
        corrected = False
        if condition.endswith("visual_critic"):
            visual, visual_chunks = inspect_visual_evidence(
                cfg, question, selection, vlm,
                max_visual_chunks=args.visual_chunks,
                max_frames_per_chunk=args.frames_per_chunk,
            )
            selection = attach_visual_evidence(selection, visual, visual_chunks, retriever.counter)
        answer = answer_from_selection(question, selection, retriever, llm)
        critic = None
        if condition.endswith("visual_critic"):
            critic = critique_evidence(question, selection, answer, llm)
            if not critic.sufficient:
                selection = retriever.expand(question.question, selection)
                answer = answer_from_selection(question, selection, retriever, llm)
                corrected = True
        append(out, {
            "phase": "test", "condition": condition, "qid": question.id,
            "type": question.type, "modality": question.modality,
            "reference": question.answer, "answer": answer.model_dump(mode="json"),
            "chunk_ids": selection.ranked_chunk_ids, "tokens": selection.tokens,
            "visual_chunks": visual_chunks, "visual_evidence": visual,
            "critic": critic.model_dump(mode="json") if critic else None,
            "corrective_retrieval": corrected,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        })
        LOG.info("TEST %d/%d condition=%s qid=%s modality=%s chunks=%s visual=%d "
                 "corrected=%s step=%.1fs ETA=%s", index, len(tasks), condition,
                 question.id, question.modality, selection.ranked_chunk_ids, len(visual),
                 corrected, time.perf_counter() - t0, eta(start, index, len(tasks)))
    LOG.info("TEST COMPLETE shard=%d rows=%d elapsed=%.1fmin", args.shard_index,
             len(tasks), (time.perf_counter() - start) / 60)


def aggregate(args, cfg) -> None:
    rows = [row for path in sorted(args.output.glob("test_shard*.jsonl")) for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError("no test shard outputs")
    llm = get_llm(cfg)
    questions = {q.id: q for q in experiment_questions(cfg, "test")}
    bounds = {chunk.chunk_id: (chunk.t_start, chunk.t_end) for chunk in Pipeline(cfg).chunks()}
    evaluated = []
    total = len(rows)
    start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        q = questions[row["qid"]]
        correct, method = deterministic_score(q, row["answer"]["answer"], judge=llm)
        evaluated.append({
            **row, "correct": correct, "scoring": method,
            "retrieved": any(
                any(min(qb, hi) - max(qa, lo) > 0 for qa, qb in q.gt_intervals)
                for chunk_id in row["chunk_ids"]
                for lo, hi in [bounds[chunk_id]]
            ),
        })
        LOG.info("EVAL %d/%d condition=%s qid=%s correct=%s method=%s ETA=%s",
                 index, total, row["condition"], row["qid"], correct, method,
                 eta(start, index, total))
    write_jsonl(args.output / "evaluated_test.jsonl", evaluated)
    fields = ["condition", "qid", "type", "modality", "reference", "correct", "scoring",
              "tokens", "chunk_ids", "visual_chunks", "corrective_retrieval", "wall_time_s"]
    with open(args.output / "evaluated_test.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluated)
    summary = {}
    table_rows = []
    for condition in sorted({r["condition"] for r in evaluated}):
        group = [r for r in evaluated if r["condition"] == condition]
        metric_rows = [{
            "correct": r["correct"], "grounded": None, "g_precision": None, "tiou": None,
            "retrieved": r["retrieved"], "recall_at_1": None, "recall_at_5": None,
            "recall_at_10": None, "reciprocal_rank": None, "temporal_error_s": None,
            "temporal_hit_20s": None, "temporal_hit_60s": None, "survived": None,
            "type": r["type"], "modality": r["modality"],
        } for r in group]
        summary[condition] = {
            **summarize_rows(metric_rows, cfg.analysis.n_bootstrap, cfg.seed),
            "mean_wall_time_s": round(sum(r["wall_time_s"] for r in group) / len(group), 2),
            "corrective_retrieval_rate": round(
                sum(bool(r["corrective_retrieval"]) for r in group) / len(group), 4),
        }
        table_rows.append({
            "condition": condition,
            "n": len(group),
            "accuracy": round(sum(bool(r["correct"]) for r in group) / len(group), 4),
            "retrieval_rate": round(sum(bool(r["retrieved"]) for r in group) / len(group), 4),
            "speech_accuracy": round(sum(bool(r["correct"]) for r in group if r["modality"] == "speech") /
                                     max(1, sum(r["modality"] == "speech" for r in group)), 4),
            "visual_accuracy": round(sum(bool(r["correct"]) for r in group if r["modality"] == "visual") /
                                     max(1, sum(r["modality"] == "visual" for r in group)), 4),
            "both_accuracy": round(sum(bool(r["correct"]) for r in group if r["modality"] == "both") /
                                   max(1, sum(r["modality"] == "both" for r in group)), 4),
            "mean_tokens": round(sum(r["tokens"] for r in group) / len(group), 1),
            "mean_wall_time_s": round(sum(r["wall_time_s"] for r in group) / len(group), 2),
            "corrective_retrieval_rate": round(
                sum(bool(r["corrective_retrieval"]) for r in group) / len(group), 4),
        })
    write_json(args.output / "summary.json", summary)
    with open(args.output / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    try:
        import matplotlib.pyplot as plt

        figures = args.output / "figures"
        figures.mkdir(exist_ok=True)
        labels = [r["condition"] for r in table_rows]
        values = [100 * r["accuracy"] for r in table_rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=["#4c78a8", "#f58518"][:len(labels)])
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set(ylabel="Accuracy (%)", ylim=(0, max(55, max(values) + 10)),
               title="Query-adaptive episodic rehydration")
        fig.tight_layout()
        fig.savefig(figures / "episodic_accuracy.png", dpi=180)
        plt.close(fig)

        modalities = ("speech", "visual", "both")
        width = 0.8 / len(labels)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = list(range(len(modalities)))
        for i, row in enumerate(table_rows):
            vals = [100 * row[f"{modality}_accuracy"] for modality in modalities]
            ax.bar([v + (i - (len(labels) - 1) / 2) * width for v in x], vals,
                   width, label=row["condition"])
        ax.set(xticks=x, xticklabels=modalities, ylabel="Accuracy (%)", ylim=(0, 100),
               title="Accuracy by evidence modality")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "episodic_accuracy_by_modality.png", dpi=180)
        plt.close(fig)
    except ImportError:
        LOG.warning("matplotlib unavailable; tables were written but figures were skipped")
    LOG.info("AGGREGATE COMPLETE %s", json.dumps(summary, indent=2))


def main() -> None:
    args = parser().parse_args()
    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    LOG.info("PHASE=%s CONFIG=%s OUTPUT=%s", args.phase, args.config, args.output)
    LOG.info("QUESTION_SHA256=%s SPLIT_SHA256=%s", sha256_file(cfg.questions_path),
             sha256_file(cfg.annotations_dir / "split.json"))
    if args.phase == "dev":
        run_dev(args, cfg)
    elif args.phase == "select":
        select_method(args)
    elif args.phase == "test":
        run_test(args, cfg)
    else:
        aggregate(args, cfg)


if __name__ == "__main__":
    main()
