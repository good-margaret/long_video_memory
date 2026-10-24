"""Resumable type-aware memory experiment with selective visual rehydration.

The baseline is frozen hybrid retrieval. Improved conditions add multi-query retrieval,
type-specific temporal diversity, structured answering, robust visual fallback, and a
targeted evidence gate. Per-question JSONL checkpoints make Kaggle timeout recovery safe.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from episodic_rehydration_experiment import (
    aggregate,
    append,
    build_retriever,
    completed,
    deterministic_score,
    eta,
    experiment_questions,
)
from memvid.config import load_config
from memvid.episodic import (
    EpisodicRetriever,
    answer_from_selection,
    attach_visual_evidence,
    choose_supported_answer,
    critique_evidence,
    inspect_visual_evidence,
)
from memvid.io import iter_jsonl, sha256_file, write_json
from memvid.models.registry import get_llm, get_vlm

LOG = logging.getLogger("adaptive_memory_experiment")
DEV_CONDITIONS = ("hybrid_baseline", "adaptive_text", "adaptive_visual_gate")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--memory-run", required=True, type=Path)
    p.add_argument("--output", default="runs/adaptive_memory_experiment", type=Path)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def should_gate(question, answer) -> bool:
    """Spend a critic call only where missing multi-part evidence is plausible."""
    return (
        question.type in ("object_change", "cross_segment")
        or question.modality != "speech"
        or answer.confidence == "low"
        or not answer.cited_ids
    )


def run_condition(args, cfg, retriever: EpisodicRetriever, llm, vlm, condition, question):
    adaptive = condition != "hybrid_baseline"
    selection = (
        retriever.select_question(question, "hybrid")
        if adaptive else retriever.select(question.question, "hybrid")
    )
    initial_chunk_ids = list(selection.ranked_chunk_ids)
    visual, visual_chunks = [], []
    visual_attempted = False
    if condition == "adaptive_visual_gate" and question.modality != "speech":
        visual_attempted = True
        visual, visual_chunks = inspect_visual_evidence(
            cfg,
            question,
            selection,
            vlm,
            max_visual_chunks=args.visual_chunks,
            max_frames_per_chunk=args.frames_per_chunk,
        )
        selection = attach_visual_evidence(selection, visual, visual_chunks, retriever.counter)

    initial = answer_from_selection(
        question, selection, retriever, llm, structured=adaptive,
    )
    answer = initial
    critic = None
    answer_choice = None
    corrected = False
    targeted_query = ""
    if condition == "adaptive_visual_gate" and should_gate(question, initial):
        critic = critique_evidence(question, selection, initial, llm)
        if not critic.sufficient:
            targeted_query = critic.targeted_query.strip() or critic.missing.strip()
            if not targeted_query:
                targeted_query = question.question
            expanded = retriever.expand_targeted(targeted_query, selection)
            revised = answer_from_selection(
                question, expanded, retriever, llm, structured=True,
            )
            answer, answer_choice = choose_supported_answer(
                question, expanded, initial, revised, llm,
            )
            selection = expanded
            corrected = True

    return {
        "answer": answer,
        "selection": selection,
        "initial_chunk_ids": initial_chunk_ids,
        "visual": visual,
        "visual_chunks": visual_chunks,
        "visual_attempted": visual_attempted,
        "critic": critic,
        "answer_choice": answer_choice,
        "corrected": corrected,
        "targeted_query": targeted_query,
    }


def run_rows(args, cfg, split: str, conditions: tuple[str, ...]) -> None:
    out = args.output / f"{split}_shard{args.shard_index}.jsonl"
    if args.force and out.exists():
        out.unlink()
    done = completed(out)
    questions = experiment_questions(cfg, split)
    questions = [q for i, q in enumerate(questions) if i % args.num_shards == args.shard_index]
    tasks = [(condition, q) for q in questions for condition in conditions
             if (condition, q.id) not in done]
    LOG.info("%s START shard=%d/%d pending=%d conditions=%s", split.upper(),
             args.shard_index, args.num_shards, len(tasks), ",".join(conditions))
    retriever = build_retriever(args, cfg)
    llm, vlm = get_llm(cfg), get_vlm(cfg)
    start = time.perf_counter()
    for index, (condition, question) in enumerate(tasks, start=1):
        t0 = time.perf_counter()
        result = run_condition(args, cfg, retriever, llm, vlm, condition, question)
        answer, selection = result["answer"], result["selection"]
        row = {
            "phase": split,
            "condition": condition,
            "qid": question.id,
            "type": question.type,
            "modality": question.modality,
            "reference": question.answer,
            "answer": answer.model_dump(mode="json"),
            "chunk_ids": selection.ranked_chunk_ids,
            "initial_chunk_ids": result["initial_chunk_ids"],
            "tokens": selection.tokens,
            "visual_attempted": result["visual_attempted"],
            "visual_chunks": result["visual_chunks"],
            "visual_evidence": result["visual"],
            "visual_parse_success": bool(result["visual"]) if result["visual_attempted"] else None,
            "critic": result["critic"].model_dump(mode="json") if result["critic"] else None,
            "answer_choice": (
                result["answer_choice"].model_dump(mode="json")
                if result["answer_choice"] else None
            ),
            "targeted_query": result["targeted_query"],
            "corrective_retrieval": result["corrected"],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        if split == "dev":
            row["correct"], row["scoring"] = deterministic_score(
                question, answer.answer, judge=llm,
            )
        append(out, row)
        LOG.info(
            "%s %d/%d condition=%s qid=%s modality=%s chunks=%s visual=%d "
            "gate=%s corrected=%s step=%.1fs ETA=%s",
            split.upper(), index, len(tasks), condition, question.id, question.modality,
            selection.ranked_chunk_ids, len(result["visual"]),
            result["critic"] is not None, result["corrected"],
            time.perf_counter() - t0, eta(start, index, len(tasks)),
        )
    LOG.info("%s COMPLETE shard=%d rows=%d elapsed=%.1fmin", split.upper(),
             args.shard_index, len(tasks), (time.perf_counter() - start) / 60)


def select_condition(args) -> None:
    rows = [row for path in sorted(args.output.glob("dev_shard*.jsonl"))
            for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError("no dev shard outputs")
    summary = {}
    for condition in DEV_CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "n": len(group),
            "accuracy": sum(bool(row["correct"]) for row in group) / max(1, len(group)),
            "mean_tokens": sum(row["tokens"] for row in group) / max(1, len(group)),
            "mean_wall_time_s": sum(row["wall_time_s"] for row in group) / max(1, len(group)),
        }
    candidates = ("adaptive_text", "adaptive_visual_gate")
    selected = max(candidates, key=lambda c: (
        summary[c]["accuracy"], -summary[c]["mean_wall_time_s"], -summary[c]["mean_tokens"],
    ))
    payload = {"selected_condition": selected, "dev": summary}
    write_json(args.output / "selection.json", payload)
    LOG.info("DEV SELECTION %s", json.dumps(payload, indent=2))


def main() -> None:
    args = parser().parse_args()
    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    LOG.info("PHASE=%s CONFIG=%s OUTPUT=%s", args.phase, args.config, args.output)
    LOG.info("QUESTION_SHA256=%s SPLIT_SHA256=%s", sha256_file(cfg.questions_path),
             sha256_file(cfg.annotations_dir / "split.json"))
    if args.phase == "dev":
        run_rows(args, cfg, "dev", DEV_CONDITIONS)
    elif args.phase == "select":
        select_condition(args)
    elif args.phase == "test":
        selected = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))[
            "selected_condition"
        ]
        run_rows(args, cfg, "test", ("hybrid_baseline", selected))
    else:
        aggregate(args, cfg)


if __name__ == "__main__":
    main()
