"""Resumable evidence-union experiment using C4 and episodic memory."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from episodic_rehydration_experiment import (
    aggregate,
    append,
    completed,
    deterministic_score,
    eta,
    experiment_questions,
)
from memvid.config import load_config
from memvid.evidence_union import EvidenceUnionRetriever, add_visual_to_union, answer_from_union
from memvid.episodic import inspect_visual_evidence
from memvid.io import sha256_file
from memvid.models.registry import get_embedder, get_llm, get_token_counter, get_vlm

LOG = logging.getLogger("evidence_union_experiment")
CONDITION = "c4_episodic_union"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--c3-memory", required=True, type=Path)
    p.add_argument("--c4-memory", required=True, type=Path)
    p.add_argument("--output", default="runs/evidence_union_experiment", type=Path)
    p.add_argument("--phase", choices=("dev", "test", "aggregate"), required=True)
    p.add_argument("--union-budget", type=int, default=3500)
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--visual-chunks", type=int, default=2)
    p.add_argument("--frames-per-chunk", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    return p


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def run(args, cfg, split: str) -> None:
    out = args.output / f"{split}_shard0.jsonl"
    if args.force and out.exists():
        out.unlink()
    done = completed(out)
    questions = experiment_questions(cfg, split)
    questions = [question for question in questions if (CONDITION, question.id) not in done]
    if args.limit is not None:
        questions = questions[: args.limit]
    LOG.info("%s START pending=%d condition=%s", split.upper(), len(questions), CONDITION)
    counter = get_token_counter(cfg)
    retriever = EvidenceUnionRetriever(
        cfg,
        args.c3_memory,
        args.c4_memory,
        get_embedder(cfg),
        counter,
        union_budget_tokens=args.union_budget,
        max_chunks=args.max_chunks,
    )
    llm, vlm = get_llm(cfg), get_vlm(cfg)
    start = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        t0 = time.perf_counter()
        selection = retriever.select(question)
        visual, visual_chunks = [], []
        if question.modality != "speech":
            visual_limit = 3 if question.type == "attribute_recall" else args.visual_chunks
            visual, visual_chunks = inspect_visual_evidence(
                cfg,
                question,
                selection.episode_selection,
                vlm,
                max_visual_chunks=visual_limit,
                max_frames_per_chunk=args.frames_per_chunk,
            )
            selection = add_visual_to_union(selection, visual, visual_chunks, counter)
        answer = answer_from_union(question, selection, llm, retriever.episodes.bounds)
        row = {
            "phase": split,
            "condition": CONDITION,
            "qid": question.id,
            "type": question.type,
            "modality": question.modality,
            "reference": question.answer,
            "answer": answer.model_dump(mode="json"),
            "chunk_ids": selection.episode_selection.ranked_chunk_ids,
            "c4_ids": selection.c4_ids,
            "archive_ids": selection.archive_ids,
            "tokens": selection.retrieved.tokens,
            "visual_chunks": visual_chunks,
            "visual_evidence": visual,
            "visual_parse_success": bool(visual) if question.modality != "speech" else None,
            "corrective_retrieval": False,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        if split == "dev":
            row["correct"], row["scoring"] = deterministic_score(
                question, answer.answer, judge=llm,
            )
        append(out, row)
        LOG.info(
            "%s %d/%d qid=%s type=%s modality=%s answer_ok=%s c4=%d archive=%d "
            "visual=%d tokens=%d step=%.1fs ETA=%s",
            split.upper(), index, len(questions), question.id, question.type,
            question.modality, answer.ok, len(selection.c4_ids), len(selection.archive_ids),
            len(visual), selection.retrieved.tokens, time.perf_counter() - t0,
            eta(start, index, len(questions)),
        )
    LOG.info("%s COMPLETE rows=%d elapsed=%.1fmin", split.upper(), len(questions),
             (time.perf_counter() - start) / 60)


def main() -> None:
    args = parser().parse_args()
    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    LOG.info("PHASE=%s CONFIG=%s OUTPUT=%s", args.phase, args.config, args.output)
    LOG.info("QUESTION_SHA256=%s SPLIT_SHA256=%s", sha256_file(cfg.questions_path),
             sha256_file(cfg.annotations_dir / "split.json"))
    LOG.info("NO_REFERENCE_LEAKAGE generation uses questions/options and retrieved evidence only")
    if args.phase in ("dev", "test"):
        run(args, cfg, args.phase)
    else:
        aggregate(args, cfg)


if __name__ == "__main__":
    main()
