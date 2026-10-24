"""Run one C3 input condition with a shared absolute B2 memory budget."""

from __future__ import annotations

import argparse
from pathlib import Path

from memvid.config import load_config
from memvid.evaluation.metrics import JudgeCache, evaluate_run, summarize_rows
from memvid.experiment import answer_run, build_memory
from memvid.io import read_json, write_json, write_jsonl
from memvid.models.registry import get_llm


CONFIGS = {
    "F4_ASR": "configs/experiment_f4_asr.yaml",
    "F8_ASR": "configs/experiment_f8_asr.yaml",
    "F4_SRT": "configs/experiment_f4_srt.yaml",
    "F8_SRT": "configs/experiment_f8_srt.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--budget-tokens", required=True, type=int)
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--output", default="runs/input_budget_parts")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    cfg = load_config(CONFIGS[args.condition])
    cfg.budgets.mode = "tokens"
    cfg.budgets.values = {"B2": float(args.budget_tokens)}
    run = build_memory(cfg, "C3", "B2", strict=True, force=args.force)
    answer_run(cfg, run, args.split, force=args.force)
    judge = None if args.no_judge else get_llm(cfg)
    rows = evaluate_run(cfg, run, args.split, judge=judge, cache=JudgeCache(cfg))
    efficiency_path = cfg.derived_dir / "perception_info.json"
    payload = {
        "condition": args.condition,
        "fixed_memory_budget_tokens": args.budget_tokens,
        "split": args.split,
        "metrics": summarize_rows(rows, cfg.analysis.n_bootstrap, cfg.seed),
        "efficiency": read_json(efficiency_path) if efficiency_path.is_file() else None,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{args.condition}.json", payload)
    write_jsonl(out / f"{args.condition}_per_question.jsonl", rows)
    print("written:", out / f"{args.condition}.json")


if __name__ == "__main__":
    main()
