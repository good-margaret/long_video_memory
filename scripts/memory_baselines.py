"""Run the C1/C2/C4 memory baselines on frozen F8_ASR observations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from memvid.config import load_config
from memvid.evaluation.metrics import JudgeCache, evaluate_run, summarize_rows
from memvid.experiment import answer_run, build_memory, resolve_budgets
from memvid.io import write_json
from memvid.models.registry import get_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_f8_asr.yaml")
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--output", default="runs/memory_baselines")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--conditions", default="C1,C2,C4",
        help="comma-separated subset of C1,C2,C4. C1 rewrites its whole summary after every "
             "chunk, so at large budgets it can take longer than a GPU session; run it separately",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fixed_b2 = resolve_budgets(cfg)["B2"]
    cfg.budgets.mode = "tokens"
    cfg.budgets.values = {"B2": float(fixed_b2)}
    selected = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(selected) - {"C1", "C2", "C4"}
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    specs = [(c, "full" if c == "C4" else "B2") for c in ("C1", "C2", "C4") if c in selected]
    judge = None if args.no_judge else get_llm(cfg)
    cache = JudgeCache(cfg)
    summaries = {}
    rows_all = []

    for condition, budget in specs:
        run = build_memory(cfg, condition, budget, strict=True, force=args.force)
        answer_run(cfg, run, args.split, force=args.force)
        rows = evaluate_run(cfg, run, args.split, judge=judge, cache=cache)
        summaries[run.name] = summarize_rows(rows, cfg.analysis.n_bootstrap, cfg.seed)
        rows_all.extend({"experiment_run": run.name, **row} for row in rows)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", {
        "split": args.split,
        "fixed_b2_tokens": fixed_b2,
        "runs": summaries,
    })
    if rows_all:
        with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_all[0]))
            writer.writeheader()
            writer.writerows(rows_all)
    print("fixed B2 budget:", fixed_b2, "tokens")
    print("written:", out / "summary.json")


if __name__ == "__main__":
    main()
