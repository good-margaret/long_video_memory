"""Run the controlled 4-vs-8 frames and ASR-vs-SRT experiment.

All variants use the B2 token count resolved from the F8_ASR unbounded log. This keeps
the actual memory allowance fixed instead of giving each perception variant 5% of a
different C4 size.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from memvid.config import load_config
from memvid.evaluation.metrics import JudgeCache, evaluate_run, summarize_rows
from memvid.evaluation.stats import mcnemar_exact
from memvid.experiment import answer_run, build_memory, resolve_budgets
from memvid.io import read_json, write_json
from memvid.models.registry import get_llm


DEFAULTS = {
    "F4_ASR": "configs/experiment_f4_asr.yaml",
    "F8_ASR": "configs/experiment_f8_asr.yaml",
    "F4_SRT": "configs/experiment_f4_srt.yaml",
    "F8_SRT": "configs/experiment_f8_srt.yaml",
}


def paired(rows_by_condition: dict[str, list[dict]]) -> list[dict]:
    comparisons = [
        ("F4_ASR", "F8_ASR"),
        ("F4_SRT", "F8_SRT"),
        ("F4_ASR", "F4_SRT"),
        ("F8_ASR", "F8_SRT"),
    ]
    output = []
    for left, right in comparisons:
        a = {row["qid"]: bool(row["correct"]) for row in rows_by_condition[left]}
        b = {row["qid"]: bool(row["correct"]) for row in rows_by_condition[right]}
        qids = sorted(set(a) & set(b))
        only_left = sum(a[q] and not b[q] for q in qids)
        only_right = sum(b[q] and not a[q] for q in qids)
        output.append(
            {
                "left": left,
                "right": right,
                "n": len(qids),
                "only_left_correct": only_left,
                "only_right_correct": only_right,
                "mcnemar_p": mcnemar_exact(only_left, only_right),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument("--output", default="runs/input_budget_experiment")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    configs = {name: load_config(path) for name, path in DEFAULTS.items()}
    baseline = configs["F8_ASR"]
    fixed_b2 = resolve_budgets(baseline)["B2"]

    rows_by_condition: dict[str, list[dict]] = {}
    summaries = {}
    efficiency = {}
    for name, cfg in configs.items():
        cfg.budgets.mode = "tokens"
        cfg.budgets.values = {"B2": float(fixed_b2)}
        run_dir = build_memory(cfg, "C3", "B2", strict=True, force=args.force)
        answer_run(cfg, run_dir, args.split, force=args.force)
        judge = None if args.no_judge else get_llm(cfg)
        rows = evaluate_run(cfg, run_dir, args.split, judge=judge, cache=JudgeCache(cfg))
        rows_by_condition[name] = rows
        summaries[name] = summarize_rows(rows, cfg.analysis.n_bootstrap, cfg.seed)
        perception_info = cfg.derived_dir / "perception_info.json"
        efficiency[name] = read_json(perception_info) if perception_info.is_file() else None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "fixed_memory_budget_tokens": fixed_b2,
        "split": args.split,
        "conditions": summaries,
        "efficiency": efficiency,
        "paired_comparisons": paired(rows_by_condition),
    }
    write_json(out / "summary.json", result)

    flat_rows = [
        {"experiment_condition": condition, **row}
        for condition, rows in rows_by_condition.items()
        for row in rows
    ]
    if flat_rows:
        with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    print(f"fixed B2 budget: {fixed_b2} tokens")
    print(f"written: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
