"""Aggregate independently computed F4/F8 and ASR/SRT C3 condition results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from memvid.evaluation.stats import mcnemar_exact
from memvid.io import read_json, read_jsonl, write_json


CONDITIONS = ("F4_ASR", "F8_ASR", "F4_SRT", "F8_SRT")
COMPARISONS = (
    ("F4_ASR", "F8_ASR"),
    ("F4_SRT", "F8_SRT"),
    ("F4_ASR", "F4_SRT"),
    ("F8_ASR", "F8_SRT"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default="runs/input_budget_parts")
    parser.add_argument("--output", default="runs/input_budget_experiment")
    args = parser.parse_args()
    parts, out = Path(args.parts), Path(args.output)
    payloads = {name: read_json(parts / f"{name}.json") for name in CONDITIONS}
    rows = {name: read_jsonl(parts / f"{name}_per_question.jsonl") for name in CONDITIONS}
    budgets = {p["fixed_memory_budget_tokens"] for p in payloads.values()}
    if len(budgets) != 1:
        raise ValueError(f"conditions used different memory budgets: {sorted(budgets)}")
    paired = []
    for left, right in COMPARISONS:
        a = {row["qid"]: bool(row["correct"]) for row in rows[left]}
        b = {row["qid"]: bool(row["correct"]) for row in rows[right]}
        qids = sorted(set(a) & set(b))
        only_left = sum(a[q] and not b[q] for q in qids)
        only_right = sum(b[q] and not a[q] for q in qids)
        paired.append({
            "left": left,
            "right": right,
            "n": len(qids),
            "only_left_correct": only_left,
            "only_right_correct": only_right,
            "mcnemar_p": mcnemar_exact(only_left, only_right),
        })
    result = {
        "fixed_memory_budget_tokens": budgets.pop(),
        "split": next(iter(payloads.values()))["split"],
        "conditions": {name: p["metrics"] for name, p in payloads.items()},
        "efficiency": {name: p["efficiency"] for name, p in payloads.items()},
        "paired_comparisons": paired,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", result)
    flat = [{"experiment_condition": name, **row} for name in CONDITIONS for row in rows[name]]
    with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print("written:", out / "summary.json")


if __name__ == "__main__":
    main()
