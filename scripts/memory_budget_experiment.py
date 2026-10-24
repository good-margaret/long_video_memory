"""Compare flat and structured memory under matched token budgets.

The experiment reuses frozen F8_ASR observations. It never runs preprocessing or
perception. C4 is built once to define B1/B2/B3; C2 and C3 are then evaluated at every
budget with the same questions, retrieval allowance, answerer and judge.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from memvid.config import load_config
from memvid.evaluation.metrics import JudgeCache, evaluate_run, summarize_rows
from memvid.evaluation.stats import mcnemar_exact
from memvid.experiment import answer_run, build_memory, resolve_budgets
from memvid.io import read_json, write_json
from memvid.memory.persist import load_memory
from memvid.models.registry import get_llm
from memvid.pipeline import video_duration


BUDGETS = ("B1", "B2", "B3")


def _paired(rows_by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for budget in BUDGETS:
        left, right = f"C2_{budget}", f"C3_{budget}"
        a = {row["qid"]: bool(row["correct"]) for row in rows_by_run[left]}
        b = {row["qid"]: bool(row["correct"]) for row in rows_by_run[right]}
        qids = sorted(set(a) & set(b))
        only_c2 = sum(a[q] and not b[q] for q in qids)
        only_c3 = sum(b[q] and not a[q] for q in qids)
        output.append({
            "budget": budget,
            "n": len(qids),
            "only_c2_correct": only_c2,
            "only_c3_correct": only_c3,
            "mcnemar_p": mcnemar_exact(only_c2, only_c3),
        })
    return output


def _age_summary(rows: list[dict[str, Any]], duration: float) -> dict[str, dict[str, float]]:
    groups = {"recent": [], "middle": [], "old": []}
    for row in rows:
        age = float(row["age_s"])
        name = "recent" if age <= duration / 3 else "middle" if age <= 2 * duration / 3 else "old"
        groups[name].append(row)
    return {
        name: {
            "n": len(group),
            "accuracy": round(sum(bool(r["correct"]) for r in group) / len(group), 4),
            "survival": round(sum(bool(r["survived"]) for r in group) / len(group), 4),
        }
        for name, group in groups.items() if group
    }


def _run_diagnostics(run_dir: Path) -> dict[str, Any]:
    saved = load_memory(run_dir / "memory.sqlite")
    labels = Counter(op["label"] for op in saved.ops)
    final = saved.sizes[-1] if saved.sizes else {"tokens": 0, "n_records": 0}
    peak_tokens = max((row["tokens"] for row in saved.sizes), default=0)
    return {
        "final_tokens": final["tokens"],
        "final_records": final["n_records"],
        "peak_tokens": peak_tokens,
        "n_operations": len(saved.ops),
        "operation_labels": dict(sorted(labels.items())),
        "wall_time_s": read_json(run_dir / "run_info.json")["wall_time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_memory_budget.yaml")
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--output", default="runs/memory_budget_experiment")
    parser.add_argument(
        "--conditions", default="C2,C3,C4",
        help="comma-separated subset; useful for assigning C2 and C3 to separate GPUs",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    budgets = resolve_budgets(cfg)
    expected = set(BUDGETS)
    if set(budgets) != expected:
        raise ValueError(f"expected budgets {sorted(expected)}, got {sorted(budgets)}")

    selected = {item.strip() for item in args.conditions.split(",") if item.strip()}
    unknown = selected - {"C2", "C3", "C4"}
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    specs = [(condition, budget) for budget in BUDGETS for condition in ("C2", "C3")
             if condition in selected]
    if "C4" in selected:
        specs.append(("C4", "full"))
    judge = None if args.no_judge else get_llm(cfg)
    cache = JudgeCache(cfg)
    duration = video_duration(cfg)
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}

    for condition, budget in specs:
        run = build_memory(cfg, condition, budget, strict=True, force=args.force)
        answer_run(cfg, run, args.split, force=args.force)
        rows = evaluate_run(cfg, run, args.split, judge=judge, cache=cache)
        name = run.name
        rows_by_run[name] = rows
        summaries[name] = {
            **summarize_rows(rows, cfg.analysis.n_bootstrap, cfg.seed),
            "by_evidence_age": _age_summary(rows, duration),
        }
        diagnostics[name] = _run_diagnostics(run)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "question_split": args.split,
        "reference_tokens": read_json(cfg.video_runs_dir / "budgets.json")["reference_tokens"],
        "budgets": budgets,
        "runs": summaries,
        "diagnostics": diagnostics,
        "paired_c2_vs_c3": _paired(rows_by_run)
        if all(f"{condition}_{budget}" in rows_by_run
               for budget in BUDGETS for condition in ("C2", "C3")) else [],
    }
    write_json(out / "summary.json", result)

    flat = [
        {"experiment_run": name, **row}
        for name, rows in rows_by_run.items()
        for row in rows
    ]
    if flat:
        with open(out / "per_question.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)

    print("Resolved budgets:", budgets)
    print("Written:", out / "summary.json")


if __name__ == "__main__":
    main()
