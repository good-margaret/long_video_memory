"""Create compact tables and presentation-ready figures from completed experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("accuracy", "recall_at_5", "mrr", "tiou", "survived")
COLORS = {"C1": "#7f7f7f", "C2": "#4c78a8", "C3": "#e45756", "C4": "#54a24b"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(metric: Any) -> float | None:
    return None if metric is None else metric.get("mean")


def _metric_rows(group: str, summaries: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, summary in summaries.items():
        row = {"group": group, "run": name}
        row.update({metric: _mean(summary.get(metric)) for metric in METRICS})
        rows.append(row)
    return rows


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/final_summary")
    args = parser.parse_args()
    out = Path(args.output)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    baseline = _read(Path("runs/memory_baselines/summary.json"))
    inputs = _read(Path("runs/input_budget_experiment/summary.json"))
    budget_path = Path("runs/memory_budget_experiment/summary.json")
    budget = _read(budget_path) if budget_path.is_file() else None

    metric_rows = _metric_rows("memory_baseline", baseline["runs"])
    metric_rows += _metric_rows("input_condition", inputs["conditions"])
    if budget:
        metric_rows += _metric_rows("budget_pressure", budget["runs"])
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "all_metrics.csv", index=False)

    by_type, by_modality = [], []
    for group, summaries in (("memory_baseline", baseline["runs"]),
                             ("input_condition", inputs["conditions"])):
        for name, summary in summaries.items():
            by_type.extend({"group": group, "run": name, "type": key,
                            "accuracy": _mean(value)}
                           for key, value in summary.get("by_type", {}).items())
            by_modality.extend({"group": group, "run": name, "modality": key,
                                "accuracy": _mean(value)}
                               for key, value in summary.get("by_modality", {}).items())
    pd.DataFrame(by_type).to_csv(out / "accuracy_by_question_type.csv", index=False)
    pd.DataFrame(by_modality).to_csv(out / "accuracy_by_modality.csv", index=False)
    pd.DataFrame(inputs["paired_comparisons"]).to_csv(out / "paired_input_tests.csv", index=False)

    base = metrics[metrics.group == "memory_baseline"].copy()
    c3_reference = metrics[(metrics.group == "input_condition") & (metrics.run == "F8_ASR")].copy()
    c3_reference["run"] = "C3_B2"
    base = pd.concat([base, c3_reference], ignore_index=True)
    base["label"] = base.run.str.replace("_B2", "", regex=False).str.replace("_full", "", regex=False)
    base["label"] = pd.Categorical(base["label"], ["C1", "C2", "C3", "C4"], ordered=True)
    base = base.sort_values("label")
    base[["run", *METRICS]].to_csv(out / "memory_organization_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    values = base.accuracy.fillna(0)
    ax.bar(base.label, values, color=[COLORS.get(x, "#888888") for x in base.label])
    ax.set(ylim=(0, 1), ylabel="answer accuracy", title="Memory organization at matched B2 budget")
    for i, value in enumerate(values):
        ax.text(i, value + .02, f"{value:.2f}", ha="center")
    _save(fig, figures / "memory_baseline_accuracy.png")

    inp = metrics[metrics.group == "input_condition"].set_index("run").reindex(
        ["F4_ASR", "F8_ASR", "F4_SRT", "F8_SRT"]
    )
    x = np.arange(len(inp))
    fig, ax = plt.subplots(figsize=(8, 4))
    width = .36
    ax.bar(x - width / 2, inp.accuracy, width, label="accuracy", color="#e45756")
    ax.bar(x + width / 2, inp.recall_at_5, width, label="Recall@5", color="#4c78a8")
    ax.set(xticks=x, xticklabels=inp.index, ylim=(0, 1), title="Visual and transcript input conditions")
    ax.legend(frameon=False)
    _save(fig, figures / "input_accuracy_recall.png")

    efficiency = inputs.get("efficiency", {})
    cost_rows = []
    for name in inp.index:
        info = efficiency.get(name) or {}
        wall_time = info.get("wall_time_s_this_run", info.get("wall_time_s", 0))
        cost_rows.append({"condition": name, "wall_time_h": wall_time / 3600,
                          "accuracy": inp.loc[name, "accuracy"]})
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(out / "perception_cost.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(cost.wall_time_h, cost.accuracy, s=70, color="#f58518")
    for row in cost.itertuples():
        ax.annotate(row.condition, (row.wall_time_h, row.accuracy), xytext=(5, 5),
                    textcoords="offset points")
    ax.set(xlabel="perception wall time (hours)", ylabel="answer accuracy", ylim=(0, 1),
           title="Accuracy–compute trade-off")
    _save(fig, figures / "accuracy_compute_tradeoff.png")

    if by_modality:
        frame = pd.DataFrame(by_modality)
        frame = frame[frame.group == "input_condition"].pivot(index="run", columns="modality",
                                                                values="accuracy")
        frame = frame.reindex(inp.index)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        image = ax.imshow(frame.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set(xticks=range(len(frame.columns)), xticklabels=frame.columns,
               yticks=range(len(frame.index)), yticklabels=frame.index,
               title="Accuracy by question modality")
        for i in range(len(frame.index)):
            for j in range(len(frame.columns)):
                ax.text(j, i, f"{frame.iloc[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="accuracy")
        _save(fig, figures / "input_modality_heatmap.png")

    if budget:
        budgets = budget["budgets"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for condition in ("C2", "C3"):
            vals = [_mean(budget["runs"][f"{condition}_{name}"]["accuracy"]) for name in budgets]
            ax.plot([budgets[name] for name in budgets], vals, marker="o", label=condition,
                    color=COLORS[condition])
        c4 = _mean(budget["runs"]["C4_full"]["accuracy"])
        ax.axhline(c4, linestyle="--", color=COLORS["C4"], label="C4 full")
        ax.set(xscale="log", ylim=(0, 1), xlabel="memory budget (tokens)",
               ylabel="answer accuracy", title="Memory quality under budget pressure")
        ax.legend(frameon=False)
        _save(fig, figures / "budget_pressure_accuracy.png")
        pd.DataFrame(budget["paired_c2_vs_c3"]).to_csv(out / "paired_budget_tests.csv", index=False)

    manifest = {
        "tables": sorted(str(path.relative_to(out)) for path in out.glob("*.csv")),
        "figures": sorted(str(path.relative_to(out)) for path in figures.glob("*.png")),
        "budget_pressure_included": budget is not None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
