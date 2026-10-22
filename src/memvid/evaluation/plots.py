"""Figures for the report (matplotlib, optional extra `analysis`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from memvid.evaluation.stats import bootstrap_ci

log = logging.getLogger(__name__)
COLORS = {"C1": "#7f7f7f", "C2": "#1f77b4", "C3": "#d62728", "C4": "#2ca02c"}
CAUSE_COLORS = {"perception_miss": "#8c564b", "storage_loss": "#ff7f0e", "corruption_merge": "#d62728",
                "corruption_state": "#e377c2", "retrieval_miss": "#1f77b4", "reasoning_error": "#7f7f7f"}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        log.warning("matplotlib not installed: figures skipped (pip install -e '.[analysis]')")
        return None


def _acc(rows, cond, budget):
    xs = [float(r["correct"]) for r in rows if r["condition"] == cond and r["budget"] == budget]
    return bootstrap_ci(xs, n=2000) if xs else (np.nan, np.nan, np.nan)


def make_figures(rows: list[dict[str, Any]], sizes: dict[str, list[dict]], attribution: dict[str, Any],
                 budgets: list[str], budget_tokens: dict[str, int], focus: str, out: Path) -> list[Path]:
    plt = _plt()
    if plt is None:
        return []
    out.mkdir(parents=True, exist_ok=True)
    files = []

    # 1. accuracy vs budget
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xs = [budget_tokens[b] for b in budgets]
    for c in ("C1", "C2", "C3"):
        pts = [_acc(rows, c, b) for b in budgets]
        m = [p[0] for p in pts]
        err = [[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]]
        ax.errorbar(xs, m, yerr=err, marker="o", capsize=3, label=c, color=COLORS[c])
    c4 = _acc(rows, "C4", "full")
    if not np.isnan(c4[0]):
        ax.axhspan(c4[1], c4[2], color=COLORS["C4"], alpha=0.12)
        ax.axhline(c4[0], color=COLORS["C4"], ls="--", label="C4 (unbounded)")
    ax.set_xscale("log")
    ax.set_xlabel("memory budget (tokens)")
    ax.set_ylabel("answer accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Accuracy vs. budget (95% bootstrap CI)", fontsize=10)
    files.append(_save(fig, out / "accuracy_vs_budget.pdf", plt))

    # 2. accuracy by question type at the focus budget
    types = sorted({r["type"] for r in rows})
    conds = ["C1", "C2", "C3", "C4"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.2
    for i, c in enumerate(conds):
        b = "full" if c == "C4" else focus
        vals = []
        for t in types:
            xs_ = [float(r["correct"]) for r in rows if r["condition"] == c and r["budget"] == b and r["type"] == t]
            vals.append(np.mean(xs_) if xs_ else np.nan)
        ax.bar(np.arange(len(types)) + (i - 1.5) * w, vals, w, label=c, color=COLORS[c])
    ax.set_xticks(range(len(types)), [t.replace("_", "\n") for t in types], fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.set_title(f"Accuracy by question type (budget {focus})", fontsize=10)
    files.append(_save(fig, out / "accuracy_by_type.pdf", plt))

    # 3. retention: accuracy vs evidence age
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bins = np.quantile([r["age_s"] / 60 for r in rows], [0, 0.25, 0.5, 0.75, 1]) if rows else [0, 1]
    for c in ("C1", "C2", "C3"):
        sub = [r for r in rows if r["condition"] == c and r["budget"] == focus]
        mids, vals = [], []
        for a, b in zip(bins[:-1], bins[1:], strict=False):
            xs_ = [float(r["correct"]) for r in sub if a <= r["age_s"] / 60 <= b]
            if xs_:
                mids.append((a + b) / 2)
                vals.append(np.mean(xs_))
        ax.plot(mids, vals, marker="o", label=c, color=COLORS[c])
    ax.set_xlabel("evidence age (minutes before the end)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Retention (budget {focus}, age quartiles)", fontsize=10)
    files.append(_save(fig, out / "retention.pdf", plt))

    # 4. failure attribution
    by_run = attribution.get("by_run", {})
    if by_run:
        runs = sorted(by_run)
        fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(runs) + 1.5))
        left = np.zeros(len(runs))
        for cause, color in CAUSE_COLORS.items():
            v = np.array([by_run[r].get(cause, 0) for r in runs], float)
            ax.barh(runs, v, left=left, color=color, label=cause.replace("_", " "))
            left += v
        ax.set_xlabel("wrong answers")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        ax.set_title("Failure attribution (first stage where evidence was lost)", fontsize=10)
        files.append(_save(fig, out / "attribution.pdf", plt))

    # 5. memory size over time
    if sizes:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for run, tr in sorted(sizes.items()):
            base = run.split("_")[0].split("-")[0]
            ax.plot([s["chunk_id"] for s in tr], [s["tokens"] for s in tr], label=run,
                    color=COLORS.get(base, "k"), ls="-" if "-" not in run else ":")
        if focus in budget_tokens:
            ax.axhline(budget_tokens[focus], color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("chunk")
        ax.set_ylabel("memory size (tokens)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"Memory size over time (budget {focus})", fontsize=10)
        files.append(_save(fig, out / "memory_size.pdf", plt))
    return files


def _save(fig, path: Path, plt) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path

