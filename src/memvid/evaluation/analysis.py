"""`memvid analyze`: evaluate all runs, draw figures, export tables.

Outputs: runs/<video_id>/analysis/{summary.json, figures/*.pdf}
         report/generated/results_*.tex, report/figures/*.pdf
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from memvid.config import ProjectConfig
from memvid.evaluation.attribution import build_attribution, summarize_attribution
from memvid.evaluation.metrics import JudgeCache, evaluate_run, summarize_rows
from memvid.evaluation.stats import mcnemar_exact
from memvid.experiment import list_runs, resolve_budgets
from memvid.io import read_json, write_json
from memvid.memory.persist import load_memory
from memvid.models.registry import get_llm

log = logging.getLogger(__name__)


def analysis_dir(cfg: ProjectConfig) -> Path:
    return cfg.video_runs_dir / "analysis"


def analyze(cfg: ProjectConfig, split: str | None = None, *, judge: bool | None = None) -> dict[str, Any]:
    split = split or cfg.experiment.split
    use_judge = cfg.analysis.judge_free_form if judge is None else judge
    judge_model = get_llm(cfg) if use_judge else None
    cache = JudgeCache(cfg)
    budgets_tok = resolve_budgets(cfg)
    budgets = list(cfg.budgets.values)
    rows: list[dict[str, Any]] = []
    sizes: dict[str, list[dict]] = {}
    heuristic = False
    runs = list_runs(cfg)
    if not runs:
        raise RuntimeError("no runs found: run `memvid run-all` first")
    for run in runs:
        rdir = Path(run["path"])
        if not (rdir / f"answers_{split}.jsonl").is_file():
            log.warning("%s has no answers for split %s; skipped", run["run"], split)
            continue
        rows += evaluate_run(cfg, rdir, split, judge=judge_model, cache=cache)
        heuristic |= bool(read_json(rdir / "run_info.json").get("heuristic_models"))
        if run["budget"] in (cfg.experiment.ablation_budget, "full"):
            sizes[run["run"]] = load_memory(rdir / "memory.sqlite").sizes
    build_attribution(cfg, split)
    attribution = summarize_attribution(cfg)

    operations = {run["run"]: operation_counts(Path(run["path"])) for run in runs
                  if (Path(run["path"]) / "memory.sqlite").is_file()}
    per_run = {}
    for run in sorted({r["run"] for r in rows}):
        sub = [r for r in rows if r["run"] == run]
        per_run[run] = {"condition": sub[0]["condition"], "budget": sub[0]["budget"],
                        **summarize_rows(sub, cfg.analysis.n_bootstrap, cfg.seed)}
    mcn = mcnemar_table(rows, budgets)
    summary = {"split": split, "n_questions": len({r["qid"] for r in rows}),
               "budgets_tokens": budgets_tok, "runs": per_run, "mcnemar": mcn,
               "operations": operations,
               "attribution": attribution, "heuristic_models": heuristic,
               "warning": "results produced with the heuristic stand-in model are not valid"
               if heuristic else None}
    out = analysis_dir(cfg)
    write_json(out / "summary.json", summary)

    from memvid.evaluation.plots import make_figures

    figs = make_figures(rows, sizes, attribution, budgets, budgets_tok,
                        cfg.experiment.ablation_budget, out / "figures")
    fig_dir = cfg.resolve("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in figs:
        shutil.copy(f, fig_dir / f.name)

    from memvid.reporting.results_tex import export_results

    export_results(cfg, summary, budgets)
    return {"summary": summary, "figures": [str(f) for f in figs]}


OP_LABELS = ["insert", "link", "change", "extend", "compress", "evict", "truncate", "rewrite"]


def operation_counts(rdir: Path) -> dict[str, Any]:
    """How the memory was updated: operations by semantic label (deletions of compressed
    originals counted separately), entity-resolution decisions, final composition."""
    saved = load_memory(rdir / "memory.sqlite")
    ops: dict[str, int] = {k: 0 for k in OP_LABELS}
    compressed_away = 0
    for o in saved.ops:
        if o["label"] == "compress":
            if o["op"] == "delete":
                compressed_away += 1
            elif (o["record"] or {}).get("summarized"):
                ops["compress"] += 1  # a summary written (single-event marks are not counted)
        elif o["label"] in ops:
            ops[o["label"]] += 1
    final: dict[str, int] = {}
    for r in saved.records.values():
        final[r.kind] = final.get(r.kind, 0) + 1
    decisions: dict[str, int] = {}
    for d in saved.resolution:
        decisions[d["decision"]] = decisions.get(d["decision"], 0) + 1
    return {"ops": ops, "compressed_away": compressed_away, "final_records": final,
            "final_tokens": saved.meta.get("final_tokens"), "resolution": decisions}


def mcnemar_table(rows, budgets) -> list[dict[str, Any]]:
    out = []
    for b in budgets:
        for other in ("C1", "C2"):
            a = {r["qid"]: r["correct"] for r in rows if r["condition"] == "C3" and r["budget"] == b}
            o = {r["qid"]: r["correct"] for r in rows if r["condition"] == other and r["budget"] == b}
            qs = sorted(set(a) & set(o))
            if not qs:
                continue
            only_c3 = sum(1 for q in qs if a[q] and not o[q])
            only_o = sum(1 for q in qs if o[q] and not a[q])
            out.append({"budget": b, "comparison": f"C3 vs {other}", "only_C3": only_c3,
                        "only_other": only_o, "n": len(qs), "p": mcnemar_exact(only_c3, only_o)})
    return out
