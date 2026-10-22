"""Failure attribution: one cause per wrong answer, checked in pipeline order.

  perception_miss   the observations of the evidence chunks lack the evidence  (manual)
  storage_loss      no record of the final memory has support overlapping G(q)
  corruption_merge  an overlapping record refers to an entity with a wrong merge (merge audit)
  corruption_state  an overlapping record is a spurious state change (state audit)
  retrieval_miss    overlapping records exist but none was retrieved
  reasoning_error   overlapping records were retrieved, the answer is still wrong

The code proposes a cause from the run data and the audits (`proposed_cause`). The
annotator confirms or overrides it in `final_cause`, and marks
`perception_has_evidence = n` when the observations never contained the evidence.
The sheet (runs/<video_id>/audits/attribution.csv) is regenerated without losing labels.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from memvid.config import ProjectConfig
from memvid.evaluation.audits import audits_dir, entity_ids_with_wrong_links, spurious_state_changes
from memvid.experiment import list_runs
from memvid.io import read_jsonl
from memvid.memory.persist import load_memory
from memvid.memory.records import record_overlaps, references
from memvid.questions.manage import load_questions

CAUSES = ["perception_miss", "storage_loss", "corruption_merge", "corruption_state",
          "retrieval_miss", "reasoning_error"]
FIELDS = ["run", "qid", "type", "question", "reference", "answer", "survived", "lost_by",
          "retrieved", "suspect_entities", "suspect_changes", "proposed_cause",
          "perception_has_evidence", "final_cause", "notes"]
MANUAL = ["perception_has_evidence", "final_cause", "notes"]


def sheet_path(cfg: ProjectConfig) -> Path:
    return audits_dir(cfg) / "attribution.csv"


def _lost_by(saved, gt) -> str:
    """Labels of the operations that deleted records overlapping G(q)."""
    last: dict[str, dict] = {}
    labels: Counter = Counter()
    for op in saved.ops:
        if op["op"] == "upsert":
            last[op["record_id"]] = op["record"]
        elif op["record_id"] in last:
            from memvid.memory.store import parse_record

            rec = parse_record(last[op["record_id"]])
            if record_overlaps(rec, saved.bounds, gt):
                labels[op["label"]] += 1
    return " ".join(f"{k}:{v}" for k, v in sorted(labels.items()))


def build_attribution(cfg: ProjectConfig, split: str) -> Path:
    path = sheet_path(cfg)
    old: dict[tuple[str, str], dict[str, str]] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                old[(r["run"], r["qid"])] = r
    qs = {q.id: q for q in load_questions(cfg, split)}
    bad_entities = entity_ids_with_wrong_links(cfg)
    bad_changes = spurious_state_changes(cfg)
    rows = []
    for run in list_runs(cfg):
        if run["condition"].startswith("C1"):
            continue
        rdir = Path(run["path"])
        ev = rdir / f"eval_{split}.jsonl"
        if not ev.is_file():
            continue
        saved = load_memory(rdir / "memory.sqlite")
        for r in read_jsonl(ev):
            if r["correct"]:
                continue
            q = qs[r["qid"]]
            gt = list(q.gt_intervals)
            # same survival proxy as metrics.py: non-entity records only
            overlapping = [x for x in saved.records.values()
                           if x.kind != "entity" and record_overlaps(x, saved.bounds, gt)]
            sus_e = sorted({e for x in overlapping for e in references(x)} & bad_entities)
            sus_s = sorted({x.id for x in overlapping} & bad_changes)
            if not overlapping:
                cause = "storage_loss"
            elif sus_e:
                cause = "corruption_merge"
            elif sus_s:
                cause = "corruption_state"
            elif not r["retrieved"]:
                cause = "retrieval_miss"
            else:
                cause = "reasoning_error"
            row = {"run": run["run"], "qid": q.id, "type": q.type, "question": q.question,
                   "reference": q.answer, "answer": r["answer"], "survived": bool(overlapping),
                   "lost_by": _lost_by(saved, gt), "retrieved": r["retrieved"],
                   "suspect_entities": " ".join(sus_e), "suspect_changes": " ".join(sus_s),
                   "proposed_cause": cause,
                   **{k: "" for k in MANUAL}}
            prev = old.get((run["run"], q.id))
            if prev:
                row.update({k: prev.get(k, "") for k in MANUAL})
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def final_cause(row: dict[str, str]) -> str:
    if (row.get("perception_has_evidence") or "").strip().lower() in ("n", "no"):
        return "perception_miss"
    fc = (row.get("final_cause") or "").strip()
    if fc:
        if fc not in CAUSES:
            raise ValueError(f"{row['run']}/{row['qid']}: unknown cause '{fc}' (use one of {CAUSES})")
        return fc
    return row["proposed_cause"]


def load_attribution(cfg: ProjectConfig) -> list[dict[str, str]]:
    p = sheet_path(cfg)
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize_attribution(cfg: ProjectConfig) -> dict[str, Any]:
    rows = load_attribution(cfg)
    by_run: dict[str, Counter] = {}
    for r in rows:
        by_run.setdefault(r["run"], Counter())[final_cause(r)] += 1
    confirmed = sum(1 for r in rows if (r.get("final_cause") or r.get("perception_has_evidence")))
    return {"by_run": {k: dict(v) for k, v in sorted(by_run.items())}, "n_rows": len(rows),
            "n_confirmed": confirmed}
