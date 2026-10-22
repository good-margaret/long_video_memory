"""Manual audits that turn subjective judgements into reported rates (RQ3, judge validity).

merge audit         sampled entity-resolution decisions of one C3 run:
                    links (was the merge right?) and new entities (should it have merged?)
state-change audit  sampled StateChange records: is the change real?
judge audit         every answer scored by the LLM judge: does a human agree?
Each `make_*` writes a CSV with empty label columns; each `summarize_*` reads it back.
Sheets live in runs/<video_id>/audits/ and are never overwritten.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from memvid.config import ProjectConfig
from memvid.errors import MemvidError, PipelineStateError
from memvid.evaluation.stats import wilson
from memvid.io import read_jsonl, write_json
from memvid.memory.persist import load_memory
from memvid.memory.records import StateChange
from memvid.perception.run import contact_sheet
from memvid.schemas import FrameRecord
from memvid.timeutils import format_timestamp


def audits_dir(cfg: ProjectConfig) -> Path:
    return cfg.video_runs_dir / "audits"


def default_c3_run(cfg: ProjectConfig) -> Path:
    return cfg.video_runs_dir / f"C3_{cfg.experiment.ablation_budget}"


def _yes(v: str) -> bool | None:
    v = (v or "").strip().lower()
    if v in ("y", "yes", "1", "true"):
        return True
    if v in ("n", "no", "0", "false"):
        return False
    return None


def _write(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> Path:
    if path.exists():
        raise MemvidError(f"{path} exists (it may hold your labels); delete it to regenerate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PipelineStateError(f"{path} missing; create it first")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sheets(cfg: ProjectConfig, chunk_ids: set[int]) -> dict[int, str]:
    frames: list[FrameRecord] = read_jsonl(cfg.derived_dir / "frames" / "index.jsonl", FrameRecord)
    out = {}
    for c in sorted(chunk_ids):
        p = audits_dir(cfg) / "sheets" / f"chunk_{c:04d}.jpg"
        if not p.exists():
            contact_sheet(cfg, [f for f in frames if f.chunk_id == c], p)
        out[c] = str(p.relative_to(audits_dir(cfg)))
    return out


# ----------------------------------------------------------------------------- merges
MERGE_FIELDS = ["kind", "chunk_id", "time", "local_name", "decision", "entity_id", "entity_name",
                "candidate_id", "candidate_name", "sim", "sheet_now", "sheet_entity_first_seen",
                "correct", "notes"]


def make_merge_audit(cfg: ProjectConfig, run: Path | None = None) -> Path:
    run = run or default_c3_run(cfg)
    saved = load_memory(run / "memory.sqlite")
    rng = random.Random(cfg.seed + 3)
    links = [r for r in saved.resolution if r["decision"].endswith("link")]
    news = [r for r in saved.resolution if r["decision"] == "new" and r.get("candidate_id")]
    links = rng.sample(links, min(cfg.analysis.audit_merges, len(links)))
    news = rng.sample(news, min(cfg.analysis.audit_new_entities, len(news)))
    first_chunk: dict[str, int] = {}
    for op in saved.ops:
        if op["op"] == "upsert" and op["record_id"].startswith("E") and op["record_id"] not in first_chunk:
            first_chunk[op["record_id"]] = op["chunk_id"]
    need = {r["chunk_id"] for r in links + news}
    for r in links + news:
        other = r["entity_id"] if r in links else r.get("candidate_id")
        if other in first_chunk:
            need.add(first_chunk[other])
    sheets = _sheets(cfg, need)
    rows = []
    for kind, items in (("link", links), ("new", news)):
        for r in sorted(items, key=lambda x: (x["chunk_id"], x["local_name"])):
            other = r["entity_id"] if kind == "link" else r.get("candidate_id")
            rows.append({
                "kind": kind, "chunk_id": r["chunk_id"], "time": format_timestamp(r["t"]),
                "local_name": r["local_name"], "decision": r["decision"], "entity_id": r["entity_id"],
                "entity_name": r["entity_name"], "candidate_id": r.get("candidate_id") or "",
                "candidate_name": r.get("candidate_name") or "", "sim": r["sim"],
                "sheet_now": sheets.get(r["chunk_id"], ""),
                "sheet_entity_first_seen": sheets.get(first_chunk.get(other, -1), ""),
                "correct": "", "notes": "",
            })
    return _write(audits_dir(cfg) / "merge_audit.csv", MERGE_FIELDS, rows)


def summarize_merge_audit(cfg: ProjectConfig) -> dict[str, Any]:
    rows = _read(audits_dir(cfg) / "merge_audit.csv")
    lab = [(r["kind"], _yes(r["correct"])) for r in rows]
    if any(v is None for _, v in lab):
        raise MemvidError("merge_audit.csv: every row needs correct = y or n")
    links = [v for k, v in lab if k == "link"]
    news = [v for k, v in lab if k == "new"]
    k1, k2 = sum(links), sum(1 for v in news if not v)
    out = {
        "links_checked": len(links), "merge_precision": round(k1 / len(links), 3) if links else None,
        "merge_precision_ci": wilson(k1, len(links)) if links else None,
        "new_checked": len(news), "missed_merge_rate": round(k2 / len(news), 3) if news else None,
        "missed_merge_rate_ci": wilson(k2, len(news)) if news else None,
        "wrong_links": [(r["chunk_id"], r["entity_id"]) for r in rows
                        if r["kind"] == "link" and _yes(r["correct"]) is False],
    }
    write_json(audits_dir(cfg) / "merge_audit_summary.json", out)
    return out


# ----------------------------------------------------------------------------- state changes
SC_FIELDS = ["id", "entity_id", "entity", "attr", "old", "new", "time", "chunks", "sheets",
             "correct", "notes"]


def make_state_audit(cfg: ProjectConfig, run: Path | None = None) -> Path:
    run = run or default_c3_run(cfg)
    saved = load_memory(run / "memory.sqlite")
    seen: dict[str, StateChange] = {}
    for op in saved.ops:  # include changes later evicted: the audit is about detection
        if op["op"] == "upsert" and op["record_id"].startswith("S"):
            seen[op["record_id"]] = StateChange.model_validate(op["record"])
    rng = random.Random(cfg.seed + 4)
    sample = rng.sample(sorted(seen.values(), key=lambda s: s.t),
                        min(cfg.analysis.audit_state_changes, len(seen)))
    sheets = _sheets(cfg, {c for s in sample for c in s.source_chunks})
    rows = [{"id": s.id, "entity_id": s.entity_id, "entity": s.entity_name, "attr": s.attr,
             "old": s.old, "new": s.new, "time": format_timestamp(s.t),
             "chunks": " ".join(map(str, s.source_chunks)),
             "sheets": " ".join(sheets[c] for c in s.source_chunks), "correct": "", "notes": ""}
            for s in sorted(sample, key=lambda s: s.t)]
    return _write(audits_dir(cfg) / "state_audit.csv", SC_FIELDS, rows)


def summarize_state_audit(cfg: ProjectConfig) -> dict[str, Any]:
    rows = _read(audits_dir(cfg) / "state_audit.csv")
    lab = [_yes(r["correct"]) for r in rows]
    if any(v is None for v in lab):
        raise MemvidError("state_audit.csv: every row needs correct = y or n")
    k = sum(lab)
    out = {"checked": len(rows), "precision": round(k / len(rows), 3) if rows else None,
           "precision_ci": wilson(k, len(rows)) if rows else None,
           "spurious": [r["id"] for r in rows if _yes(r["correct"]) is False]}
    write_json(audits_dir(cfg) / "state_audit_summary.json", out)
    return out


# ----------------------------------------------------------------------------- judge
JUDGE_FIELDS = ["qid", "question", "reference", "answer", "judge_correct", "manual_correct"]


def make_judge_audit(cfg: ProjectConfig, split: str) -> Path:
    from memvid.experiment import list_runs

    seen: dict[tuple[str, str], dict[str, Any]] = {}
    qtext = {}
    from memvid.questions.manage import load_questions

    for q in load_questions(cfg, split):
        qtext[q.id] = q.question
    for run in list_runs(cfg):
        p = Path(run["path"]) / f"eval_{split}.jsonl"
        if not p.is_file():
            continue
        for r in read_jsonl(p):
            if r["method"] == "judge":
                seen.setdefault((r["qid"], r["answer"]), r)
    rows = [{"qid": q, "question": qtext.get(q, ""), "reference": r["reference"], "answer": a,
             "judge_correct": "y" if r["correct"] else "n", "manual_correct": ""}
            for (q, a), r in sorted(seen.items())]
    return _write(audits_dir(cfg) / "judge_audit.csv", JUDGE_FIELDS, rows)


def summarize_judge_audit(cfg: ProjectConfig) -> dict[str, Any]:
    rows = _read(audits_dir(cfg) / "judge_audit.csv")
    pairs = [(_yes(r["judge_correct"]), _yes(r["manual_correct"])) for r in rows]
    if any(m is None for _, m in pairs):
        raise MemvidError("judge_audit.csv: every row needs manual_correct = y or n")
    agree = sum(1 for j, m in pairs if j == m)
    out = {"checked": len(rows), "agreement": round(agree / len(rows), 3) if rows else None,
           "agreement_ci": wilson(agree, len(rows)) if rows else None}
    write_json(audits_dir(cfg) / "judge_audit_summary.json", out)
    return out


def entity_ids_with_wrong_links(cfg: ProjectConfig) -> set[str]:
    p = audits_dir(cfg) / "merge_audit.csv"
    if not p.is_file():
        return set()
    return {r["entity_id"] for r in _read(p) if r["kind"] == "link" and _yes(r["correct"]) is False}


def spurious_state_changes(cfg: ProjectConfig) -> set[str]:
    p = audits_dir(cfg) / "state_audit.csv"
    if not p.is_file():
        return set()
    return {r["id"] for r in _read(p) if _yes(r["correct"]) is False}
