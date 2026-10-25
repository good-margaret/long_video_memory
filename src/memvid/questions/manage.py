"""Loading, validation, dev/test split and freezing of the question set.

Files (data/annotations/<video_id>/):
  questions.yaml   human-written questions (never rewritten by code)
  split.json       {question_id: "dev" | "test"} written by `memvid questions-split`
Freezing stores the SHA-256 of the question file and split.json in the derived manifest;
afterwards loading fails if either file changed. Each question file (questions.filename,
e.g. questions.yaml and the audited questions_v2.yaml) is frozen separately, so a corrected
set can be added without touching the frozen original. A later set may drop questions:
split.json must cover every question, extra ids in it are unused.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from memvid.config import ProjectConfig
from memvid.errors import ConfigError, MemvidError, PipelineStateError
from memvid.io import read_json, sha256_file, write_json
from memvid.manifest import load_manifest, save_manifest
from memvid.questions.schema import QUESTION_TYPES, Question, QuestionFile


class ValidationReport(BaseModel):
    n_questions: int
    by_type: dict[str, int]
    by_answer_type: dict[str, int]
    by_modality: dict[str, int]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def split_path(cfg: ProjectConfig) -> Path:
    return cfg.annotations_dir / "split.json"


def load_question_file(path: Path) -> QuestionFile:
    if not path.is_file():
        raise ConfigError(f"Question file not found: {path} (copy templates/questions.yaml)")
    try:
        return QuestionFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ConfigError(f"Invalid question file {path}:\n{exc}") from exc


def validate_questions(cfg: ProjectConfig, duration_s: float) -> ValidationReport:
    qf = load_question_file(cfg.questions_path)
    errors, warnings = [], []
    if qf.video_id != cfg.video.video_id:
        errors.append(f"video_id '{qf.video_id}' != config '{cfg.video.video_id}'")
    for q in qf.questions:
        for a, b in q.gt_intervals:
            if b > duration_s + 1e-6:
                errors.append(f"{q.id}: interval [{a}, {b}] ends after the video ({duration_s:.1f} s)")
        if q.evidence_time is not None and q.evidence_time > duration_s:
            errors.append(f"{q.id}: evidence_time after the end of the video")
    by_type = Counter(q.type for q in qf.questions)
    unknown = sorted(set(cfg.questions.required_types) - set(QUESTION_TYPES))
    if unknown:
        errors.append(f"questions.required_types contains unknown types: {unknown}")
    for t in cfg.questions.required_types:
        if by_type.get(t, 0) < cfg.questions.min_per_type:
            warnings.append(f"only {by_type.get(t, 0)} '{t}' questions (< {cfg.questions.min_per_type}); "
                            f"this type is required by the brief")
    early = sum(1 for q in qf.questions if min(a for a, _ in q.gt_intervals) < 600)
    if early < 0.15 * max(1, len(qf.questions)):
        warnings.append(f"only {early} questions have evidence in the first 10 minutes")
    quart = Counter(min(int(4 * (q.evidence_time or 0) / duration_s), 3) for q in qf.questions)
    if len(quart) < 4:
        warnings.append(f"evidence covers only quartiles {sorted(quart)} of the video")
    return ValidationReport(
        n_questions=len(qf.questions),
        by_type=dict(sorted(by_type.items())),
        by_answer_type=dict(sorted(Counter(q.answer_type for q in qf.questions).items())),
        by_modality=dict(sorted(Counter(q.modality for q in qf.questions).items())),
        errors=errors,
        warnings=warnings,
    )


def make_split(cfg: ProjectConfig, *, force: bool = False) -> dict[str, str]:
    """Stratified by question type: about dev_size questions in dev, proportionally."""
    path = split_path(cfg)
    if path.exists() and not force:
        raise MemvidError(f"{path} exists; pass --force to re-split (only before freezing)")
    _refuse_if_frozen(cfg)
    qf = load_question_file(cfg.questions_path)
    rng = random.Random(cfg.seed)
    n_total = len(qf.questions)
    n_dev = min(cfg.questions.dev_size, n_total)
    by_type = {t: sorted(q.id for q in qf.questions if q.type == t) for t in QUESTION_TYPES}
    # largest-remainder allocation: exactly n_dev dev questions, proportional per type
    raw = {t: n_dev * len(ids) / n_total for t, ids in by_type.items()} if n_total else {}
    quota = {t: int(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: (-(raw[t] - quota[t]), t))[: n_dev - sum(quota.values())]:
        quota[t] += 1
    split: dict[str, str] = {}
    for t, ids in by_type.items():
        rng.shuffle(ids)
        for i, qid in enumerate(ids):
            split[qid] = "dev" if i < quota.get(t, 0) else "test"
    write_json(path, dict(sorted(split.items())))
    return split


def load_split(cfg: ProjectConfig) -> dict[str, str]:
    path = split_path(cfg)
    if not path.is_file():
        raise PipelineStateError("split.json missing: run `memvid questions-split` first")
    return read_json(path)


def load_questions(cfg: ProjectConfig, split: str = "all", *, require_frozen: bool = False) -> list[Question]:
    _check_frozen_hashes(cfg, require=require_frozen)
    qs = load_question_file(cfg.questions_path).questions
    if split == "all":
        return qs
    assignment = load_split(cfg)
    missing = [q.id for q in qs if q.id not in assignment]
    if missing:
        raise MemvidError(f"questions without a split: {missing}; re-run questions-split")
    return [q for q in qs if assignment[q.id] == split]


# ----------------------------------------------------------------------------- freezing
LEGACY_FILENAME = "questions.yaml"  # frozen before per-file freezing, under stats["questions"]


def _frozen_info(m, filename: str) -> dict[str, Any] | None:
    info = m.stats.get("question_sets", {}).get(filename)
    if info is None and filename == LEGACY_FILENAME:
        info = m.stats.get("questions")
    return info


def _refuse_if_frozen(cfg: ProjectConfig) -> None:
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    frozen = [k for k, v in m.stats.get("question_sets", {}).items() if v.get("frozen")]
    if m.stats.get("questions", {}).get("frozen"):
        frozen.append(LEGACY_FILENAME)
    if frozen:
        raise MemvidError(f"question sets {sorted(set(frozen))} are frozen; changing the split "
                          "would invalidate their results")


def freeze_questions(cfg: ProjectConfig) -> dict[str, Any]:
    qf = load_question_file(cfg.questions_path)
    split = load_split(cfg)
    ids = {q.id for q in qf.questions}
    if ids - set(split):
        raise MemvidError(f"questions without a split: {sorted(ids - set(split))}; re-run questions-split")
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    filename = cfg.questions.filename
    previous = _frozen_info(m, filename)
    info = {
        "frozen": True,
        "filename": filename,
        "sha256_questions": sha256_file(cfg.questions_path),
        "sha256_split": sha256_file(split_path(cfg)),
        "n_questions": len(qf.questions),
        "n_dev": sum(1 for q in qf.questions if split[q.id] == "dev"),
        "n_test": sum(1 for q in qf.questions if split[q.id] == "test"),
        "by_type": dict(sorted(Counter(q.type for q in qf.questions).items())),
    }
    if previous and previous.get("frozen") and (
            previous["sha256_questions"] != info["sha256_questions"]
            or previous["sha256_split"] != info["sha256_split"]):
        raise MemvidError(f"{filename} is already frozen with different content; give a changed "
                          "set a new filename (questions.filename) instead of editing it")
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    m.stats.setdefault("question_sets", {})[filename] = info
    if filename == LEGACY_FILENAME:
        m.stats["questions"] = info
    save_manifest(cfg.derived_dir, m)
    return info


def _check_frozen_hashes(cfg: ProjectConfig, *, require: bool) -> None:
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    info = _frozen_info(m, cfg.questions.filename)
    if not info or not info.get("frozen"):
        if require:
            raise PipelineStateError(f"the question set {cfg.questions.filename} is not frozen: "
                                     "run `memvid questions-freeze`")
        return
    if sha256_file(cfg.questions_path) != info["sha256_questions"]:
        raise MemvidError(f"{cfg.questions.filename} changed after it was frozen")
    if sha256_file(split_path(cfg)) != info["sha256_split"]:
        raise MemvidError("split.json changed after it was frozen")
