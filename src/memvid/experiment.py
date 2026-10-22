"""Stage 5 runner: build memories for every (condition, budget), answer the questions.

Layout (runs/<video_id>/):
  budgets.json                 token budgets B1..B3 (resolved from the unbounded log C4)
  <condition>_<budget>/        e.g. C3_B2, C3-no_resolution_B2, C4_full
      memory.sqlite            final records, operation log, resolution log, size trace
      run_info.json            models, config hashes, observation hash, counters, timing
      answers_<split>.jsonl    one AnswerRecord per question
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from memvid import __version__
from memvid.config import ProjectConfig
from memvid.errors import MemvidError, PipelineStateError
from memvid.io import read_json, read_jsonl, sha256_file, sha256_json, write_json, write_jsonl
from memvid.manifest import load_manifest
from memvid.memory.answer import AnswerRecord, answer_question
from memvid.memory.persist import load_memory, save_memory
from memvid.memory.retrieval import retrieve
from memvid.memory.store import MemoryStore
from memvid.memory.structured import StructuredMemory
from memvid.memory.systems import build_system, parse_condition
from memvid.models.registry import describe_models, get_embedder, get_llm, get_token_counter, is_heuristic
from memvid.perception.run import load_observations
from memvid.pipeline import video_duration
from memvid.prompts import PROMPT_VERSION
from memvid.questions.manage import load_questions

log = logging.getLogger(__name__)
FULL = "full"


def run_dir(cfg: ProjectConfig, condition: str, budget: str) -> Path:
    return cfg.video_runs_dir / f"{condition}_{budget}"


def budgets_path(cfg: ProjectConfig) -> Path:
    return cfg.video_runs_dir / "budgets.json"


def _observations_sha(cfg: ProjectConfig) -> str | None:
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    rec = m.artifacts.get("observations")
    return rec.sha256 if rec else None


def run_signature(cfg: ProjectConfig, condition: str, budget_tokens: int | None) -> str:
    """Everything that determines a memory run, and nothing else. A run whose stored
    signature differs is stale and is rebuilt. Flat logs (C2, C4) depend only on the
    observations and the token counter, so tuning C3 on the dev split never rebuilds C4
    (and with it the budgets) or the LLM-written rolling summaries (C1)."""
    base, _ = parse_condition(condition)
    inputs = {
        "condition": condition, "budget_tokens": budget_tokens,
        "observations": _observations_sha(cfg),
        "tokenizer": [cfg.models.tokenizer, cfg.models.llm.backend, cfg.models.llm.model_id],
        "prompt_version": PROMPT_VERSION, "package": __version__,
    }
    if base in ("C1", "C3"):
        inputs["llm"] = cfg.models.llm.model_dump(mode="json")
    if base == "C3":
        inputs["memory"] = cfg.memory.model_dump(mode="json")
        inputs["embedder"] = [cfg.models.embedder, cfg.models.embedder_model]
    return sha256_json(inputs)[:16]


def _clear_downstream(out: Path) -> None:
    for f in list(out.glob("answers_*")) + list(out.glob("eval_*")):
        f.unlink()


def build_memory(cfg: ProjectConfig, condition: str, budget_name: str, *, strict: bool = False,
                 force: bool = False) -> Path:
    base, _ = parse_condition(condition)
    out = run_dir(cfg, condition, FULL if base == "C4" else budget_name)
    budget_tokens = None
    if base != "C4":
        budgets = resolve_budgets(cfg)
        if budget_name not in budgets:
            raise MemvidError(f"unknown budget {budget_name}; known: {sorted(budgets)}")
        budget_tokens = budgets[budget_name]
    signature = run_signature(cfg, condition, budget_tokens)
    if (out / "memory.sqlite").exists() and not force:
        old = read_json(out / "run_info.json").get("signature") if (out / "run_info.json").is_file() else None
        if old == signature:
            log.info("%s is up to date; skipping (use --force to rebuild)", out.name)
            return out
        log.warning("%s is stale (configuration, models, observations or budget changed); "
                    "rebuilding", out.name)
    obs = load_observations(cfg, require_frozen=strict)
    embedder, counter = get_embedder(cfg), get_token_counter(cfg)
    llm = get_llm(cfg) if base in ("C1", "C3") else None
    store = MemoryStore(embedder, counter)
    system = build_system(condition, store, budget_tokens, cfg.memory, llm)
    t0 = time.perf_counter()
    for o in obs:
        system.process(o)
    wall = time.perf_counter() - t0
    resolution = system.resolution_log if isinstance(system, StructuredMemory) else []
    m = load_manifest(cfg.derived_dir, cfg.video.video_id)
    meta = {
        "signature": signature,
        "condition": condition, "budget_name": FULL if base == "C4" else budget_name,
        "budget_tokens": budget_tokens, "final_tokens": store.tokens(),
        "n_records": len(store.records), "n_ops": len(store.ops),
        "observations_sha256": m.artifacts["observations"].sha256 if "observations" in m.artifacts else None,
        "memory_config_hash": cfg.section_hash("memory", "models"),
        "system": system.describe(), "models": describe_models(cfg),
        "heuristic_models": is_heuristic(cfg), "wall_time_s": round(wall, 2),
    }
    if out.is_dir():
        _clear_downstream(out)
    save_memory(out / "memory.sqlite", meta=meta, records=store.records, ops=store.ops,
                resolution=resolution, sizes=system.sizes, bounds=system.chunk_bounds)
    write_json(out / "run_info.json", meta)
    log.info("%s: %d records, %d tokens (budget %s), %.1fs", out.name, len(store.records),
             store.tokens(), budget_tokens, wall)
    return out


def resolve_budgets(cfg: ProjectConfig) -> dict[str, int]:
    """Budgets in tokens. In 'fraction' mode they are fractions of the final size of the
    unbounded log C4, which is built first if needed."""
    path = budgets_path(cfg)
    spec = {**cfg.budgets.model_dump(mode="json"), "observations": _observations_sha(cfg),
            "c4_signature": run_signature(cfg, "C4", None)}
    if path.is_file():
        saved = read_json(path)
        if saved.get("spec") == spec:
            return saved["tokens"]
    if cfg.budgets.mode == "tokens":
        tokens = {k: int(v) for k, v in cfg.budgets.values.items()}
        ref = None
    else:
        c4 = build_memory(cfg, "C4", FULL)
        ref = read_json(c4 / "run_info.json")["final_tokens"]
        tokens = {k: max(50, round(v * ref)) for k, v in cfg.budgets.values.items()}
    write_json(path, {"spec": spec, "reference_tokens": ref, "tokens": tokens})
    return tokens


def answer_signature(cfg: ProjectConfig, rdir: Path, split: str) -> str:
    q = cfg.questions_path
    return sha256_json({
        "memory": sha256_file(rdir / "memory.sqlite"), "split": split,
        "questions": sha256_file(q) if q.is_file() else None,
        "split_file": sha256_file(cfg.annotations_dir / "split.json")
        if (cfg.annotations_dir / "split.json").is_file() else None,
        "retrieval": cfg.retrieval.model_dump(mode="json"),
        "llm": cfg.models.llm.model_dump(mode="json"), "embedder": cfg.models.embedder,
        "embedder_model": cfg.models.embedder_model, "tokenizer": cfg.models.tokenizer,
        "prompt_version": PROMPT_VERSION, "package": __version__,
    })[:16]


def answer_run(cfg: ProjectConfig, rdir: Path, split: str, *, force: bool = False) -> Path:
    out = rdir / f"answers_{split}.jsonl"
    meta_path = rdir / f"answers_{split}.meta.json"
    signature = answer_signature(cfg, rdir, split)
    if out.exists() and not force and meta_path.is_file() and \
            read_json(meta_path).get("signature") == signature:
        return out
    saved = load_memory(rdir / "memory.sqlite")
    base, _ = parse_condition(saved.meta["condition"])
    store = MemoryStore.from_records(saved.records, get_embedder(cfg), get_token_counter(cfg))
    llm = get_llm(cfg)
    duration = video_duration(cfg)
    answers: list[AnswerRecord] = []
    for q in load_questions(cfg, split):
        ret = retrieve(store, q.question, cfg.retrieval, duration, whole_memory=(base == "C1"))
        answers.append(answer_question(q, ret, llm, saved.bounds))
    write_jsonl(out, answers)
    write_json(meta_path, {"signature": signature, "n": len(answers)})
    for f in rdir.glob(f"eval_{split}*"):
        f.unlink()  # evaluation of the previous answers is stale
    return out


def load_answers(rdir: Path, split: str) -> list[AnswerRecord]:
    p = rdir / f"answers_{split}.jsonl"
    if not p.is_file():
        raise PipelineStateError(f"{p} missing: run `memvid answer` first")
    return read_jsonl(p, AnswerRecord)


def planned_runs(cfg: ProjectConfig) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for c in cfg.experiment.conditions:
        if parse_condition(c)[0] == "C4":
            runs.append((c, FULL))
        else:
            runs += [(c, b) for b in cfg.budgets.values]
    runs += [(f"C3-{a}", cfg.experiment.ablation_budget) for a in cfg.experiment.ablations]
    return list(dict.fromkeys(runs))


def run_all(cfg: ProjectConfig, *, split: str | None = None, strict: bool = True,
            force: bool = False) -> list[Path]:
    split = split or cfg.experiment.split
    load_questions(cfg, split, require_frozen=strict)  # fail early if not frozen
    resolve_budgets(cfg)
    dirs = []
    for cond, b in planned_runs(cfg):
        d = build_memory(cfg, cond, b, strict=strict, force=force)
        answer_run(cfg, d, split, force=force)
        dirs.append(d)
    return dirs


def list_runs(cfg: ProjectConfig) -> list[dict[str, Any]]:
    out = []
    root = cfg.video_runs_dir
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if (d / "run_info.json").is_file():
            info = read_json(d / "run_info.json")
            out.append({"run": d.name, "path": str(d), "condition": info["condition"],
                        "budget": info["budget_name"], "tokens": info["final_tokens"],
                        "records": info["n_records"]})
    return out
