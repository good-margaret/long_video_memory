"""Command-line interface: `memvid <command> [options]` (`memvid --help` lists all).

Each command is a small function `(cfg, args) -> result` registered in COMMANDS with its
options. `main` loads the config, calls the function and prints the result: a path as
"written: <path>", anything else as JSON. A function may return `(result, exit_code)`;
exit code 2 means "ran, but a quality gate failed". Imports are local to the functions
so that `memvid --help` stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from memvid import __version__
from memvid.config import ProjectConfig, load_config
from memvid.errors import MemvidError
from memvid.io import write_json
from memvid.logging_utils import setup_logging

log = logging.getLogger("memvid")
DEFAULT_CONFIG = "configs/default.yaml"


def gate(result: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Exit code 2 when a quality gate is not passed."""
    return result, 0 if result["accepted"] else 2


def pipeline(cfg: ProjectConfig):
    from memvid.pipeline import Pipeline

    return Pipeline(cfg)


def split_of(cfg: ProjectConfig, args: argparse.Namespace) -> str:
    return args.split or cfg.experiment.split


# ============================================================================ Stage 1
def check_env(cfg, args):
    from memvid.env_check import collect_environment, format_report

    rep = collect_environment()
    print(format_report(rep))
    write_json(Path(args.out), rep)
    return Path(args.out), 0 if rep.ok else 1


def make_test_video(cfg, args):
    from memvid.video.synthetic import make_test_video

    return make_test_video(args.out, duration_s=args.duration)


def probe(cfg, args):
    return pipeline(cfg).probe(compute_hash=not args.no_hash, force=args.force)


def extract_audio(cfg, args):
    return pipeline(cfg).extract_audio(force=args.force)


def transcribe(cfg, args):
    return {"segments": len(pipeline(cfg).transcribe(force=args.force))}


def chunk(cfg, args):
    chunks = pipeline(cfg).chunk(allow_missing_transcript=args.allow_missing_transcript,
                                 force=args.force)
    return {"chunks": len(chunks)}


def sample_frames(cfg, args):
    return {"frames": len(pipeline(cfg).sample_frames(progress=not args.quiet, force=args.force))}


def preprocess(cfg, args):
    p = pipeline(cfg)
    p.preprocess(skip_asr=args.skip_asr, progress=not args.quiet, force=args.force)
    return p.manifest().stats


def verify(cfg, args):
    problems = pipeline(cfg).verify()
    return {"problems": problems}, 0 if not problems else 2


def freeze(cfg, args):
    return {"frozen_artifacts": len(pipeline(cfg).freeze().artifacts)}


# ============================================================================ Stage 2
def questions_validate(cfg, args):
    from memvid.pipeline import video_duration
    from memvid.questions.manage import validate_questions

    rep = validate_questions(cfg, video_duration(cfg))
    return rep, 0 if rep.ok else 2


def questions_split(cfg, args):
    from memvid.questions.manage import make_split

    return dict(Counter(make_split(cfg, force=args.force).values()))


def questions_freeze(cfg, args):
    from memvid.questions.manage import freeze_questions

    return freeze_questions(cfg)


# ============================================================================ Stage 3
def perceive(cfg, args):
    from memvid.models.registry import get_vlm
    from memvid.perception.run import observation_stats, run_perception

    obs = run_perception(cfg, get_vlm(cfg), force=args.force, progress=not args.quiet,
                         limit=args.limit)
    return observation_stats(obs)


def perception_audit(cfg, args):
    from memvid.perception.run import make_perception_audit

    return make_perception_audit(cfg)


def perception_audit_summary(cfg, args):
    from memvid.perception.run import summarize_perception_audit

    return gate(summarize_perception_audit(cfg))


def perception_freeze(cfg, args):
    from memvid.perception.run import freeze_perception

    freeze_perception(cfg)
    return {"frozen": "perception"}


# ============================================================================ Stages 4-5
def budgets(cfg, args):
    from memvid.experiment import resolve_budgets

    return resolve_budgets(cfg)


def build(cfg, args):
    from memvid.experiment import build_memory

    return build_memory(cfg, args.condition, args.budget, strict=args.strict, force=args.force)


def answer(cfg, args):
    from memvid.experiment import answer_run

    return answer_run(cfg, Path(args.run), split_of(cfg, args), force=args.force)


def runs(cfg, args):
    from memvid.experiment import list_runs

    return list_runs(cfg)


def run_all(cfg, args):
    from memvid.experiment import run_all as run

    return [str(d) for d in run(cfg, split=args.split, strict=not args.no_strict, force=args.force)]


def analyze(cfg, args):
    from memvid.evaluation.analysis import analyze as run_analysis

    res = run_analysis(cfg, args.split, judge=False if args.no_judge else None)
    if res["summary"].get("warning"):
        print(f"WARNING: {res['summary']['warning']}")
    return {"accuracy": {run: r["accuracy"] for run, r in res["summary"]["runs"].items()}}


def audit_merges(cfg, args):
    from memvid.evaluation.audits import make_merge_audit

    return make_merge_audit(cfg)


def audit_merges_summary(cfg, args):
    from memvid.evaluation.audits import summarize_merge_audit

    return summarize_merge_audit(cfg)


def audit_states(cfg, args):
    from memvid.evaluation.audits import make_state_audit

    return make_state_audit(cfg)


def audit_states_summary(cfg, args):
    from memvid.evaluation.audits import summarize_state_audit

    return summarize_state_audit(cfg)


def audit_judge(cfg, args):
    from memvid.evaluation.audits import make_judge_audit

    return make_judge_audit(cfg, split_of(cfg, args))


def audit_judge_summary(cfg, args):
    from memvid.evaluation.audits import summarize_judge_audit

    return summarize_judge_audit(cfg)


def attribution(cfg, args):
    from memvid.evaluation.attribution import build_attribution, summarize_attribution

    build_attribution(cfg, split_of(cfg, args))
    return summarize_attribution(cfg)


def demo(cfg, args):
    from memvid.demo.app import build_app

    build_app(cfg).launch(share=args.share, server_port=args.port)


# ============================================================================ any stage
def status(cfg, args):
    from memvid.experiment import list_runs
    from memvid.manifest import load_manifest

    m = load_manifest(cfg.derived_dir, cfg.video.video_id)

    def state(frozen: bool, done: bool) -> str:
        return "frozen" if frozen else ("done" if done else "not run")

    return {
        "stage 1 preprocessing": state("preprocess" in m.frozen_stages, "frames_index" in m.artifacts),
        "stage 2 questions": state(bool(m.stats.get("questions", {}).get("frozen")),
                                   cfg.questions_path.is_file()),
        "stage 3 perception": state("perception" in m.frozen_stages, "observations" in m.artifacts),
        "perception audit accepted": bool(m.stats.get("perception_audit", {}).get("accepted")),
        "runs built": len(list_runs(cfg)),
        "analysis done": (cfg.video_runs_dir / "analysis" / "summary.json").is_file(),
    }


def export_tex(cfg, args):
    from memvid.reporting.results_tex import export_stage_tables
    from memvid.reporting.tex import export_all

    written = export_all(cfg, Path(args.env_report) if args.env_report else None)
    export_stage_tables(cfg)
    return [str(p) for p in written]


# ============================================================================ registry
FLAG = {"action": "store_true"}
FORCE = {"--force": {**FLAG, "help": "overwrite even if frozen or up to date"}}
QUIET = {"--quiet": FLAG}
SPLIT = {"--split": {"choices": ["dev", "test", "all"]}}
NO_CONFIG = "no_config"

# name: (handler, help, options). Options map argparse flags to their keyword arguments.
COMMANDS: dict[str, tuple[Any, str, dict[str, Any]]] = {
    "check-env": (check_env, "Check the environment and save a report",
                  {NO_CONFIG: True, "--out": {"default": "runs/environment.json"}}),
    "make-test-video": (make_test_video, "Create a synthetic test video",
                        {NO_CONFIG: True, "--out": {"default": "data/raw/smoke_test.mp4"},
                         "--duration": {"type": float, "default": 90.0}}),
    "probe": (probe, "Probe video metadata", {**FORCE, "--no-hash": FLAG}),
    "extract-audio": (extract_audio, "Extract 16 kHz mono audio", FORCE),
    "transcribe": (transcribe, "Transcribe with faster-whisper", FORCE),
    "chunk": (chunk, "Build fixed-length chunks", {**FORCE, "--allow-missing-transcript": FLAG}),
    "sample-frames": (sample_frames, "Sample frames per chunk", {**FORCE, **QUIET}),
    "preprocess": (preprocess, "Probe, extract audio, transcribe, chunk and sample frames",
                   {**FORCE, **QUIET, "--skip-asr": FLAG}),
    "verify": (verify, "Verify artifact hashes against the manifest", {}),
    "freeze": (freeze, "Freeze the Stage 1 outputs", {}),
    "questions-validate": (questions_validate, "Validate questions.yaml", {}),
    "questions-split": (questions_split, "Stratified dev/test split", FORCE),
    "questions-freeze": (questions_freeze, "Freeze questions and split", {}),
    "perceive": (perceive, "VLM observations for every chunk (resumable)",
                 {**FORCE, **QUIET, "--limit": {"type": int, "help": "at most N more chunks"}}),
    "perception-audit": (perception_audit, "Create the perception audit sheet", {}),
    "perception-audit-summary": (perception_audit_summary, "Score the perception audit", {}),
    "perception-freeze": (perception_freeze, "Freeze the observations", {}),
    "budgets": (budgets, "Resolve budgets B1..B3 in tokens (builds C4 if needed)", {}),
    "build": (build, "Build one memory run",
              {**FORCE, "--condition": {"required": True, "help": "C1|C2|C3|C4|C3-<ablation>"},
               "--budget": {"default": "B2"}, "--strict": FLAG}),
    "answer": (answer, "Answer the questions with one run",
               {**FORCE, **SPLIT, "--run": {"required": True, "help": "run folder"}}),
    "runs": (runs, "List built runs", {}),
    "run-all": (run_all, "Build and answer every planned run",
                {**FORCE, **SPLIT, "--no-strict": {**FLAG, "help": "allow unfrozen inputs"}}),
    "analyze": (analyze, "Evaluate runs, draw figures, export result tables",
                {**SPLIT, "--no-judge": FLAG}),
    "audit-merges": (audit_merges, "Create the entity-resolution audit sheet", {}),
    "audit-merges-summary": (audit_merges_summary, "Score the entity-resolution audit", {}),
    "audit-states": (audit_states, "Create the state-change audit sheet", {}),
    "audit-states-summary": (audit_states_summary, "Score the state-change audit", {}),
    "audit-judge": (audit_judge, "Create the LLM-judge audit sheet", SPLIT),
    "audit-judge-summary": (audit_judge_summary, "Score the LLM-judge audit", {}),
    "attribution": (attribution, "Build or refresh the failure-attribution sheet", SPLIT),
    "demo": (demo, "Start the Gradio demo",
             {"--share": FLAG, "--port": {"type": int, "default": 7860}}),
    "status": (status, "Show which stages are done and frozen", {}),
    "export-tex": (export_tex, "Write LaTeX tables into report/generated/", {"--env-report": {}}),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memvid", description="Memory for long videos")
    parser.add_argument("--version", action="version", version=f"memvid {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (handler, help_, options) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_, description=help_)
        needs_config = not options.get(NO_CONFIG, False)
        sp.set_defaults(handler=handler, needs_config=needs_config)
        if needs_config:
            sp.add_argument("-c", "--config", default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
        for flag, kwargs in options.items():
            if flag != NO_CONFIG:
                sp.add_argument(flag, **kwargs)
    return parser


def show(result: Any) -> None:
    if result is None:
        return
    if isinstance(result, Path):
        print(f"written: {result}")
        return
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config) if args.needs_config else None
        result = args.handler(cfg, args)
        result, code = result if isinstance(result, tuple) else (result, 0)
        show(result)
        return code
    except MemvidError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
