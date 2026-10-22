"""Typed project configuration loaded from YAML.

Every experiment is defined by one YAML file in configs/. Relative paths in the
config are resolved against the project root (the nearest ancestor directory of
the config file that contains pyproject.toml), so scripts behave identically
regardless of the current working directory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from memvid.errors import ConfigError
from memvid.io import sha256_json


class _Strict(BaseModel):
    """Reject unknown keys: a typo in a YAML file must fail loudly, not be ignored."""

    model_config = ConfigDict(extra="forbid")


class VideoConfig(_Strict):
    video_id: str = Field(description="Short identifier used for derived-data folder names.")
    filename: str = Field(description="File name of the video inside paths.raw_dir.")

    @field_validator("video_id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}", v):
            raise ValueError("video_id must be 1-64 chars of letters, digits, '_' or '-'")
        return v


class PathsConfig(_Strict):
    raw_dir: str = "data/raw"
    derived_dir: str = "data/derived"
    annotations_dir: str = "data/annotations"
    runs_dir: str = "runs"
    report_generated_dir: str = "report/generated"


class AudioConfig(_Strict):
    sample_rate: int = Field(16000, gt=0, description="Whisper expects 16 kHz.")
    channels: int = Field(1, ge=1, le=2)
    max_duration_mismatch_s: float = Field(
        0.5, ge=0, description="Tolerance between audio and video durations."
    )


class ASRConfig(_Strict):
    model_size: str = "large-v3"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = Field("auto", description="auto -> float16 on CUDA, int8 on CPU.")
    language: str | None = Field(None, description="ISO code, or null for auto-detection.")
    beam_size: int = Field(5, ge=1)
    vad_filter: bool = Field(True, description="Suppresses hallucinated text in silences.")
    condition_on_previous_text: bool = Field(
        False, description="False reduces repetition loops on long audio."
    )
    temperature_fallback: bool = Field(
        True,
        description="Library default: re-decode difficult windows at higher temperature "
        "(stochastic). False forces greedy/beam decoding at temperature 0 (deterministic).",
    )
    suspicious_no_speech_prob: float = Field(
        0.6, ge=0, le=1, description="Segments above this are flagged in statistics."
    )


class ChunkingConfig(_Strict):
    chunk_seconds: float = Field(20.0, gt=0)
    min_last_chunk_fraction: float = Field(
        0.5,
        ge=0,
        le=1,
        description="A trailing remainder shorter than this fraction of chunk_seconds is "
        "merged into the previous chunk instead of forming its own chunk.",
    )
    transcript_min_overlap_s: float = Field(
        0.25,
        ge=0,
        description="A segment is attached to a chunk if their overlap is at least "
        "min(this value, segment duration).",
    )


class FramesConfig(_Strict):
    frames_per_chunk: int = Field(8, ge=1, le=64)
    max_side: int = Field(768, ge=64, description="Longest image side after resizing.")
    jpeg_quality: int = Field(90, ge=50, le=100)


# =========================================================== Stages 2-5
class ModelSpec(_Strict):
    """One generative model. The same spec may be shared by several roles; models with
    identical (backend, model_id, base_url) are loaded once."""

    backend: Literal["transformers", "openai", "heuristic"] = Field(
        "transformers",
        description="transformers: local HF model. openai: any OpenAI-compatible server "
        "(e.g. `vllm serve`). heuristic: deterministic rule-based stand-in for smoke tests "
        "only; never use it for reported results.",
    )
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    base_url: str | None = Field(None, description="openai backend: e.g. http://localhost:8000/v1")
    api_key_env: str = Field("OPENAI_API_KEY", description="openai backend: env var with the key")
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = Field(1024, ge=16)
    guided_json: bool = Field(
        True, description="openai backend: request JSON-schema constrained decoding."
    )
    timeout_s: float = Field(300.0, gt=0)


class ModelsConfig(_Strict):
    vlm: ModelSpec = ModelSpec()
    llm: ModelSpec = ModelSpec(max_new_tokens=512)
    embedder: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer: Literal["auto", "hf", "approx"] = Field(
        "auto", description="Budget unit. auto: HF tokenizer of models.llm.model_id if "
        "transformers is installed, else the approximate counter (stated in results).",
    )


class QuestionsConfig(_Strict):
    filename: str = "questions.yaml"
    dev_size: int = Field(15, ge=0)
    numeric_rel_tol: float = Field(0.05, ge=0)
    min_per_type: int = Field(8, ge=0, description="Validation warning below this per required type.")
    required_types: list[str] = Field(
        default_factory=lambda: ["past_event", "object_change", "cross_segment"],
        description="Types the project brief requires; the others are optional additions.")


class PerceptionConfig(_Strict):
    max_pixels: int = Field(
        448 * 448, ge=64 * 64,
        description="Per-image pixel budget enforced by resizing before the VLM; fixes the "
        "visual-token cost independent of processor version.",
    )
    frames_per_chunk: int | None = Field(None, description="Subsample stored frames (None = all).")
    max_transcript_chars: int = Field(2000, ge=0)
    repair_attempts: int = Field(1, ge=0)
    audit_size: int = Field(20, ge=1)
    max_false_entity_rate: float = Field(0.15, ge=0, le=1)


class ResolutionConfig(_Strict):
    tau_low: float = Field(0.60, ge=-1, le=1)
    tau_high: float = Field(0.80, ge=-1, le=1)
    adjudicate: bool = Field(True, description="Ask the LLM in the ambiguous band.")
    generic_penalty: float = Field(0.10, ge=0)
    generic_names: list[str] = Field(
        default_factory=lambda: ["person", "man", "woman", "people", "hand", "hands", "object",
                                 "item", "thing", "tool", "container", "someone"]
    )
    top_k: int = Field(5, ge=1)

    @model_validator(mode="after")
    def _order(self) -> ResolutionConfig:
        if self.tau_low > self.tau_high:
            raise ValueError("tau_low must be <= tau_high")
        return self


class StructuredConfig(_Strict):
    resolution: ResolutionConfig = ResolutionConfig()
    debounce_observations: int = Field(
        2, ge=1, description="A changed attribute must be observed this many consecutive "
        "times (observations of that entity) before a StateChange is committed.",
    )
    volatile_attributes: list[str] = Field(default_factory=lambda: ["location", "position"])
    record_volatile_changes: bool = True
    event_dedup_tau: float = Field(0.85, ge=-1, le=1)
    event_dedup_lookback: int = Field(5, ge=1)
    compression_window_s: float = Field(300.0, gt=0)
    protect_recent_chunks: int = Field(3, ge=0)
    priority_weights: dict[str, float] = Field(
        default_factory=lambda: {"salience": 1.0, "refs": 0.5, "recency": 1.0,
                                 "state_change": 2.0, "entity": 3.0}
    )
    # ablation switches (Stage 5)
    use_entity_resolution: bool = True
    use_state_changes: bool = True
    use_compression: bool = True


class BudgetsConfig(_Strict):
    mode: Literal["fraction", "tokens"] = Field(
        "fraction", description="fraction: of the unbounded log's final size (C4)."
    )
    values: dict[str, float] = Field(default_factory=lambda: {"B1": 0.02, "B2": 0.05, "B3": 0.15})


class RetrievalConfig(_Strict):
    budget_tokens: int = Field(1500, ge=50, description="B_R, identical for all conditions.")
    top_k: int = Field(40, ge=1)
    entity_boost: float = Field(0.25, ge=0)
    temporal_boost: float = Field(0.20, ge=0)


class ExperimentConfig(_Strict):
    conditions: list[str] = Field(default_factory=lambda: ["C1", "C2", "C3", "C4"])
    ablations: list[str] = Field(
        default_factory=lambda: ["no_resolution", "no_state_changes", "no_compression",
                                 "no_debounce"]
    )
    ablation_budget: str = "B2"
    split: Literal["dev", "test", "all"] = "test"


class AnalysisConfig(_Strict):
    n_bootstrap: int = Field(10000, ge=100)
    judge_free_form: bool = Field(True, description="Use the LLM judge for short_text answers "
                                  "that fail exact matching (validated manually).")
    audit_merges: int = Field(50, ge=1)
    audit_new_entities: int = Field(50, ge=1)
    audit_state_changes: int = Field(30, ge=1)


class ProjectConfig(_Strict):
    video: VideoConfig
    paths: PathsConfig = PathsConfig()
    audio: AudioConfig = AudioConfig()
    asr: ASRConfig = ASRConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    frames: FramesConfig = FramesConfig()
    models: ModelsConfig = ModelsConfig()
    questions: QuestionsConfig = QuestionsConfig()
    perception: PerceptionConfig = PerceptionConfig()
    memory: StructuredConfig = StructuredConfig()
    budgets: BudgetsConfig = BudgetsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    seed: int = 0

    _root: Path = PrivateAttr(default_factory=Path.cwd)
    _source: Path | None = PrivateAttr(default=None)

    # ----- path helpers -------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    @property
    def source(self) -> Path | None:
        return self._source

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self._root / p)

    @property
    def video_path(self) -> Path:
        return self.resolve(self.paths.raw_dir) / self.video.filename

    @property
    def derived_dir(self) -> Path:
        """data/derived/<video_id>/ : all machine-generated artifacts for this video."""
        return self.resolve(self.paths.derived_dir) / self.video.video_id

    @property
    def annotations_dir(self) -> Path:
        """data/annotations/<video_id>/ : all human-written files for this video."""
        return self.resolve(self.paths.annotations_dir) / self.video.video_id

    @property
    def runs_dir(self) -> Path:
        return self.resolve(self.paths.runs_dir)

    @property
    def report_generated_dir(self) -> Path:
        return self.resolve(self.paths.report_generated_dir)

    @property
    def questions_path(self) -> Path:
        return self.annotations_dir / self.questions.filename

    @property
    def video_runs_dir(self) -> Path:
        """runs/<video_id>/ : one folder per (condition, budget) run plus analysis/."""
        return self.runs_dir / self.video.video_id

    def section_hash(self, *sections: str) -> str:
        """Hash of selected config sections; stored in the manifest for provenance."""
        payload = {s: getattr(self, s).model_dump(mode="json") for s in sections}
        return sha256_json(payload)[:16]


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def load_config(path: str | Path, *, root: str | Path | None = None) -> ProjectConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    try:
        cfg = ProjectConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, re-raised with the file name
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc
    cfg._root = Path(root).resolve() if root else find_project_root(path.parent)
    cfg._source = path
    return cfg
