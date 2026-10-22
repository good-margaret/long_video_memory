"""Construction and caching of models from the config.

Models are cached per (backend, model_id, base_url): if the VLM and the text LLM are the
same checkpoint (the default), it is loaded once and serves both roles.
"""

from __future__ import annotations

import logging
from typing import Any

from memvid.config import ModelSpec, ProjectConfig
from memvid.models.base import GenerativeModel
from memvid.models.embed import (
    ApproxTokenCounter,
    Embedder,
    HashingEmbedder,
    HFTokenCounter,
    SentenceTransformerEmbedder,
    TokenCounter,
)

log = logging.getLogger(__name__)
_CACHE: dict[tuple, Any] = {}


def _build(spec: ModelSpec) -> GenerativeModel:
    key = ("gen", spec.backend, spec.model_id, spec.base_url)
    if key in _CACHE:
        return _CACHE[key]
    if spec.backend == "heuristic":
        from memvid.models.heuristic import HeuristicModel

        model: GenerativeModel = HeuristicModel()
    elif spec.backend == "openai":
        from memvid.models.openai_backend import OpenAICompatibleModel

        model = OpenAICompatibleModel(spec)
    else:
        from memvid.models.transformers_backend import TransformersModel

        model = TransformersModel(spec)
    _CACHE[key] = model
    return model


def get_vlm(cfg: ProjectConfig) -> GenerativeModel:
    return _build(cfg.models.vlm)


def get_llm(cfg: ProjectConfig) -> GenerativeModel:
    return _build(cfg.models.llm)


def get_embedder(cfg: ProjectConfig) -> Embedder:
    key = ("emb", cfg.models.embedder, cfg.models.embedder_model)
    if key not in _CACHE:
        if cfg.models.embedder == "hashing":
            _CACHE[key] = HashingEmbedder()
        else:
            _CACHE[key] = SentenceTransformerEmbedder(cfg.models.embedder_model)
    return _CACHE[key]


def get_token_counter(cfg: ProjectConfig) -> TokenCounter:
    mode = cfg.models.tokenizer
    key = ("tok", mode, cfg.models.llm.model_id)
    if key in _CACHE:
        return _CACHE[key]
    counter: TokenCounter
    if mode == "approx" or cfg.models.llm.backend == "heuristic":
        counter = ApproxTokenCounter()
    else:
        try:
            from memvid.models.transformers_backend import load_hf_tokenizer

            counter = HFTokenCounter(load_hf_tokenizer(cfg.models.llm.model_id),
                                     cfg.models.llm.model_id)
        except Exception as exc:
            if mode == "hf":
                raise
            log.warning("HF tokenizer unavailable (%s); using the approximate counter", exc)
            counter = ApproxTokenCounter()
    _CACHE[key] = counter
    return counter


def describe_models(cfg: ProjectConfig) -> dict[str, Any]:
    """Provenance record stored with every run (without loading anything new)."""
    return {
        "vlm": cfg.models.vlm.model_dump(mode="json"),
        "llm": cfg.models.llm.model_dump(mode="json"),
        "embedder": get_embedder(cfg).name,
        "tokens": get_token_counter(cfg).name,
    }


def is_heuristic(cfg: ProjectConfig) -> bool:
    return "heuristic" in (cfg.models.vlm.backend, cfg.models.llm.backend)
