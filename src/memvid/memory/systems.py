"""Construction of the memory organisations compared in the experiments.

C1 RollingSummary  one free-text record rewritten by the LLM after every chunk
C2 FlatLog         one record per observation; oldest evicted when over budget
C3 StructuredMemory typed records (memory/structured.py), plus ablations "C3-<name>"
C4 FlatLog         without a budget: the unbounded upper bound
"""

from __future__ import annotations

from memvid.config import StructuredConfig
from memvid.errors import ConfigError
from memvid.memory.base import MemorySystem
from memvid.memory.baselines import FlatLog, RollingSummary
from memvid.memory.store import MemoryStore
from memvid.memory.structured import StructuredMemory
from memvid.models.base import GenerativeModel

ABLATIONS = {
    "no_resolution": {"use_entity_resolution": False},
    "no_state_changes": {"use_state_changes": False},
    "no_compression": {"use_compression": False},
    "no_debounce": {"debounce_observations": 1},
}


def parse_condition(name: str) -> tuple[str, str | None]:
    base, _, ablation = name.partition("-")
    if base not in ("C1", "C2", "C3", "C4"):
        raise ConfigError(f"unknown condition '{name}' (use C1, C2, C3, C4 or C3-<ablation>)")
    if ablation and (base != "C3" or ablation not in ABLATIONS):
        raise ConfigError(f"unknown ablation '{name}' (ablations apply to C3: {sorted(ABLATIONS)})")
    return base, ablation or None


def build_system(name: str, store: MemoryStore, budget: int | None, cfg: StructuredConfig,
                 llm: GenerativeModel | None) -> MemorySystem:
    base, ablation = parse_condition(name)
    if base == "C4":
        sys_: MemorySystem = FlatLog(store, None)
        sys_.condition = "C4"
        return sys_
    if budget is None and base in ("C1", "C2"):
        raise ValueError(f"{name} needs a budget")
    if base == "C2":
        return FlatLog(store, budget)
    if base == "C1":
        if llm is None:
            raise ValueError("C1 needs an LLM")
        return RollingSummary(store, budget, llm)
    scfg = cfg.model_copy(update=ABLATIONS[ablation]) if ablation else cfg
    sm = StructuredMemory(store, budget, scfg, llm)
    sm.condition = name
    return sm
