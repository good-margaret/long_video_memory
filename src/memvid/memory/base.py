"""Common interface of all memory organisations.

Each organisation reads ChunkObservations in order and writes to a MemoryStore, which
provides the token count, the provenance of every record and the operation log.
"""

from __future__ import annotations

from typing import Any

from memvid.memory.store import MemoryStore
from memvid.perception.schema import ChunkObservation


class MemorySystem:
    condition = "base"

    def __init__(self, store: MemoryStore, budget: int | None):
        self.store = store
        self.budget = budget
        self.now = 0.0
        self.chunk_bounds: dict[int, tuple[float, float]] = {}
        self.sizes: list[dict[str, Any]] = []
        self.counters: dict[str, int] = {}

    def _count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def process(self, obs: ChunkObservation) -> None:
        self.chunk_bounds[obs.chunk_id] = (obs.t_start, obs.t_end)
        self.now = obs.t_end
        if obs.ok:
            self._process(obs)
        else:
            self._count("failed_observations")
        self.sizes.append({"chunk_id": obs.chunk_id, "tokens": self.store.tokens(),
                           "n_records": len(self.store.records)})

    def _process(self, obs: ChunkObservation) -> None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"condition": self.condition, "budget": self.budget, "counters": self.counters}
