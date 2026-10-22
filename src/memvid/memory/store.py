"""Event-sourced record store with an incremental token count and a vector index.

Every mutation is an operation appended to `ops`:
    {"seq", "chunk_id", "op": "upsert" | "delete", "label", "record_id", "record", "reason"}
`label` is the semantic operation (insert, link, update, extend, change, compress, evict,
truncate, rewrite). The memory after chunk i is `replay(ops, upto_chunk=i)`; replaying the
whole log reproduces the live state exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import TypeAdapter

from memvid.memory.records import PREFIX, Record, RecordT
from memvid.models.embed import Embedder, TokenCounter

_ADAPTER: TypeAdapter = TypeAdapter(Record)


def parse_record(data: dict[str, Any]) -> RecordT:
    return _ADAPTER.validate_python(data)


class VectorIndex:
    """Exact cosine search over unit vectors; supports removal. Exact search is cheap at
    this scale (at most a few thousand records), so no approximate index is needed."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: dict[str, np.ndarray] = {}
        self._mat: np.ndarray | None = None
        self._ids: list[str] = []

    def set(self, rid: str, vec: np.ndarray) -> None:
        self._vecs[rid] = vec.astype(np.float32)
        self._mat = None

    def remove(self, rid: str) -> None:
        if self._vecs.pop(rid, None) is not None:
            self._mat = None

    def __len__(self) -> int:
        return len(self._vecs)

    def scores(self, q: np.ndarray) -> dict[str, float]:
        if not self._vecs:
            return {}
        if self._mat is None:
            self._ids = list(self._vecs)
            self._mat = np.stack([self._vecs[i] for i in self._ids])
        s = self._mat @ q.astype(np.float32)
        return dict(zip(self._ids, s.tolist(), strict=True))


class MemoryStore:
    def __init__(self, embedder: Embedder, counter: TokenCounter):
        self.embedder = embedder
        self.counter = counter
        self.records: dict[str, RecordT] = {}
        self.ops: list[dict[str, Any]] = []
        self.index = VectorIndex(embedder.dim)
        self._counters: dict[str, int] = {}
        self._tok: dict[str, int] = {}
        self.total_tokens = 0
        self._text_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ ids
    def new_id(self, kind: str) -> str:
        p = PREFIX[kind]
        self._counters[p] = self._counters.get(p, 0) + 1
        return f"{p}{self._counters[p]}"

    # ------------------------------------------------------------------ mutation
    def upsert(self, rec: RecordT, *, chunk_id: int, label: str, reason: str = "") -> None:
        rec = parse_record(rec.model_dump())  # validate + detach from caller's object
        old = self._tok.pop(rec.id, 0)
        n = self.counter.count(rec.line())
        self._tok[rec.id] = n
        self.total_tokens += n - old
        self.records[rec.id] = rec
        text = rec.text()
        if self._text_cache.get(rec.id) != text:
            self.index.set(rec.id, self.embedder.encode_one(text))
            self._text_cache[rec.id] = text
        self.ops.append({"seq": len(self.ops), "chunk_id": chunk_id, "op": "upsert", "label": label,
                         "record_id": rec.id, "record": rec.model_dump(mode="json"), "reason": reason})

    def delete(self, rid: str, *, chunk_id: int, label: str, reason: str = "") -> None:
        if rid not in self.records:
            raise KeyError(rid)
        del self.records[rid]
        self.total_tokens -= self._tok.pop(rid)
        self.index.remove(rid)
        self._text_cache.pop(rid, None)
        self.ops.append({"seq": len(self.ops), "chunk_id": chunk_id, "op": "delete", "label": label,
                         "record_id": rid, "record": None, "reason": reason})

    # ------------------------------------------------------------------ queries
    def get(self, rid: str) -> RecordT | None:
        return self.records.get(rid)

    def of_kind(self, kind: str) -> list[RecordT]:
        return [r for r in self.records.values() if r.kind == kind]

    def tokens(self) -> int:
        return self.total_tokens

    def tokens_of(self, rid: str) -> int:
        return self._tok[rid]

    @classmethod
    def from_records(cls, records: dict[str, RecordT], embedder: Embedder,
                     counter: TokenCounter) -> MemoryStore:
        """Read-only view of a saved memory (index and token counts rebuilt, no ops)."""
        s = cls(embedder, counter)
        texts = [r.text() for r in records.values()]
        vecs = embedder.encode(texts)
        for (rid, r), v in zip(records.items(), vecs, strict=True):
            s.records[rid] = r
            n = counter.count(r.line())
            s._tok[rid] = n
            s.total_tokens += n
            s.index.set(rid, v)
        return s


def replay(ops: list[dict[str, Any]], upto_chunk: int | None = None) -> dict[str, RecordT]:
    """State of the memory after all operations with chunk_id <= upto_chunk."""
    state: dict[str, RecordT] = {}
    for op in ops:
        if upto_chunk is not None and op["chunk_id"] > upto_chunk:
            break
        if op["op"] == "upsert":
            state[op["record_id"]] = parse_record(op["record"])
        else:
            state.pop(op["record_id"], None)
    return state
