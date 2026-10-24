"""Memory records and their one-line serialisation sigma(r).

Every record carries `source_chunks`, the provenance set pi(r): the chunks it was derived
from. Merging and compression take the union of provenance sets; it is never dropped.

The serialisation of a record does not depend on any other record (names are copied into
records that reference entities), so per-record token counts can be cached and the memory
size |M|_tau = sum_r |tau(sigma(r))| is maintained incrementally and exactly.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memvid.timeutils import format_timestamp


def ts(t: float) -> str:
    return format_timestamp(t, decimals=0)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_chunks: list[int] = Field(min_length=1)

    @field_validator("source_chunks")
    @classmethod
    def _sorted(cls, v: list[int]) -> list[int]:
        return sorted(set(v))

    # time used for recency and ordering
    @property
    def t_ref(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def text(self) -> str:
        """Content without id and times (used for embeddings)."""
        raise NotImplementedError

    def line(self) -> str:
        raise NotImplementedError


class Entity(_Record):
    kind: Literal["entity"] = "entity"
    name: str
    category: str = "other"
    attributes: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    first_seen: float
    last_seen: float

    @property
    def t_ref(self) -> float:
        return self.last_seen

    @property
    def t_start(self) -> float:
        return self.first_seen

    def names(self) -> list[str]:
        return [self.name, *self.aliases]

    def text(self) -> str:
        attrs = "; ".join(f"{k}={v}" for k, v in sorted(self.attributes.items()))
        aka = f" aka {', '.join(self.aliases)}" if self.aliases else ""
        return f"{self.name} ({self.category}){aka}" + (f": {attrs}" if attrs else "")

    def line(self) -> str:
        return f"[{self.id}] entity {self.text()} | seen {ts(self.first_seen)}-{ts(self.last_seen)}"


class Event(_Record):
    kind: Literal["event"] = "event"
    t_start: float
    t_end: float
    description: str
    entity_ids: list[str] = Field(default_factory=list)
    salience: int = Field(2, ge=1, le=3)
    modality: Literal["visual", "speech"] = "visual"
    compressed: bool = False   # handled by first-level compression (kept or summarised)
    summarized: bool = False   # text written by the compression step (replaces originals)
    level: int = Field(0, ge=0)  # 0 original; L >= 1 has passed L rounds of compression

    @property
    def t_ref(self) -> float:
        return self.t_end

    def text(self) -> str:
        return self.description

    def line(self) -> str:
        label = "speech" if self.modality == "speech" else ("summary" if self.summarized else "event")
        ents = f" {{{','.join(self.entity_ids)}}}" if self.entity_ids else ""
        return f"[{self.id}] {ts(self.t_start)}-{ts(self.t_end)} {label}: {self.description}{ents}"


class StateChange(_Record):
    kind: Literal["state_change"] = "state_change"
    entity_id: str
    entity_name: str
    attr: str
    old: str
    new: str
    t: float

    @property
    def t_ref(self) -> float:
        return self.t

    @property
    def t_start(self) -> float:
        return self.t

    def text(self) -> str:
        return f"{self.entity_name} {self.attr} changed from {self.old} to {self.new}"

    def line(self) -> str:
        return (f"[{self.id}] {ts(self.t)} change: {self.entity_name} [{self.entity_id}] "
                f"{self.attr}: {self.old} -> {self.new}")


class Relation(_Record):
    kind: Literal["relation"] = "relation"
    subj: str
    subj_name: str
    pred: str
    obj: str
    obj_name: str
    t: float

    @property
    def t_ref(self) -> float:
        return self.t

    @property
    def t_start(self) -> float:
        return self.t

    def text(self) -> str:
        return f"{self.subj_name} {self.pred} {self.obj_name}"

    def line(self) -> str:
        return (f"[{self.id}] {ts(self.t)} relation: {self.subj_name} [{self.subj}] {self.pred} "
                f"{self.obj_name} [{self.obj}]")


class Note(_Record):
    """Free-text record: the rolling summary (C1) and flat-log entries (C2, C4)."""

    kind: Literal["note"] = "note"
    t_start: float
    t_end: float
    body: str

    @property
    def t_ref(self) -> float:
        return self.t_end

    def text(self) -> str:
        return self.body

    def line(self) -> str:
        return f"[{self.id}] {ts(self.t_start)}-{ts(self.t_end)} note: {self.body}"


Record = Annotated[Entity | Event | StateChange | Relation | Note, Field(discriminator="kind")]
RecordT = Entity | Event | StateChange | Relation | Note
PREFIX = {"entity": "E", "event": "V", "state_change": "S", "relation": "R", "note": "N"}


def record_span(r: RecordT) -> tuple[float, float]:
    if isinstance(r, Entity):
        return r.first_seen, r.last_seen
    if isinstance(r, (Event, Note)):
        return r.t_start, r.t_end
    return r.t, r.t


def references(r: RecordT) -> list[str]:
    """Entity ids a record refers to."""
    if isinstance(r, Event):
        return list(r.entity_ids)
    if isinstance(r, StateChange):
        return [r.entity_id]
    if isinstance(r, Relation):
        return [r.subj, r.obj]
    return []


def record_overlaps(r: RecordT, bounds: dict[int, tuple[float, float]],
                    intervals: list[tuple[float, float]]) -> bool:
    for c in r.source_chunks:
        u, v = bounds[c]
        if any(min(v, b) - max(u, a) > 0 for a, b in intervals):
            return True
    return False
