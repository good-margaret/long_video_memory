"""Per-chunk observations: the frozen, memory-agnostic output of perception.

`ModelObservation` is what the VLM must produce (validated, then normalised);
`ChunkObservation` adds the chunk identity and the outcome of generation.
Normalisation: names and attribute keys are lower-cased and whitespace-collapsed,
unknown categories map to "other", duplicate entity names are merged, and references in
events and relations to names that are not declared entities are dropped (counted in
`dropped_refs`, reported as a perception statistic).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memvid.prompts import CATEGORIES
from memvid.timeutils import format_timestamp

_WS = re.compile(r"\s+")


def norm(text: Any) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def norm_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(text)).strip("_")


class ObsEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    category: str = "other"
    attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v):
        return norm(v)

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v):
        c = norm(v or "other")
        return c if c in CATEGORIES else "other"

    @field_validator("attributes", mode="before")
    @classmethod
    def _attrs(cls, v):
        if not isinstance(v, dict):
            return {}
        out = {}
        for k, val in v.items():
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            if isinstance(val, (list, tuple)):
                val = ", ".join(map(str, val))
            key = norm_key(k)
            if key:
                out[key] = norm(val)
        return out


class ObsEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)
    salience: int = 2

    @field_validator("description", mode="before")
    @classmethod
    def _desc(cls, v):
        return _WS.sub(" ", str(v)).strip()

    @field_validator("entities", mode="before")
    @classmethod
    def _ents(cls, v):
        return [norm(x) for x in (v or []) if str(x).strip()]

    @field_validator("salience", mode="before")
    @classmethod
    def _sal(cls, v):
        try:
            return min(3, max(1, int(v)))
        except (TypeError, ValueError):
            return 2


class ModelObservation(BaseModel):
    """Exactly what the VLM is asked to return."""

    model_config = ConfigDict(extra="ignore")

    entities: list[ObsEntity]
    events: list[ObsEvent]
    relations: list[list[str]]
    speech_summary: str

    @field_validator("relations", mode="before")
    @classmethod
    def _rels(cls, v):
        out = []
        for r in v or []:
            if isinstance(r, dict):
                r = [r.get("subject"), r.get("predicate"), r.get("object")]
            if isinstance(r, (list, tuple)) and len(r) == 3 and all(x for x in r):
                out.append([norm(x) for x in r])
        return out

    @field_validator("speech_summary", mode="before")
    @classmethod
    def _speech(cls, v):
        return "" if v is None else _WS.sub(" ", str(v)).strip()

    @model_validator(mode="after")
    def _require_content(self) -> ModelObservation:
        """Reject syntactically valid but useless empty model responses."""
        if not (self.entities or self.events or self.relations or self.speech_summary):
            raise ValueError(
                "observation is empty; extract concrete visual or spoken information"
            )
        return self


class ChunkObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    t_start: float
    t_end: float
    entities: list[ObsEntity] = Field(default_factory=list)
    events: list[ObsEvent] = Field(default_factory=list)
    relations: list[tuple[str, str, str]] = Field(default_factory=list)
    speech_summary: str = ""
    ok: bool = True
    error: str | None = None
    attempts: int = 1
    dropped_refs: int = 0

    @model_validator(mode="after")
    def _normalise(self) -> ChunkObservation:
        merged: dict[str, ObsEntity] = {}
        for e in self.entities:
            if e.name in merged:
                merged[e.name].attributes.update(e.attributes)
            else:
                merged[e.name] = e
        self.entities = list(merged.values())
        names = set(merged)
        dropped = 0
        for ev in self.events:
            kept = [n for n in dict.fromkeys(ev.entities) if n in names]
            dropped += len(ev.entities) - len(kept)
            ev.entities = kept
        rels = []
        for s, p, o in self.relations:
            if s in names and o in names:
                rels.append((s, p, o))
            else:
                dropped += 1
        self.relations = list(dict.fromkeys(rels))
        self.dropped_refs += dropped
        return self

    @classmethod
    def from_model(cls, chunk_id: int, t_start: float, t_end: float, mo: ModelObservation,
                   attempts: int) -> ChunkObservation:
        return cls(chunk_id=chunk_id, t_start=t_start, t_end=t_end, entities=mo.entities,
                   events=mo.events, relations=[tuple(r) for r in mo.relations],
                   speech_summary=mo.speech_summary, attempts=attempts)

    def to_text(self) -> str:
        """One-paragraph serialisation (flat-log record and rolling-summary input)."""
        parts = [f"{format_timestamp(self.t_start, decimals=0)}-{format_timestamp(self.t_end, decimals=0)}"]
        if self.entities:
            ents = "; ".join(
                e.name + (" (" + ", ".join(f"{k}: {v}" for k, v in e.attributes.items()) + ")"
                          if e.attributes else "")
                for e in self.entities
            )
            parts.append(f"Entities: {ents}.")
        if self.events:
            parts.append("Events: " + " ".join(
                ev.description.rstrip(".") + "." for ev in self.events))
        if self.relations:
            parts.append("Relations: " + "; ".join(" ".join(r) for r in self.relations) + ".")
        if self.speech_summary:
            parts.append(f"Speech: {self.speech_summary}")
        return " ".join(parts)
