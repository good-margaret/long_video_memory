"""Hybrid retrieval R(q, M) under a token allowance B_R, identical for all conditions.

score(r) = cos(q, r)                                   semantic similarity
         + entity_boost   if r is, or refers to, an entity named in the question
         + temporal_boost * w_t(r)                     if the question has a temporal cue

Temporal cues: "beginning/start/first/early" favour early records, "end/last/final/later"
favour late ones, "after X" / "before X" favour records just after / before the best match
for X. Records are taken in score order while they fit B_R, then presented to the answerer
in temporal order. The rolling summary (C1) is a single record and is always given whole.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from memvid.config import RetrievalConfig
from memvid.memory.records import Entity, RecordT, record_span, references
from memvid.memory.store import MemoryStore
from memvid.textutils import content_words

_EARLY = re.compile(r"\b(beginning|start|first|early|earliest|initially|at first|originally)\b", re.I)
_LATE = re.compile(r"\b(end|last|final|finally|later|latest|eventually)\b", re.I)
_AFTER = re.compile(r"\bafter\s+(.+?)(?:\?|,|$)", re.I)
_BEFORE = re.compile(r"\bbefore\s+(.+?)(?:\?|,|$)", re.I)


@dataclass
class Retrieved:
    records: list[RecordT]
    scores: dict[str, float]
    tokens: int
    matched_entities: list[str]
    cues: list[str]


def matched_entities(store: MemoryStore, question: str) -> list[str]:
    qw = set(content_words(question))
    out = []
    for e in store.of_kind("entity"):
        for name in e.names():  # type: ignore[union-attr]
            nw = set(content_words(name))
            if nw and nw <= qw:
                out.append(e.id)
                break
    return out


def retrieve(store: MemoryStore, question: str, cfg: RetrievalConfig, duration: float, *,
             whole_memory: bool = False) -> Retrieved:
    """whole_memory=True (used for C1): return every record, unranked."""
    recs = list(store.records.values())
    if whole_memory:
        return Retrieved(records=recs, scores={r.id: 1.0 for r in recs}, tokens=store.tokens(),
                         matched_entities=[], cues=["whole-memory"])
    if not recs:
        return Retrieved(records=[], scores={}, tokens=0, matched_entities=[], cues=[])
    sem = store.index.scores(store.embedder.encode_one(question))
    ents = set(matched_entities(store, question))
    cues: list[str] = []
    temporal: dict[str, float] = {}
    if _EARLY.search(question):
        cues.append("early")
        for r in recs:
            temporal[r.id] = temporal.get(r.id, 0.0) + 1 - record_span(r)[0] / max(duration, 1e-6)
    if _LATE.search(question):
        cues.append("late")
        for r in recs:
            temporal[r.id] = temporal.get(r.id, 0.0) + record_span(r)[1] / max(duration, 1e-6)
    for rx, direction in ((_AFTER, "after"), (_BEFORE, "before")):
        m = rx.search(question)
        if not m:
            continue
        anchor_scores = store.index.scores(store.embedder.encode_one(m.group(1)))
        anchor_id = max((r.id for r in recs if not isinstance(r, Entity)),
                        key=lambda i: anchor_scores.get(i, -1), default=None)
        if anchor_id is None:
            continue
        cues.append(f"{direction}:{anchor_id}")
        a0, a1 = record_span(store.records[anchor_id])
        for r in recs:
            s0, s1 = record_span(r)
            gap = (s0 - a1) if direction == "after" else (a0 - s1)
            if gap >= -1e-6:
                temporal[r.id] = temporal.get(r.id, 0.0) + math.exp(-gap / 300.0)
    scores: dict[str, float] = {}
    for r in recs:
        s = sem.get(r.id, 0.0)
        if r.id in ents or ents.intersection(references(r)):
            s += cfg.entity_boost
        s += cfg.temporal_boost * temporal.get(r.id, 0.0)
        scores[r.id] = s
    ranked = sorted(recs, key=lambda r: (-scores[r.id], r.id))[: cfg.top_k]
    chosen, used = [], 0
    for r in ranked:
        n = store.tokens_of(r.id)
        if used + n > cfg.budget_tokens:
            continue
        chosen.append(r)
        used += n
    chosen.sort(key=lambda r: (record_span(r)[0], r.id))
    return Retrieved(records=chosen, scores=scores, tokens=used, matched_entities=sorted(ents),
                     cues=cues)
