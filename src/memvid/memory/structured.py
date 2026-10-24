"""C3: structured memory of entities, events, state changes and relations.

Per observation: entity resolution (link or create), debounced state changes, event and
speech insertion with lossless deduplication, relations, then budget enforcement
(compress the oldest window of events, else delete the lowest-priority record).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from memvid.config import StructuredConfig
from memvid.memory.base import MemorySystem
from memvid.memory.records import Entity, Event, RecordT, Relation, StateChange, references
from memvid.memory.store import MemoryStore
from memvid.models.base import GenerativeModel
from memvid.models.jsonutil import generate_validated
from memvid.perception.schema import ChunkObservation, norm
from memvid.prompts import ADJUDICATE, COMPRESS
from memvid.textutils import adds_information
from memvid.timeutils import format_timestamp


class _Verdict(BaseModel):
    same: bool


class _CompressedItem(BaseModel):
    description: str = Field(min_length=1)
    entity_ids: list[str] = Field(default_factory=list)
    salience: int = Field(2, ge=1, le=3)


class _Compressed(BaseModel):
    summaries: list[_CompressedItem] = Field(min_length=1, max_length=3)


@dataclass
class _Pending:
    value: str
    count: int
    chunks: list[int]
    t: float


@dataclass
class _ChunkContext:
    chunk_id: int
    name_to_id: dict[str, str] = field(default_factory=dict)
    linked: set[str] = field(default_factory=set)


def _same_value(a: str, b: str) -> bool:
    return norm(a) == norm(b)


class StructuredMemory(MemorySystem):
    condition = "C3"

    def __init__(self, store: MemoryStore, budget: int | None, cfg: StructuredConfig,
                 llm: GenerativeModel | None):
        super().__init__(store, budget)
        self.cfg = cfg
        self.llm = llm
        self.pending: dict[tuple[str, str], _Pending] = {}
        self.resolution_log: list[dict[str, Any]] = []
        self._rel_key: dict[tuple[str, str, str], str] = {}
        self._name_vecs: dict[str, np.ndarray] = {}
        self._chunk_len = 20.0

    # ------------------------------------------------------------------ main
    def _process(self, obs: ChunkObservation) -> None:
        self._chunk_len = obs.t_end - obs.t_start
        ctx = _ChunkContext(chunk_id=obs.chunk_id)
        self._resolve_entities(obs, ctx)
        self._insert_events(obs, ctx)
        self._insert_relations(obs, ctx)
        if self.budget is not None:
            self._enforce_budget(obs.chunk_id)

    def _vec(self, text: str) -> np.ndarray:
        v = self._name_vecs.get(text)
        if v is None:
            v = self.store.embedder.encode_one(text)
            self._name_vecs[text] = v
        return v

    # ------------------------------------------------------------------ 7.1 entities
    def _similarity(self, local: str, ent: Entity) -> float:
        q = self._vec(local)
        sim = max(float(q @ self._vec(n)) for n in ent.names())
        generic = set(self.cfg.resolution.generic_names)
        if local in generic or ent.name in generic:
            sim -= self.cfg.resolution.generic_penalty
        return sim

    def _adjudicate(self, local, ent: Entity) -> bool | None:
        if self.llm is None or not self.cfg.resolution.adjudicate:
            return None

        def describe(name, category, attrs, extra=""):
            a = "; ".join(f"{k}={v}" for k, v in sorted(attrs.items())) or "none"
            return f"name: {name}\ncategory: {category}\nattributes: {a}{extra}"

        a = describe(ent.name, ent.category, ent.attributes,
                     f"\nalso called: {', '.join(ent.aliases) or '-'}\n"
                     f"last seen: {format_timestamp(ent.last_seen, decimals=0)}")
        b = describe(local.name, local.category, local.attributes,
                     f"\nseen: {format_timestamp(self.now, decimals=0)}")
        res = generate_validated(self.llm, ADJUDICATE.format(a=a, b=b), _Verdict, max_new_tokens=32)
        self._count("adjudications")
        if not res.ok:
            self._count("adjudication_failures")
            return None
        return res.value.same

    def _resolve_entities(self, obs: ChunkObservation, ctx: _ChunkContext) -> None:
        rc = self.cfg.resolution
        existing = self.store.of_kind("entity")
        locals_ = obs.entities
        assigned: dict[int, tuple[str, str, float]] = {}  # local index -> (entity id, method, sim)
        best: dict[int, tuple[str | None, float]] = {i: (None, -1.0) for i in range(len(locals_))}
        if self.cfg.use_entity_resolution and existing:
            pairs = []
            for i, e in enumerate(locals_):
                scored = []
                for ent in existing:
                    if e.category != ent.category and "other" not in (e.category, ent.category):
                        continue
                    scored.append((self._similarity(e.name, ent), ent.id))
                scored.sort(reverse=True)
                for sim, eid in scored[: rc.top_k]:
                    if sim > best[i][1]:
                        best[i] = (eid, sim)
                    if sim >= rc.tau_low:
                        pairs.append((sim, i, eid))
            pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
            claimed: set[str] = set()
            for sim, i, eid in pairs:
                if i in assigned or eid in claimed:
                    continue
                if sim >= rc.tau_high:
                    method = "auto_link"
                else:
                    verdict = self._adjudicate(locals_[i], self.store.get(eid))
                    if verdict is None:
                        if sim < (rc.tau_low + rc.tau_high) / 2:
                            self._log(obs, locals_[i].name, "band_new", eid, sim)
                            continue
                        method = "midpoint_link"
                    elif verdict:
                        method = "adjudicated_link"
                    else:
                        self._log(obs, locals_[i].name, "adjudicated_new", eid, sim)
                        continue
                assigned[i] = (eid, method, sim)
                claimed.add(eid)

        for i, e in enumerate(locals_):
            if i in assigned:
                eid, method, sim = assigned[i]
                ent: Entity = self.store.get(eid)  # type: ignore[assignment]
                aliases = list(ent.aliases)
                if e.name != ent.name and e.name not in aliases:
                    aliases.append(e.name)
                upd = ent.model_copy(update={
                    "aliases": aliases, "last_seen": obs.t_end,
                    "source_chunks": [*ent.source_chunks, obs.chunk_id],
                    "attributes": dict(ent.attributes)})
                self._update_attributes(upd, e.attributes, obs)
                self.store.upsert(upd, chunk_id=obs.chunk_id, label="link",
                                  reason=f"{method} '{e.name}' sim={sim:.3f}")
                ctx.name_to_id[e.name] = eid
                ctx.linked.add(eid)
                self._log(obs, e.name, method, eid, sim)
            else:
                ent = Entity(id=self.store.new_id("entity"), source_chunks=[obs.chunk_id],
                             name=e.name, category=e.category, attributes=dict(e.attributes),
                             first_seen=obs.t_start, last_seen=obs.t_end)
                self.store.upsert(ent, chunk_id=obs.chunk_id, label="insert")
                ctx.name_to_id[e.name] = ent.id
                cand, sim = best[i]
                self._log(obs, e.name, "new", ent.id, sim, candidate=cand)

    def _log(self, obs, local, decision, eid, sim, candidate=None) -> None:
        ent = self.store.get(eid) if eid else None
        cand = self.store.get(candidate) if candidate else None
        self.resolution_log.append({
            "chunk_id": obs.chunk_id, "t": obs.t_start, "local_name": local, "decision": decision,
            "entity_id": eid, "entity_name": getattr(ent, "name", None),
            "candidate_id": candidate, "candidate_name": getattr(cand, "name", None),
            "sim": round(float(sim), 4),
        })
        self._count(f"resolution_{decision}")

    # ------------------------------------------------------------------ 7.2 state changes
    def _update_attributes(self, ent: Entity, observed: dict[str, str], obs: ChunkObservation) -> None:
        for attr, val in observed.items():
            cur = ent.attributes.get(attr)
            key = (ent.id, attr)
            if cur is None:
                ent.attributes[attr] = val
                continue
            if _same_value(cur, val):
                self.pending.pop(key, None)
                continue
            if not self.cfg.use_state_changes:
                ent.attributes[attr] = val
                continue
            p = self.pending.get(key)
            if p is not None and _same_value(p.value, val):
                p.count += 1
                p.chunks.append(obs.chunk_id)
            else:
                p = _Pending(value=val, count=1, chunks=[obs.chunk_id], t=obs.t_start)
                self.pending[key] = p
            if p.count < self.cfg.debounce_observations:
                continue
            del self.pending[key]
            ent.attributes[attr] = val
            if attr in self.cfg.volatile_attributes and not self.cfg.record_volatile_changes:
                continue
            sc = StateChange(id=self.store.new_id("state_change"), source_chunks=p.chunks,
                             entity_id=ent.id, entity_name=ent.name, attr=attr, old=cur, new=val,
                             t=p.t)
            self.store.upsert(sc, chunk_id=obs.chunk_id, label="change",
                              reason=f"observed {p.count}x")
            self._count("state_changes")

    # ------------------------------------------------------------------ 7.3 events
    def _insert_events(self, obs: ChunkObservation, ctx: _ChunkContext) -> None:
        items = [(ev.description, [ctx.name_to_id[n] for n in ev.entities if n in ctx.name_to_id],
                  ev.salience, "visual") for ev in obs.events]
        for speech in (obs.speech_summary, f'said: "{obs.transcript}"' if obs.transcript else ""):
            if not speech:
                continue
            low = speech.lower()
            ids = [eid for name, eid in ctx.name_to_id.items()
                   if re.search(rf"\b{re.escape(name)}\b", low)]
            items.append((speech, ids, 2, "speech"))
        for desc, ids, sal, modality in items:
            ids = list(dict.fromkeys(ids))
            target = self._find_duplicate(desc, ids, modality, obs)
            if target is not None:
                upd = target.model_copy(update={
                    "t_end": max(target.t_end, obs.t_end), "salience": max(target.salience, sal),
                    "source_chunks": [*target.source_chunks, obs.chunk_id]})
                self.store.upsert(upd, chunk_id=obs.chunk_id, label="extend",
                                  reason="near-duplicate of the previous event")
                self._count("events_extended")
                continue
            ev = Event(id=self.store.new_id("event"), source_chunks=[obs.chunk_id],
                       t_start=obs.t_start, t_end=obs.t_end, description=desc, entity_ids=ids,
                       salience=sal, modality=modality)
            self.store.upsert(ev, chunk_id=obs.chunk_id, label="insert")

    def _find_duplicate(self, desc, ids, modality, obs) -> Event | None:
        recent = sorted((r for r in self.store.records.values()
                         if isinstance(r, Event) and not r.compressed and r.modality == modality),
                        key=lambda r: (-r.t_end, r.id))
        q = self._vec(desc)
        for r in recent[: self.cfg.event_dedup_lookback]:
            if obs.t_start - r.t_end > 1.01 * self._chunk_len or r.t_end > obs.t_start + 1e-6:
                continue  # only the immediately preceding chunk, not the same chunk
            if set(r.entity_ids) != set(ids):
                continue
            # merge only if similar AND the new description adds no content word: the merged
            # event keeps the old wording, so anything new would otherwise be lost silently
            if adds_information(desc, r.description):
                continue
            if float(q @ self._vec(r.description)) >= self.cfg.event_dedup_tau:
                return r
        return None

    # ------------------------------------------------------------------ 7.4 relations
    def _insert_relations(self, obs: ChunkObservation, ctx: _ChunkContext) -> None:
        for s, p, o in obs.relations:
            sid, oid = ctx.name_to_id.get(s), ctx.name_to_id.get(o)
            if not sid or not oid:
                continue
            key = (sid, norm(p), oid)
            rid = self._rel_key.get(key)
            if rid and rid in self.store.records:
                r: Relation = self.store.get(rid)  # type: ignore[assignment]
                self.store.upsert(r.model_copy(update={"source_chunks": [*r.source_chunks, obs.chunk_id]}),
                                  chunk_id=obs.chunk_id, label="extend", reason="repeated relation")
                continue
            sname = self.store.get(sid).name  # type: ignore[union-attr]
            oname = self.store.get(oid).name  # type: ignore[union-attr]
            r = Relation(id=self.store.new_id("relation"), source_chunks=[obs.chunk_id], subj=sid,
                         subj_name=sname, pred=norm(p), obj=oid, obj_name=oname, t=obs.t_start)
            self.store.upsert(r, chunk_id=obs.chunk_id, label="insert")
            self._rel_key[key] = r.id

    # ------------------------------------------------------------------ 7.5 / 7.6 budget
    def _enforce_budget(self, chunk_id: int) -> None:
        # each step removes a compression candidate or a record, so the loop terminates;
        # an empty memory has 0 tokens and therefore always fits
        while self.store.tokens() > self.budget:
            if not (self.cfg.use_compression and self._compress_oldest(chunk_id)):
                self._delete_lowest(chunk_id)

    def _compress_oldest(self, chunk_id: int) -> bool:
        """Compress the oldest window at the lowest level that has one; False if nothing
        outside the protected recent chunks can be compressed any further.

        Level 0 merges original events within compression_window_s. Summaries (and single
        events kept verbatim) are level >= 1; level L merges windows of
        compression_window_s * 2**L, up to max_compression_level. Without this, summaries
        were final, and under a small budget C3 degenerated into pure eviction."""
        limit = chunk_id - self.cfg.protect_recent_chunks
        old = [r for r in self.store.records.values()
               if isinstance(r, Event) and max(r.source_chunks) <= limit]
        for level in range(self.cfg.max_compression_level + 1):
            cands = sorted((r for r in old if (not r.compressed if level == 0
                                               else r.compressed and r.level == level)),
                           key=lambda r: (r.t_start, r.id))
            if cands:
                return self._compress_window(cands, level, chunk_id)
        return False

    def _compress_window(self, cands: list[Event], level: int, chunk_id: int) -> bool:
        t0 = cands[0].t_start
        group = [r for r in cands if r.t_start < t0 + self.cfg.compression_window_s * 2 ** level]
        if len(group) == 1:
            r = group[0]
            # nothing to merge with at this level: move it up a level, text unchanged (it is
            # shown as a summary only if it is one), so the next window or level is considered
            self.store.upsert(r.model_copy(update={"compressed": True, "level": level + 1}),
                              chunk_id=chunk_id, label="compress",
                              reason=f"single event at level {level} (kept verbatim)")
            return True
        t1 = max(r.t_end for r in group)
        sources = sorted({c for r in group for c in r.source_chunks})
        known = {i for r in group for i in r.entity_ids}
        items = self._summarize(group, t0, t1)
        for r in group:
            self.store.delete(r.id, chunk_id=chunk_id, label="compress",
                              reason=f"level {level} window {format_timestamp(t0, decimals=0)}")
        for it in items:
            ids = [i for i in dict.fromkeys(it.entity_ids) if i in known]
            ev = Event(id=self.store.new_id("event"), source_chunks=sources, t_start=t0, t_end=t1,
                       description=it.description, entity_ids=ids, salience=it.salience,
                       compressed=True, summarized=True, level=level + 1)
            self.store.upsert(ev, chunk_id=chunk_id, label="compress",
                              reason=f"summary of {len(group)} events (level {level + 1})")
        self._count("compressions" if level == 0 else "recompressions")
        return True

    def _summarize(self, group: list[Event], t0: float, t1: float) -> list[_CompressedItem]:
        if self.llm is not None:
            lines = "\n".join(f"{r.line()} (salience {r.salience})" for r in group)
            prompt = COMPRESS.format(t_start=format_timestamp(t0, decimals=0),
                                     t_end=format_timestamp(t1, decimals=0), events=lines,
                                     max_items=max(1, min(3, len(group) // 2)))
            res = generate_validated(self.llm, prompt, _Compressed, max_new_tokens=512)
            if res.ok:
                return res.value.summaries
            self._count("compression_failures")
        top = sorted(group, key=lambda r: (-r.salience, r.t_start))[:2]
        return [_CompressedItem(description="; ".join(r.description for r in top),
                                entity_ids=[i for r in top for i in r.entity_ids],
                                salience=max(r.salience for r in group))]

    def priority(self, r: RecordT, refcount: dict[str, int], t_min: float) -> float:
        w = self.cfg.priority_weights
        span = max(1e-6, self.now - t_min)
        recency = (r.t_ref - t_min) / span
        base = w.get("recency", 1.0) * recency
        if isinstance(r, Event):
            return base + w.get("salience", 1.0) * r.salience + w.get("refs", 0.5) * len(r.entity_ids)
        if isinstance(r, Relation):
            return base + w.get("salience", 1.0) * 1 + w.get("refs", 0.5) * 2
        if isinstance(r, StateChange):
            return base + w.get("salience", 1.0) * 2 + w.get("state_change", 2.0)
        if isinstance(r, Entity):
            # an entity is worth keeping because other records refer to it; one that nothing
            # refers to any more only carries its attributes and ranks like a background event
            refs = refcount.get(r.id, 0)
            if refs == 0:
                return base + w.get("salience", 1.0) * 1
            return base + w.get("entity", 3.0) + w.get("refs", 0.5) * refs
        return base

    def _delete_lowest(self, chunk_id: int) -> None:
        recs = list(self.store.records.values())
        refcount: dict[str, int] = {}
        for r in recs:
            for e in references(r):
                refcount[e] = refcount.get(e, 0) + 1
        t_min = min(r.t_ref for r in recs)
        victim = min(recs, key=lambda r: (self.priority(r, refcount, t_min), r.t_ref, r.id))
        self.store.delete(victim.id, chunk_id=chunk_id, label="evict",
                          reason=f"lowest priority {self.priority(victim, refcount, t_min):.2f}")
        self._count(f"evicted_{victim.kind}")

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["config"] = self.cfg.model_dump(mode="json")
        d["open_pending_changes"] = len(self.pending)
        return d
