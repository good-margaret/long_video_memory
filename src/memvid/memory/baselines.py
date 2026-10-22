"""Baselines: C1 rolling summary, C2 flat log (FIFO), C4 unbounded flat log."""

from __future__ import annotations

from pydantic import BaseModel

from memvid.memory.base import MemorySystem
from memvid.memory.records import Note
from memvid.memory.store import MemoryStore
from memvid.models.base import GenerativeModel
from memvid.models.embed import TokenCounter, truncate_to_tokens
from memvid.models.jsonutil import generate_validated
from memvid.perception.schema import ChunkObservation
from memvid.prompts import SUMMARIZE


def fit_note(note: Note, budget: int, counter: TokenCounter) -> Note | None:
    """Longest word prefix of the note body whose full line fits the budget (exact for any
    tokenizer: the whole line is re-counted, so boundary merges cannot overshoot)."""
    if counter.count(note.line()) <= budget:
        return note
    overhead = counter.count(note.model_copy(update={"body": ""}).line())
    limit = budget - overhead
    while limit > 0:
        body = truncate_to_tokens(note.body, limit, counter)
        cand = note.model_copy(update={"body": body})
        if body and counter.count(cand.line()) <= budget:
            return cand
        limit -= 1
    return None


class FlatLog(MemorySystem):
    condition = "C2"

    def _process(self, obs: ChunkObservation) -> None:
        s = self.store
        note = Note(id=s.new_id("note"), source_chunks=[obs.chunk_id], t_start=obs.t_start,
                    t_end=obs.t_end, body=obs.to_text())
        s.upsert(note, chunk_id=obs.chunk_id, label="insert")
        if self.budget is None:
            return
        while s.tokens() > self.budget:
            notes = s.of_kind("note")
            if len(notes) > 1:
                s.delete(notes[0].id, chunk_id=obs.chunk_id, label="evict", reason="fifo")
                self._count("evicted")
                continue
            only = notes[0]
            fitted = fit_note(only, self.budget, s.counter)
            if fitted is None:
                s.delete(only.id, chunk_id=obs.chunk_id, label="evict", reason="larger than budget")
            else:
                s.upsert(fitted, chunk_id=obs.chunk_id, label="truncate",
                         reason="single record larger than budget")
            self._count("truncated")
            break


class _Summary(BaseModel):
    summary: str


class RollingSummary(MemorySystem):
    condition = "C1"

    def __init__(self, store: MemoryStore, budget: int, llm: GenerativeModel):
        super().__init__(store, budget)
        self.llm = llm
        self.note_id = store.new_id("note")
        self.text = ""
        self.seen: list[int] = []

    def _process(self, obs: ChunkObservation) -> None:
        s = self.store
        self.seen.append(obs.chunk_id)
        overhead = s.counter.count(
            Note(id=self.note_id, source_chunks=[0], t_start=0, t_end=obs.t_end, body="").line())
        max_tokens = max(1, self.budget - overhead)
        max_words = max(10, int(max_tokens * 0.7))
        prompt = SUMMARIZE.format(summary=self.text or "(empty)", observation=obs.to_text(),
                                  max_words=max_words)
        res = generate_validated(self.llm, prompt, _Summary, max_new_tokens=min(4096, 2 * max_tokens + 64))
        if res.ok and res.value.summary.strip():
            new = res.value.summary.strip()
        else:
            self._count("summary_failures")
            new = (self.text + " " + obs.to_text()).strip()
            words = new.split()
            new = " ".join(words[-max_words:])
        note = Note(id=self.note_id, source_chunks=self.seen, t_start=0.0, t_end=obs.t_end,
                    body=new or "(empty)")
        fitted = fit_note(note, self.budget, s.counter)
        if fitted is None:
            raise ValueError(f"C1 budget {self.budget} is too small for a single summary line")
        if fitted.body != note.body:
            self._count("hard_truncations")
        self.text = fitted.body
        s.upsert(fitted, chunk_id=obs.chunk_id, label="rewrite")
