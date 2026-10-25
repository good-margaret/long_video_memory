"""Evidence union over structured C4 memory and immutable episodic observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memvid.config import ProjectConfig
from memvid.episodic import (
    EpisodicRetriever,
    EpisodeSelection,
    TYPE_GUIDANCE,
    attach_visual_evidence,
)
from memvid.memory.answer import AnswerRecord, answer_question
from memvid.memory.persist import load_memory
from memvid.memory.records import RecordT
from memvid.memory.retrieval import Retrieved, retrieve
from memvid.memory.store import MemoryStore
from memvid.models.base import GenerativeModel
from memvid.models.embed import Embedder, TokenCounter
from memvid.pipeline import video_duration
from memvid.questions.schema import Question


@dataclass
class UnionSelection:
    retrieved: Retrieved
    episode_selection: EpisodeSelection
    c4_ids: list[str]
    archive_ids: list[str]


class EvidenceUnionRetriever:
    """Retrieve complementary compressed and source-level evidence under one budget."""

    def __init__(
        self,
        cfg: ProjectConfig,
        c3_memory: Path,
        c4_memory: Path,
        embedder: Embedder,
        counter: TokenCounter,
        *,
        union_budget_tokens: int = 3500,
        max_chunks: int = 8,
    ):
        self.cfg = cfg
        self.counter = counter
        self.union_budget_tokens = union_budget_tokens
        self.duration = video_duration(cfg)
        self.episodes = EpisodicRetriever(
            cfg,
            c3_memory,
            embedder,
            counter,
            answer_budget_tokens=min(1700, union_budget_tokens // 2),
            max_chunks=max_chunks,
        )
        path = c4_memory / "memory.sqlite" if c4_memory.is_dir() else c4_memory
        saved = load_memory(path)
        self.c4 = MemoryStore.from_records(saved.records, embedder, counter)

    @staticmethod
    def _key(record: RecordT) -> tuple:
        body = getattr(record, "body", None) or getattr(record, "description", None) or record.line()
        return tuple(record.source_chunks), " ".join(str(body).lower().split())

    def select(self, question: Question) -> UnionSelection:
        c4 = retrieve(self.c4, question.question, self.cfg.retrieval, self.duration)
        if question.type in ("cross_segment", "object_change"):
            episodes = self.episodes.select_question(question, "hybrid")
        else:
            episodes = self.episodes.select(question.question, "hybrid")

        # Round-robin prevents either compressed C4 or detailed archive evidence from
        # monopolising the union budget. The final records are chronological for reasoning.
        streams = [list(c4.records), list(episodes.notes)]
        chosen: list[RecordT] = []
        seen: set[tuple] = set()
        tokens = 0
        while any(streams):
            progressed = False
            for stream in streams:
                if not stream:
                    continue
                record = stream.pop(0)
                key = self._key(record)
                if key in seen:
                    continue
                cost = counter_cost(self.counter, record)
                if tokens + cost > self.union_budget_tokens:
                    continue
                chosen.append(record)
                seen.add(key)
                tokens += cost
                progressed = True
            if not progressed and not any(streams):
                break

        chosen.sort(key=lambda record: (
            min(record.source_chunks) if record.source_chunks else 10**9,
            record.id,
        ))
        ranked_ids = list(dict.fromkeys([*c4.ranked_ids, *[
            f"A{chunk_id:04d}" for chunk_id in episodes.ranked_chunk_ids
        ]]))
        scores = dict(c4.scores)
        scores.update({f"A{chunk_id:04d}": score for chunk_id, score in episodes.scores.items()})
        retrieved = Retrieved(
            records=chosen,
            ranked_ids=ranked_ids,
            scores=scores,
            tokens=tokens,
            matched_entities=c4.matched_entities,
            cues=[*c4.cues, "evidence-union", question.type],
        )
        return UnionSelection(
            retrieved=retrieved,
            episode_selection=episodes,
            c4_ids=[record.id for record in c4.records],
            archive_ids=[record.id for record in episodes.notes],
        )


def counter_cost(counter: TokenCounter, record: RecordT) -> int:
    return counter.count(record.line())


def answer_from_union(
    question: Question,
    selection: UnionSelection,
    llm: GenerativeModel,
    bounds: dict[int, tuple[float, float]],
) -> AnswerRecord:
    guidance = TYPE_GUIDANCE.get(question.type, "Use the most specific evidence available.")
    question_text = (
        f"{question.question}\n\nEvidence protocol: {guidance} "
        "Reconcile compressed records with timestamped source episodes. Prefer a concrete "
        "source episode when summaries conflict. Check every part of the question, then return "
        "only the required short answer with supporting record citations."
    )
    return answer_question(
        question,
        selection.retrieved,
        llm,
        bounds,
        question_text=question_text,
    )


def add_visual_to_union(
    selection: UnionSelection,
    observations: list[str],
    visual_chunks: list[int],
    counter: TokenCounter,
) -> UnionSelection:
    enriched = attach_visual_evidence(
        selection.episode_selection, observations, visual_chunks, counter,
    )
    added = [record for record in enriched.notes if record.id == "Z1"]
    if not added:
        return selection
    records = [*selection.retrieved.records, *added]
    tokens = selection.retrieved.tokens + sum(counter_cost(counter, record) for record in added)
    retrieved = Retrieved(
        records=records,
        ranked_ids=[*selection.retrieved.ranked_ids, "Z1"],
        scores={**selection.retrieved.scores, "Z1": 1.0},
        tokens=tokens,
        matched_entities=selection.retrieved.matched_entities,
        cues=[*selection.retrieved.cues, "targeted-visual"],
    )
    return UnionSelection(
        retrieved=retrieved,
        episode_selection=enriched,
        c4_ids=selection.c4_ids,
        archive_ids=selection.archive_ids,
    )
