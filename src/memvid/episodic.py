"""Query-adaptive retrieval from a compact memory and an immutable episode archive.

The structured memory is used as a query-time index rather than the only copy of every
detail.  Candidate chunks from C3 are fused with dense retrieval over frozen observations;
only a token-bounded set of original episodes is passed to the answer model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from memvid.config import ProjectConfig
from memvid.memory.answer import AnswerRecord, answer_question
from memvid.memory.persist import load_memory
from memvid.memory.records import Entity, Note, RecordT
from memvid.memory.retrieval import Retrieved, retrieve
from memvid.memory.store import MemoryStore
from memvid.models.base import GenerativeModel
from memvid.models.embed import Embedder, TokenCounter
from memvid.models.jsonutil import generate_validated
from memvid.perception.run import load_observations
from memvid.perception.schema import ChunkObservation
from memvid.pipeline import Pipeline, video_duration
from memvid.questions.schema import Question
from memvid.schemas import Chunk, FrameRecord

Method = Literal["archive", "c3", "hybrid"]
LOG = logging.getLogger(__name__)


@dataclass
class EpisodeSelection:
    method: Method
    notes: list[Note]
    ranked_chunk_ids: list[int]
    scores: dict[int, float]
    tokens: int


class VisualEvidence(BaseModel):
    observations: list[str] = Field(default_factory=list)
    sufficient: bool = False


class EvidenceCritique(BaseModel):
    sufficient: bool
    missing: str = ""
    targeted_query: str = ""


class AnswerChoice(BaseModel):
    choice: Literal["initial", "revised"]
    reason: str = ""


VISUAL_EVIDENCE_PROMPT = """### TASK: visual_evidence
Inspect the supplied frames from the candidate video chunks. Extract only concrete visual
evidence relevant to the question. Do not answer from general knowledge and do not invent
details that are not visible. The images are ordered exactly as described in FRAME GROUPS.

### QUESTION
{question}

### FRAME GROUPS
{groups}

### OUTPUT
Reply with one JSON object:
{{"observations": ["timestamped concrete observation"], "sufficient": true}}
"""


VISUAL_FALLBACK_PROMPT = """### TASK: visual_evidence_fallback
Inspect the frames in chronological order for the question below. Return one short factual
observation per line. Begin every line with '-'. Mention only visible evidence and do not answer
the question. If nothing relevant is visible, return '- no relevant visual evidence visible'.

### QUESTION
{question}

### FRAME GROUPS
{groups}
"""


CRITIC_PROMPT = """### TASK: evidence_gate
Decide whether the evidence contains every fact required to answer the question. Do not judge
writing style and do not demand more evidence when the draft is already directly supported.
For a change question require before and after states. For a cross-segment question require the
two relevant episodes. If something is missing, write a short search query for that exact fact.

### QUESTION
{question}

### EVIDENCE
{evidence}

### DRAFT ANSWER
{answer}

### OUTPUT
Reply with one JSON object:
{{"sufficient": false, "missing": "short description", "targeted_query": "focused search query"}}
"""


ANSWER_CHOICE_PROMPT = """### TASK: answer_choice
Choose which candidate is better supported by the supplied video-memory evidence. Do not prefer
the revised answer merely because it is newer. Select the initial answer when revision adds an
unsupported detail or reverses a role, relation, state, order, number, or attribute.

### QUESTION
{question}

### EVIDENCE
{evidence}

### INITIAL
{initial}

### REVISED
{revised}

### OUTPUT
Reply with one JSON object: {{"choice": "initial", "reason": "short reason"}}
"""


TYPE_GUIDANCE = {
    "speech": "Prefer exact transcript wording and speaker attribution.",
    "attribute_recall": "Identify the target entity first, then extract only its requested attribute.",
    "past_event": "Order the relevant events chronologically before answering.",
    "object_change": "Find and compare explicit before and after states of the same object.",
    "cross_segment": "Use evidence from each distinct episode named or implied by the question.",
}


def query_variants(question: Question) -> list[str]:
    """Deterministic, type-aware queries; no test labels or reference answers are used."""
    q = " ".join(question.question.split())
    variants = [q]
    if question.modality in ("speech", "both"):
        variants.append(f"spoken statement dialogue transcript speaker: {q}")
    if question.modality in ("visual", "both"):
        variants.append(f"visible object action appearance location: {q}")
    if question.type == "object_change":
        variants.extend((f"before state {q}", f"after state change {q}"))
    elif question.type == "cross_segment":
        variants.extend((f"earlier episode {q}", f"later episode relation {q}"))
    elif question.type == "past_event":
        variants.append(f"event sequence immediately before after {q}")
    elif question.type == "attribute_recall":
        variants.append(f"entity attribute color size clothing location {q}")
    return list(dict.fromkeys(variants))


def episode_text(observation: ChunkObservation, chunk: Chunk) -> str:
    """Lossless-enough archive text: frozen perception plus the aligned raw transcript."""
    text = observation.to_text()
    transcript = " ".join(chunk.transcript_text.split())
    if transcript:
        text += f" Transcript: {transcript}"
    return text


def fuse_chunk_rankings(
    archive: list[tuple[int, float]],
    c3: list[tuple[int, float]],
    method: Method,
    *,
    rrf_constant: int = 60,
    archive_weight: float = 2.0,
    c3_weight: float = 1.0,
) -> list[tuple[int, float]]:
    """Deterministic reciprocal-rank fusion; raw scores need not share a scale.

    The episode archive is the precise locator, while broad C3 provenance is a secondary
    expansion signal.  The weights are fixed before test evaluation and archive-only remains
    a dev competitor, so a noisy memory index cannot be reported as an improvement by fiat.
    """
    fused: dict[int, float] = {}
    if method in ("archive", "hybrid"):
        for rank, (chunk_id, _) in enumerate(archive, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + archive_weight / (rrf_constant + rank)
    if method in ("c3", "hybrid"):
        for rank, (chunk_id, _) in enumerate(c3, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + c3_weight / (rrf_constant + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class EpisodicRetriever:
    def __init__(
        self,
        cfg: ProjectConfig,
        c3_memory: Path,
        embedder: Embedder,
        counter: TokenCounter,
        *,
        answer_budget_tokens: int | None = None,
        max_chunks: int = 8,
        archive_candidates: int = 24,
        c3_candidates: int = 24,
        neighbour_radius: int = 1,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.counter = counter
        self.answer_budget_tokens = answer_budget_tokens or cfg.retrieval.budget_tokens
        self.max_chunks = max_chunks
        self.archive_candidates = archive_candidates
        self.c3_candidates = c3_candidates
        self.neighbour_radius = neighbour_radius
        self.duration = video_duration(cfg)

        pipeline = Pipeline(cfg)
        chunks = pipeline.chunks()
        observations = load_observations(cfg, require_frozen=True)
        if [c.chunk_id for c in chunks] != [o.chunk_id for o in observations]:
            raise ValueError("chunk and observation ids are not aligned")
        self.chunks = {c.chunk_id: c for c in chunks}
        self.observations = {o.chunk_id: o for o in observations}
        self.bounds = {c.chunk_id: (c.t_start, c.t_end) for c in chunks}

        notes = {
            f"A{c.chunk_id:04d}": Note(
                id=f"A{c.chunk_id:04d}", source_chunks=[c.chunk_id],
                t_start=c.t_start, t_end=c.t_end,
                body=episode_text(self.observations[c.chunk_id], c),
            )
            for c in chunks
        }
        LOG.info("building episodic archive index: %d chunks", len(notes))
        self.archive = MemoryStore.from_records(notes, embedder, counter)
        memory_path = c3_memory / "memory.sqlite" if c3_memory.is_dir() else c3_memory
        saved = load_memory(memory_path)
        LOG.info("building C3 retrieval index: %d records from %s", len(saved.records), memory_path)
        self.c3 = MemoryStore.from_records(saved.records, embedder, counter)
        LOG.info("retrieval indices ready: archive_tokens=%d c3_tokens=%d",
                 self.archive.tokens(), self.c3.tokens())

    @staticmethod
    def _chunk_id(note_id: str) -> int:
        return int(note_id[1:])

    def _archive_ranking(self, question: str) -> list[tuple[int, float]]:
        scores = self.archive.index.scores(self.embedder.encode_one(question))
        ranked = sorted(self.archive.records.values(), key=lambda r: (-scores[r.id], r.id))
        return [(self._chunk_id(r.id), scores[r.id]) for r in ranked[: self.archive_candidates]]

    def _c3_ranking(self, question: str) -> list[tuple[int, float]]:
        ret = retrieve(self.c3, question, self.cfg.retrieval, self.duration)
        scores: dict[int, float] = {}
        ranked_records = [self.c3.records[rid] for rid in ret.ranked_ids if rid in self.c3.records]
        for rank, record in enumerate(ranked_records[: self.c3_candidates], start=1):
            if isinstance(record, Entity):
                continue
            # Broad compressed provenance remains useful for localization, but each source
            # receives less weight so a large union cannot dominate exact episode matches.
            weight = 1.0 / (60 + rank) / math.sqrt(max(1, len(record.source_chunks)))
            for chunk_id in record.source_chunks:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def rank(self, question: str, method: Method) -> list[tuple[int, float]]:
        archive = self._archive_ranking(question)
        c3 = self._c3_ranking(question)
        ranked = fuse_chunk_rankings(archive, c3, method)
        fused = dict(ranked)

        # Neighbours preserve local temporal continuity but never outrank their anchor.
        expanded = dict(fused)
        for chunk_id, score in ranked[: self.max_chunks]:
            for delta in range(1, self.neighbour_radius + 1):
                for neighbour in (chunk_id - delta, chunk_id + delta):
                    if neighbour in self.chunks:
                        expanded[neighbour] = max(expanded.get(neighbour, 0.0), score * 0.45 / delta)
        return sorted(expanded.items(), key=lambda item: (-item[1], item[0]))

    def rank_question(self, question: Question, method: Method) -> list[tuple[int, float]]:
        """Fuse several modality/type-specific searches without mixing raw score scales."""
        fused: dict[int, float] = {}
        for query in query_variants(question):
            for rank, (chunk_id, _) in enumerate(self.rank(query, method), start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def _selection_from_ranking(
        self,
        ranking: list[tuple[int, float]],
        method: Method,
        *,
        skip: set[int] | None = None,
        diversify: bool = False,
    ) -> EpisodeSelection:
        skip = skip or set()
        if diversify:
            # Put two separated high-ranking anchors first, then preserve the original
            # ranking. This covers distinct moments without promoting weak distant hits.
            anchors: list[tuple[int, float]] = []
            for item in ranking:
                if all(abs(item[0] - chosen[0]) > self.neighbour_radius for chosen in anchors):
                    anchors.append(item)
                if len(anchors) == 2:
                    break
            anchor_ids = {item[0] for item in anchors}
            ranking = anchors + [item for item in ranking if item[0] not in anchor_ids]
        selected: list[Note] = []
        selected_ranked: list[int] = []
        used = 0
        scores = dict(ranking)
        for chunk_id, _ in ranking:
            if chunk_id in skip:
                continue
            note = self.archive.records[f"A{chunk_id:04d}"]
            tokens = self.archive.tokens_of(note.id)
            if used + tokens > self.answer_budget_tokens:
                continue
            selected.append(note)  # type: ignore[arg-type]
            selected_ranked.append(chunk_id)
            used += tokens
            if len(selected) >= self.max_chunks:
                break
        if not selected:
            chunk_id = next(chunk_id for chunk_id, _ in ranking if chunk_id not in skip)
            note = self.archive.records[f"A{chunk_id:04d}"]
            selected = [note]  # type: ignore[list-item]
            selected_ranked = [chunk_id]
            used = self.archive.tokens_of(note.id)
        selected.sort(key=lambda r: (r.t_start, r.id))
        return EpisodeSelection(method, selected, selected_ranked, scores, used)

    def select(self, question: str, method: Method, *, skip: set[int] | None = None) -> EpisodeSelection:
        return self._selection_from_ranking(self.rank(question, method), method, skip=skip)

    def select_question(self, question: Question, method: Method) -> EpisodeSelection:
        diversify = question.type in ("cross_segment", "object_change")
        return self._selection_from_ranking(
            self.rank_question(question, method), method, diversify=diversify,
        )

    def expand(self, question: str, current: EpisodeSelection, n_chunks: int = 2) -> EpisodeSelection:
        extra = self.select(question, current.method, skip=set(current.ranked_chunk_ids))
        notes = list(current.notes)
        used = current.tokens
        added = 0
        for note in extra.notes:
            tokens = self.archive.tokens_of(note.id)
            # Corrective retrieval gets a separate, bounded allowance rather than silently
            # changing the primary answer budget.
            if used + tokens > 2 * self.answer_budget_tokens:
                continue
            notes.append(note)
            used += tokens
            added += 1
            if added >= n_chunks:
                break
        notes.sort(key=lambda r: (r.t_start, r.id))
        ids = list(dict.fromkeys([
            *current.ranked_chunk_ids,
            *(note.source_chunks[0] for note in extra.notes[:added]),
        ]))
        return EpisodeSelection(current.method, notes, ids, current.scores, used)

    def expand_targeted(
        self,
        query: str,
        current: EpisodeSelection,
        *,
        n_chunks: int = 2,
    ) -> EpisodeSelection:
        return self.expand(query, current, n_chunks=n_chunks)

    def as_retrieved(self, selection: EpisodeSelection) -> Retrieved:
        ranked_ids = [f"A{chunk_id:04d}" for chunk_id in selection.ranked_chunk_ids]
        return Retrieved(
            records=list(selection.notes), ranked_ids=ranked_ids,
            scores={f"A{k:04d}": v for k, v in selection.scores.items()},
            tokens=selection.tokens, matched_entities=[], cues=[selection.method, "episodic-archive"],
        )


def answer_from_selection(
    question: Question,
    selection: EpisodeSelection,
    retriever: EpisodicRetriever,
    llm: GenerativeModel,
    *,
    structured: bool = False,
) -> AnswerRecord:
    if not structured:
        return answer_question(question, retriever.as_retrieved(selection), llm, retriever.bounds)
    guidance = TYPE_GUIDANCE.get(question.type, "Use the most specific evidence available.")
    structured_question = (
        f"{question.question}\n\nReasoning protocol: {guidance} "
        "First compare the timestamped records internally; return only the required short answer."
    )
    return answer_question(
        question, retriever.as_retrieved(selection), llm, retriever.bounds,
        question_text=structured_question,
    )


def inspect_visual_evidence(
    cfg: ProjectConfig,
    question: Question,
    selection: EpisodeSelection,
    vlm: GenerativeModel,
    *,
    max_visual_chunks: int = 2,
    max_frames_per_chunk: int = 4,
) -> tuple[list[str], list[int]]:
    if question.modality == "speech":
        return [], []
    pipeline = Pipeline(cfg)
    frame_index: dict[int, list[FrameRecord]] = {}
    for frame in pipeline.frames():
        frame_index.setdefault(frame.chunk_id, []).append(frame)
    chosen = selection.ranked_chunk_ids[:max_visual_chunks]
    images: list[Image.Image] = []
    groups: list[str] = []
    cursor = 1
    for chunk_id in chosen:
        records = sorted(frame_index.get(chunk_id, []), key=lambda r: r.frame_index)
        if len(records) > max_frames_per_chunk:
            indices = [round(i * (len(records) - 1) / (max_frames_per_chunk - 1))
                       for i in range(max_frames_per_chunk)]
            records = [records[i] for i in dict.fromkeys(indices)]
        if not records:
            continue
        start = cursor
        for frame in records:
            image = Image.open(cfg.derived_dir / frame.path)
            image.load()
            images.append(image.convert("RGB"))
            cursor += 1
        lo, hi = selection.notes[0].t_start, selection.notes[0].t_end
        matching = [n for n in selection.notes if n.source_chunks == [chunk_id]]
        if matching:
            lo, hi = matching[0].t_start, matching[0].t_end
        groups.append(f"images {start}-{cursor - 1}: chunk {chunk_id}, {lo:.1f}-{hi:.1f}s")
    if not images:
        return [], []
    result = generate_validated(
        vlm,
        VISUAL_EVIDENCE_PROMPT.format(question=question.question, groups="\n".join(groups)),
        VisualEvidence,
        images=images,
        max_new_tokens=256,
        repair_attempts=1,
    )
    if result.ok and result.value.observations:
        return result.value.observations, chosen
    LOG.warning("visual JSON failed; using plain-text fallback: chunks=%s error=%s",
                chosen, result.error or "empty observations")
    raw = vlm.generate(
        VISUAL_FALLBACK_PROMPT.format(question=question.question, groups="\n".join(groups)),
        images=images,
        max_new_tokens=192,
    )
    observations = [
        line.strip().lstrip("-*• ").strip()
        for line in raw.splitlines()
        if line.strip().startswith(("-", "*", "•"))
    ]
    observations = [line for line in observations if line]
    if not observations and raw.strip():
        observations = [" ".join(raw.split())[:1000]]
    LOG.info("visual fallback produced %d observations for chunks=%s",
             len(observations), chosen)
    return observations, chosen


def attach_visual_evidence(
    selection: EpisodeSelection,
    observations: list[str],
    visual_chunks: list[int],
    counter: TokenCounter,
) -> EpisodeSelection:
    if not observations or not visual_chunks:
        return selection
    notes = list(selection.notes)
    matching = [note for note in notes if note.source_chunks[0] in visual_chunks]
    t_start = min(note.t_start for note in matching)
    t_end = max(note.t_end for note in matching)
    visual_note = Note(
        id="Z1", source_chunks=visual_chunks, t_start=t_start, t_end=t_end,
        body="Targeted visual inspection: " + " ".join(observations),
    )
    notes.append(visual_note)
    notes.sort(key=lambda r: (r.t_start, r.id))
    added_tokens = counter.count(visual_note.line())
    return EpisodeSelection(selection.method, notes, selection.ranked_chunk_ids,
                            selection.scores, selection.tokens + added_tokens)


def critique_evidence(
    question: Question,
    selection: EpisodeSelection,
    answer: AnswerRecord,
    llm: GenerativeModel,
) -> EvidenceCritique:
    evidence = "\n".join(note.line() for note in selection.notes)
    result = generate_validated(
        llm,
        CRITIC_PROMPT.format(question=question.question, evidence=evidence, answer=answer.answer),
        EvidenceCritique,
        max_new_tokens=96,
        repair_attempts=1,
    )
    return result.value if result.ok else EvidenceCritique(
        sufficient=True, missing="critic failed", targeted_query="",
    )


def choose_supported_answer(
    question: Question,
    selection: EpisodeSelection,
    initial: AnswerRecord,
    revised: AnswerRecord,
    llm: GenerativeModel,
) -> tuple[AnswerRecord, AnswerChoice]:
    """Keep corrective retrieval from silently replacing a stronger initial answer."""
    evidence = "\n".join(note.line() for note in selection.notes)
    result = generate_validated(
        llm,
        ANSWER_CHOICE_PROMPT.format(
            question=question.question,
            evidence=evidence,
            initial=initial.answer,
            revised=revised.answer,
        ),
        AnswerChoice,
        max_new_tokens=96,
        repair_attempts=1,
    )
    choice = result.value if result.ok else AnswerChoice(
        choice="initial", reason="answer selector failed",
    )
    return (revised if choice.choice == "revised" else initial), choice
