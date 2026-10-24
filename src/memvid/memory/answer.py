"""Answering from retrieved records, with citations mapped back to video time."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from memvid.memory.records import RecordT
from memvid.memory.retrieval import Retrieved
from memvid.models.base import GenerativeModel
from memvid.models.jsonutil import generate_validated
from memvid.prompts import ANSWER, ANSWER_FORMATS
from memvid.questions.schema import Question
from memvid.timeutils import merge_intervals


class _ModelAnswer(BaseModel):
    answer: str = ""
    cited_record_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"

    @field_validator("answer", mode="before")
    @classmethod
    def _ans(cls, v):
        return "" if v is None else str(v).strip()

    @field_validator("cited_record_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        if isinstance(v, str):
            v = [v]
        return [str(x).strip().strip("[]") for x in (v or [])]

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        v = str(v or "low").lower().strip()
        return v if v in ("low", "medium", "high") else "low"


class AnswerRecord(BaseModel):
    qid: str
    answer: str
    cited_ids: list[str]
    invalid_citations: list[str] = Field(default_factory=list)
    confidence: str
    retrieved_ids: list[str]
    retrieval_ranked_ids: list[str] = Field(default_factory=list)
    retrieved_tokens: int
    cues: list[str] = Field(default_factory=list)
    matched_entities: list[str] = Field(default_factory=list)
    cited_intervals: list[tuple[float, float]] = Field(default_factory=list)
    retrieved_intervals: list[tuple[float, float]] = Field(default_factory=list)
    ok: bool = True
    error: str | None = None


def support(records: list[RecordT], bounds: dict[int, tuple[float, float]]) -> list[tuple[float, float]]:
    """I(C): union of the chunk intervals in the provenance of the records."""
    return merge_intervals([bounds[c] for r in records for c in r.source_chunks if c in bounds])


def build_answer_prompt(q: Question, memory_text: str, question_text: str | None = None) -> str:
    options = ""
    if q.answer_type == "multiple_choice":
        options = "\n### OPTIONS\n" + "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(q.options)) + "\n"
    return ANSWER.format(format_instruction=ANSWER_FORMATS[q.answer_type],
                         memory=memory_text or "(empty)", question=question_text or q.question,
                         options_block=options)


def answer_question(q: Question, ret: Retrieved, llm: GenerativeModel,
                    bounds: dict[int, tuple[float, float]]) -> AnswerRecord:
    memory_text = "\n".join(r.line() for r in ret.records)
    res = generate_validated(llm, build_answer_prompt(q, memory_text), _ModelAnswer,
                             max_new_tokens=128)
    by_id = {r.id: r for r in ret.records}
    if res.ok:
        cited = [i for i in dict.fromkeys(res.value.cited_record_ids) if i in by_id]
        invalid = [i for i in res.value.cited_record_ids if i not in by_id]
        ans, conf = res.value.answer, res.value.confidence
        if ans.lower() in ("low", "medium", "high") and q.answer_type != "short_text":
            ans = ""  # the confidence word copied into the answer field is not an answer
    else:
        cited, invalid, ans, conf = [], [], "", "low"
    return AnswerRecord(
        qid=q.id, answer=ans, cited_ids=cited, invalid_citations=invalid, confidence=conf,
        retrieved_ids=[r.id for r in ret.records], retrieval_ranked_ids=ret.ranked_ids,
        retrieved_tokens=ret.tokens, cues=ret.cues,
        matched_entities=ret.matched_entities,
        cited_intervals=support([by_id[i] for i in cited], bounds),
        retrieved_intervals=support(ret.records, bounds),
        ok=res.ok, error=res.error,
    )


def adhoc_question(text: str) -> Question:
    """A free question typed in the demo (no reference answer)."""
    return Question(id="adhoc", question=text if len(text) >= 5 else text + " ?????", answer="-",
                    type="past_event", gt_intervals=[(0.0, 1.0)], modality="both")
