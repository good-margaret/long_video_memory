"""Question records (human-written, data/annotations/<video_id>/questions.yaml)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memvid.timeutils import parse_timestamp

QuestionType = Literal["past_event", "object_change", "cross_segment", "attribute_recall", "speech"]
AnswerType = Literal["short_text", "number", "yes_no", "multiple_choice"]
QUESTION_TYPES: tuple[str, ...] = QuestionType.__args__  # type: ignore[attr-defined]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,32}$")
    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)
    accepted_answers: list[str] = Field(default_factory=list)
    answer_type: AnswerType = "short_text"
    options: list[str] = Field(default_factory=list, description="multiple_choice only")
    type: QuestionType
    gt_intervals: list[tuple[float, float]] = Field(min_length=1)
    evidence_time: float | None = Field(
        None, description="Latest moment whose information is needed; default: the latest "
        "interval end. Evidence age = video duration - evidence_time.")
    modality: Literal["visual", "speech", "both"]
    notes: str = ""

    @field_validator("gt_intervals", mode="before")
    @classmethod
    def _intervals(cls, v):
        out = []
        for item in v:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("each interval must be [start, end]")
            a, b = parse_timestamp(item[0]), parse_timestamp(item[1])
            if b <= a:
                raise ValueError(f"interval end must exceed start: {item}")
            out.append((a, b))
        return sorted(out)

    @field_validator("evidence_time", mode="before")
    @classmethod
    def _evt(cls, v):
        return None if v is None else parse_timestamp(v)

    @model_validator(mode="after")
    def _consistency(self) -> Question:
        if self.answer_type == "multiple_choice":
            if len(self.options) < 2:
                raise ValueError(f"{self.id}: multiple_choice needs >= 2 options")
            a = self.answer.strip().upper()
            if len(a) != 1 or not (0 <= ord(a) - 65 < len(self.options)):
                raise ValueError(f"{self.id}: answer must be an option letter A..")
        elif self.options:
            raise ValueError(f"{self.id}: options only allowed for multiple_choice")
        if self.answer_type == "yes_no" and self.answer.strip().lower() not in ("yes", "no"):
            raise ValueError(f"{self.id}: yes_no answer must be yes or no")
        if self.answer_type == "number":
            from memvid.evaluation.scoring import parse_number

            if parse_number(self.answer) is None:
                raise ValueError(f"{self.id}: number answer is not a number")
        if self.evidence_time is None:
            self.evidence_time = max(b for _, b in self.gt_intervals)
        return self

    def evidence_age(self, duration_s: float) -> float:
        return max(0.0, duration_s - float(self.evidence_time or 0.0))


class QuestionFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    annotator: str
    questions: list[Question]

    @model_validator(mode="after")
    def _unique(self) -> QuestionFile:
        ids = [q.id for q in self.questions]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate question ids: {dupes}")
        return self
