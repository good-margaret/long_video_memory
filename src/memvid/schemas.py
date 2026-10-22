"""Data records produced by Stage 1. These are the contracts between steps:
later stages (perception, memory) read these files and nothing else."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoInfo(_Record):
    path: str
    size_bytes: int
    sha256: str | None = None
    duration_s: float = Field(gt=0)
    width: int
    height: int
    fps: float = Field(gt=0)
    video_codec: str
    n_frames: int | None = None
    has_audio: bool
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None


class TranscriptSegment(_Record):
    segment_id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> TranscriptSegment:
        if self.end < self.start:
            raise ValueError(f"segment {self.segment_id}: end < start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class Chunk(_Record):
    chunk_id: int = Field(ge=0)
    t_start: float = Field(ge=0)
    t_end: float
    transcript_text: str = ""
    segment_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> Chunk:
        if self.t_end <= self.t_start:
            raise ValueError(f"chunk {self.chunk_id}: t_end must exceed t_start")
        return self

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class FrameRecord(_Record):
    chunk_id: int = Field(ge=0)
    frame_index: int = Field(ge=0, description="Index of the frame within its chunk.")
    t_target: float = Field(ge=0, description="Requested time (s).")
    t_actual: float | None = Field(None, description="Presentation time of the decoded frame.")
    path: str = Field(description="Path relative to the video's derived directory.")
    width: int
    height: int
