"""Speech recognition with faster-whisper (optional dependency: pip install -e '.[asr]').

Design choices (see report, Section "Preprocessing"):
* vad_filter=True: Silero VAD removes non-speech regions before decoding, which
  suppresses Whisper's well-known hallucinations on silence and music.
* condition_on_previous_text=False: prevents a decoding error from propagating
  through the previous-text prompt into repetition loops on long recordings.
* temperature_fallback=True keeps the library default (re-decoding at higher
  temperature when a window looks degenerate). This is stochastic; the
  transcript is therefore frozen after verification instead of being
  regenerated. Set it to False for a fully deterministic run.
* Segment-level timestamps only; word timestamps are not needed because chunk
  assignment works at segment granularity.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from memvid.config import ASRConfig
from memvid.errors import MissingDependencyError
from memvid.schemas import TranscriptSegment
from memvid.timeutils import merge_intervals

log = logging.getLogger(__name__)


class ASRInfo(BaseModel):
    model_size: str
    device: str
    compute_type: str
    language: str | None
    language_probability: float | None
    audio_duration_s: float | None
    duration_after_vad_s: float | None
    wall_time_s: float
    real_time_factor: float | None  # wall time / audio duration (lower is faster)
    settings: dict[str, Any]


def _resolve_device(cfg: ASRConfig) -> tuple[str, str]:
    device = cfg.device
    if device == "auto":
        try:
            import ctranslate2  # type: ignore[import-not-found]

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = cfg.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def transcribe(audio_path: str | Path, cfg: ASRConfig) -> tuple[list[TranscriptSegment], ASRInfo]:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "faster-whisper is not installed. Install the ASR extra: pip install -e '.[asr]'"
        ) from exc

    device, compute_type = _resolve_device(cfg)
    log.info("loading Whisper %s on %s (%s)", cfg.model_size, device, compute_type)
    model = WhisperModel(cfg.model_size, device=device, compute_type=compute_type)

    t0 = time.perf_counter()
    seg_iter, info = model.transcribe(
        str(audio_path),
        language=cfg.language,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
        condition_on_previous_text=cfg.condition_on_previous_text,
        **({} if cfg.temperature_fallback else {"temperature": 0.0}),
    )
    segments: list[TranscriptSegment] = []
    # seg_iter is a lazy generator: decoding happens while iterating.
    for s in seg_iter:
        text = s.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                segment_id=len(segments),
                start=round(float(s.start), 3),
                end=round(float(max(s.end, s.start)), 3),
                text=text,
                avg_logprob=getattr(s, "avg_logprob", None),
                no_speech_prob=getattr(s, "no_speech_prob", None),
            )
        )
    wall = time.perf_counter() - t0
    audio_d = getattr(info, "duration", None)
    asr_info = ASRInfo(
        model_size=cfg.model_size,
        device=device,
        compute_type=compute_type,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        audio_duration_s=audio_d,
        duration_after_vad_s=getattr(info, "duration_after_vad", None),
        wall_time_s=round(wall, 2),
        real_time_factor=round(wall / audio_d, 4) if audio_d else None,
        settings=cfg.model_dump(mode="json"),
    )
    log.info("transcribed %d segments in %.1fs", len(segments), wall)
    return segments, asr_info


def transcript_stats(
    segments: list[TranscriptSegment], duration_s: float, suspicious_no_speech_prob: float
) -> dict[str, Any]:
    """Descriptive statistics reported in the Stage 1 results table."""
    if not segments:
        return {
            "n_segments": 0,
            "speech_coverage": 0.0,
            "mean_segment_s": None,
            "median_segment_s": None,
            "max_gap_s": round(duration_s, 2),
            "n_words": 0,
            "n_suspicious": 0,
        }
    covered = sum(e - s for s, e in merge_intervals([(x.start, min(x.end, duration_s)) for x in segments]))
    lengths = sorted(x.duration for x in segments)
    ordered = sorted(segments, key=lambda x: x.start)
    gaps = [ordered[0].start] + [
        max(0.0, b.start - a.end) for a, b in zip(ordered, ordered[1:], strict=False)
    ] + [max(0.0, duration_s - ordered[-1].end)]
    n = len(lengths)
    median = lengths[n // 2] if n % 2 else 0.5 * (lengths[n // 2 - 1] + lengths[n // 2])
    return {
        "n_segments": n,
        "speech_coverage": round(covered / duration_s, 4),
        "mean_segment_s": round(sum(lengths) / n, 2),
        "median_segment_s": round(median, 2),
        "max_gap_s": round(max(gaps), 2),
        "n_words": sum(len(x.text.split()) for x in segments),
        "n_suspicious": sum(
            1
            for x in segments
            if x.no_speech_prob is not None and x.no_speech_prob > suspicious_no_speech_prob
        ),
    }
