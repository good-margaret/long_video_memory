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
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from memvid.config import ASRConfig
from memvid.errors import MemvidError, MissingDependencyError
from memvid.schemas import TranscriptSegment
from memvid.timeutils import merge_intervals

log = logging.getLogger(__name__)


class ASRInfo(BaseModel):
    source: str = "whisper"
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


_SRT_TIME = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})(?:\s+.*)?$"
)
_SRT_TAG = re.compile(r"<[^>]+>")


def _srt_seconds(parts: tuple[str, str, str, str]) -> float:
    h, m, s, ms = (int(x) for x in parts)
    return 3600 * h + 60 * m + s + ms / 1000


def import_srt(path: str | Path, cfg: ASRConfig) -> tuple[list[TranscriptSegment], ASRInfo]:
    """Import ordinary SRT subtitles into the transcript contract used by the pipeline."""
    path = Path(path)
    if not path.is_file():
        raise MemvidError(f"subtitle file not found: {path}")
    t0 = time.perf_counter()
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    segments: list[TranscriptSegment] = []
    for block_no, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_idx is None:
            continue
        match = _SRT_TIME.match(lines[time_idx])
        if match is None:
            raise MemvidError(f"{path}: subtitle block {block_no} has an invalid timestamp")
        start = _srt_seconds(match.groups()[:4])
        end = _srt_seconds(match.groups()[4:])
        body = " ".join(lines[time_idx + 1 :])
        body = _SRT_TAG.sub("", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            continue
        segments.append(
            TranscriptSegment(
                segment_id=len(segments), start=round(start, 3), end=round(end, 3), text=body
            )
        )
    if not segments:
        raise MemvidError(f"{path}: no subtitle segments found")
    if any(b.start < a.start for a, b in zip(segments, segments[1:], strict=False)):
        raise MemvidError(f"{path}: subtitle timestamps are not ordered")
    wall = time.perf_counter() - t0
    duration = max(s.end for s in segments)
    return segments, ASRInfo(
        source="srt",
        model_size="provided-subtitles",
        device="cpu",
        compute_type="text",
        language=cfg.language,
        language_probability=None,
        audio_duration_s=duration,
        duration_after_vad_s=None,
        wall_time_s=round(wall, 3),
        real_time_factor=round(wall / duration, 6) if duration else None,
        settings=cfg.model_dump(mode="json"),
    )


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


def load_pcm_wav(path: str | Path, expected_sample_rate: int = 16000) -> np.ndarray:
    """Load the pipeline's PCM WAV without PyAV.

    faster-whisper accepts a float32 waveform directly. This avoids incompatibilities
    between the PyAV version preinstalled in hosted notebooks and faster-whisper's
    file-decoding wrapper.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1:
        raise MemvidError(f"expected mono WAV, got {channels} channels: {path}")
    if sample_rate != expected_sample_rate:
        raise MemvidError(
            f"expected {expected_sample_rate} Hz WAV, got {sample_rate} Hz: {path}"
        )
    if sample_width != 2:
        raise MemvidError(f"expected 16-bit PCM WAV, got {8 * sample_width}-bit: {path}")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


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
    audio = load_pcm_wav(audio_path)
    seg_iter, info = model.transcribe(
        audio,
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
        source="whisper",
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
