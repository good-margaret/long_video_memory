"""Audio extraction: video -> 16 kHz mono 16-bit PCM WAV (the format Whisper expects)."""

from __future__ import annotations

import logging
from pathlib import Path

from memvid._proc import run
from memvid.config import AudioConfig
from memvid.errors import MemvidError
from memvid.video.probe import media_duration

log = logging.getLogger(__name__)


def extract_audio(
    video: str | Path, out: str | Path, cfg: AudioConfig, *, video_duration_s: float | None = None
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.wav")
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            "-map", "0:a:0",  # first audio stream; fails loudly if there is none
            "-vn", "-ac", str(cfg.channels), "-ar", str(cfg.sample_rate),
            "-c:a", "pcm_s16le",
            str(tmp),
        ]
    )
    tmp.replace(out)

    if video_duration_s is not None:
        audio_d = media_duration(out)
        diff = abs(audio_d - video_duration_s)
        if diff > cfg.max_duration_mismatch_s:
            raise MemvidError(
                f"Audio duration {audio_d:.2f}s differs from video duration "
                f"{video_duration_s:.2f}s by {diff:.2f}s (> {cfg.max_duration_mismatch_s}s). "
                f"Timestamps derived from the audio would be misaligned with frames."
            )
        log.info("audio duration %.2fs (video %.2fs, diff %.3fs)", audio_d, video_duration_s, diff)
    return out
