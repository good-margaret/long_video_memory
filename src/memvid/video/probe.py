"""Video metadata via ffprobe."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from memvid._proc import run
from memvid.errors import MemvidError
from memvid.io import sha256_file
from memvid.schemas import VideoInfo


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        value = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def probe_video(path: str | Path, *, compute_hash: bool = True) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise MemvidError(f"Video not found: {path}")
    out = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    ).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MemvidError(f"No video stream in {path}")

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    if fps is None:
        raise MemvidError(f"Cannot determine frame rate of {path}")
    duration = _float(data.get("format", {}).get("duration")) or _float(video.get("duration"))
    if not duration:
        raise MemvidError(f"Cannot determine duration of {path}")
    n_frames = video.get("nb_frames")

    return VideoInfo(
        path=str(path),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path) if compute_hash else None,
        duration_s=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        video_codec=video.get("codec_name", "unknown"),
        n_frames=int(n_frames) if n_frames and str(n_frames).isdigit() else None,
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        audio_channels=int(audio["channels"]) if audio and audio.get("channels") else None,
    )


def media_duration(path: str | Path) -> float:
    """Container duration of any media file (used to verify extracted audio)."""
    out = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    ).stdout
    d = _float(json.loads(out).get("format", {}).get("duration"))
    if d is None:
        raise MemvidError(f"Cannot determine duration of {path}")
    return d
