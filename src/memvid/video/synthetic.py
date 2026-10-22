"""Synthetic test video for smoke tests (no real footage needed).

Video: ffmpeg's `testsrc2` pattern, which changes every frame, so frame-accuracy
of the sampler can be checked pixel-wise. Audio: a tone whose frequency steps
every 5 s. Neither contains speech, so ASR is expected to return few or no
segments on it; use a real video to test transcription.
"""

from __future__ import annotations

from pathlib import Path

from memvid._proc import run


def make_test_video(
    out: str | Path,
    *,
    duration_s: float = 60.0,
    fps: int = 25,
    size: str = "640x360",
    gop: int = 50,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=duration={duration_s}:size={size}:rate={fps}",
            "-f", "lavfi", "-i", f"sine=frequency=440:beep_factor=4:duration={duration_s}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(gop), "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", "-shortest",
            str(out),
        ]
    )
    return out
