"""Per-chunk frame sampling.

Sampling times. For chunk [u, v) and N frames per chunk, frame k is taken at the
midpoint of the k-th of N equal sub-intervals:

    t_k = u + (k + 1/2) (v - u) / N,   k = 0..N-1.

Midpoints (rather than numpy.linspace endpoints) never fall on a chunk boundary,
so consecutive chunks never share a frame, and spacing stays uniform across
boundaries (gap (v-u)/N everywhere when chunks have equal length).

Decoding. One ffmpeg process per chunk with input seeking (`-ss` before `-i`,
frame-accurate when decoding) and a `select` filter that keeps the first frame at or
after each target. The decoded presentation time is read from the `showinfo` filter.
No Python dependency; exact timestamps.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

from PIL import Image

from memvid._proc import run
from memvid._progress import progress_iter
from memvid.config import FramesConfig
from memvid.errors import MemvidError
from memvid.schemas import Chunk, FrameRecord

log = logging.getLogger(__name__)
_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")


def frame_targets(t_start: float, t_end: float, n: int) -> list[float]:
    if n < 1:
        raise ValueError("n must be >= 1")
    step = (t_end - t_start) / n
    return [round(t_start + (k + 0.5) * step, 4) for k in range(n)]


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------- backends
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def split_png_stream(data: bytes) -> list[bytes]:
    """Split concatenated PNG files (ffmpeg image2pipe output) at IEND chunks."""
    images, pos = [], 0
    while pos < len(data):
        if data[pos : pos + 8] != _PNG_SIG:
            raise MemvidError(f"Corrupt PNG stream at byte {pos}")
        p = pos + 8
        while True:
            if p + 8 > len(data):
                raise MemvidError("Truncated PNG stream")
            length = int.from_bytes(data[p : p + 4], "big")
            ctype = data[p + 4 : p + 8]
            p += 12 + length  # length + type + data + crc
            if ctype == b"IEND":
                break
        images.append(data[pos:p])
        pos = p
    return images


class FFmpegFrameReader:
    """Frame-accurate extraction via ffmpeg input seeking (`-ss` before `-i`).

    One ffmpeg process per chunk: seek to the chunk start, then a `select`
    filter keeps, for each target offset o_k, the first frame with t >= o_k.
    `showinfo` (placed after `select`) reports each kept frame's pts_time, which
    restarts near 0 at the seek point, so t_actual = chunk_start + pts_time.
    Selection rule: the first frame at or after the target, so the timing error
    is in [0, 1/fps).
    """

    def __init__(self, video: str | Path, fps: float):
        self.video = str(video)
        self.frame_dur = 1.0 / fps

    @staticmethod
    def _select_expr(offsets: list[float]) -> str:
        terms = [
            f"gte(t,{o:.4f})*(isnan(prev_selected_t)+lt(prev_selected_t,{o:.4f}))" for o in offsets
        ]
        return "+".join(terms)

    def _decode(self, start: float, duration: float, offsets: list[float]) -> list[tuple[Image.Image, float]]:
        proc = run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
                "-ss", f"{start:.4f}", "-t", f"{duration:.4f}", "-i", self.video,
                "-map", "0:v:0",
                "-vf", f"select='{self._select_expr(offsets)}',showinfo",
                "-fps_mode", "passthrough",
                "-f", "image2pipe", "-c:v", "png", "-",
            ],
            binary_stdout=True,
        )
        pngs = split_png_stream(proc.stdout) if proc.stdout else []
        pts = [float(x) for x in _PTS_RE.findall(proc.stderr.decode(errors="replace"))]
        if len(pts) != len(pngs):
            raise MemvidError(f"ffmpeg returned {len(pngs)} images but {len(pts)} timestamps")
        out = []
        for data, p in zip(pngs, pts, strict=True):
            img = Image.open(io.BytesIO(data))
            img.load()
            out.append((img.convert("RGB"), start + p))
        return out

    def get_many(self, targets: list[float]) -> list[tuple[Image.Image, float]]:
        """Frames for sorted targets that lie close together (one chunk)."""
        if not targets:
            return []
        start = targets[0]
        span = targets[-1] - start + 2 * self.frame_dur
        offsets = [t - start for t in targets]
        got = self._decode(start, span, offsets)
        if len(got) == len(targets):
            return got
        # Fewer frames than targets: targets closer than one frame period, or the
        # last targets lie beyond the final frame. Resolve each target separately.
        return [self.get(t) for t in targets]

    def get(self, t: float) -> tuple[Image.Image, float]:
        # Near the end of the stream a seek can land past the last frame; step back.
        for k in range(8):
            ts = max(0.0, t - k * self.frame_dur)
            got = self._decode(ts, 2 * self.frame_dur, [0.0])
            if got:
                return got[0]
        raise MemvidError(f"ffmpeg returned no frame near t={t:.3f}s")


# --------------------------------------------------------------------- pipeline
def sample_frames(
    video: Path,
    chunks: list[Chunk],
    out_dir: Path,
    derived_dir: Path,
    cfg: FramesConfig,
    *,
    fps: float,
    progress: bool = True,
) -> list[FrameRecord]:
    """Extract N frames per chunk to out_dir/chunk_XXXX/fYY.jpg. Returns the index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = FFmpegFrameReader(video, fps)
    records: list[FrameRecord] = []
    for c in progress_iter(chunks, desc="chunks", enabled=progress):
        targets = frame_targets(c.t_start, c.t_end, cfg.frames_per_chunk)
        frames = reader.get_many(targets)
        for k, (t, (img, t_actual)) in enumerate(zip(targets, frames, strict=True)):
            img = resize_max_side(img, cfg.max_side)
            path = out_dir / f"chunk_{c.chunk_id:04d}" / f"f{k:02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, format="JPEG", quality=cfg.jpeg_quality)
            records.append(
                FrameRecord(
                    chunk_id=c.chunk_id,
                    frame_index=k,
                    t_target=t,
                    t_actual=round(t_actual, 4),
                    path=path.resolve().relative_to(derived_dir.resolve()).as_posix(),
                    width=img.width,
                    height=img.height,
                )
            )
    return records


def frame_stats(records: list[FrameRecord], fps: float) -> dict:
    errs = [abs(r.t_actual - r.t_target) for r in records if r.t_actual is not None]
    return {
        "n_frames": len(records),
        "max_timing_error_s": round(max(errs), 4) if errs else None,
        "mean_timing_error_s": round(sum(errs) / len(errs), 4) if errs else None,
        "frames_beyond_one_frame_period": sum(1 for e in errs if e > 1.0 / fps + 1e-3),
        "image_size": f"{records[0].width}x{records[0].height}" if records else None,
    }
