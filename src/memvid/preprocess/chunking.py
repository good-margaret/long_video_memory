"""Fixed-length temporal chunking and transcript-to-chunk assignment.

Definitions (also stated in the report):

* Chunk bounds. With video duration T and chunk length L, let n = floor(T / L)
  and r = T - nL. Chunks are [iL, (i+1)L) for i < n. If r >= alpha*L the
  remainder forms an extra chunk [nL, T); otherwise it is merged into the last
  chunk, so no chunk is shorter than alpha*L (unless T itself is shorter).

* Segment assignment. Segment s = [a, b] is attached to chunk c = [u, v) iff
  overlap(s, c) = min(b, v) - max(a, u) >= min(delta, b - a), with delta the
  minimum overlap. A segment straddling a boundary may belong to two chunks;
  this duplication is intentional, because speech near a boundary is evidence
  for both chunks.
"""

from __future__ import annotations

import math
from typing import Any

from memvid.config import ChunkingConfig
from memvid.schemas import Chunk, TranscriptSegment

_EPS = 1e-6


def make_chunk_bounds(
    duration_s: float, chunk_seconds: float, min_last_fraction: float
) -> list[tuple[float, float]]:
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    n = int(math.floor(duration_s / chunk_seconds + _EPS))
    bounds = [(round(i * chunk_seconds, 3), round((i + 1) * chunk_seconds, 3)) for i in range(n)]
    remainder = duration_s - n * chunk_seconds
    if remainder > _EPS:
        if bounds and remainder < min_last_fraction * chunk_seconds:
            bounds[-1] = (bounds[-1][0], round(duration_s, 3))
        else:
            bounds.append((round(n * chunk_seconds, 3), round(duration_s, 3)))
    elif bounds:
        bounds[-1] = (bounds[-1][0], round(duration_s, 3))
    return bounds


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_segments(
    bounds: list[tuple[float, float]], segments: list[TranscriptSegment], min_overlap_s: float
) -> list[list[int]]:
    """For each chunk, the ids of segments attached to it (in temporal order)."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    result: list[list[int]] = []
    for u, v in bounds:
        ids = []
        for s in ordered:
            if s.start >= v:
                break
            need = min(min_overlap_s, s.duration)
            ov = overlap(s.start, s.end, u, v)
            # zero-length segments: attach to the chunk containing their timestamp
            if (s.duration <= _EPS and u <= s.start < v) or (ov > 0 and ov + _EPS >= need):
                ids.append(s.segment_id)
        result.append(ids)
    return result


def build_chunks(
    duration_s: float, segments: list[TranscriptSegment], cfg: ChunkingConfig
) -> list[Chunk]:
    bounds = make_chunk_bounds(duration_s, cfg.chunk_seconds, cfg.min_last_chunk_fraction)
    by_id = {s.segment_id: s for s in segments}
    assignment = assign_segments(bounds, segments, cfg.transcript_min_overlap_s)
    return [
        Chunk(
            chunk_id=i,
            t_start=u,
            t_end=v,
            transcript_text=" ".join(by_id[j].text for j in ids),
            segment_ids=ids,
        )
        for i, ((u, v), ids) in enumerate(zip(bounds, assignment, strict=True))
    ]


def chunk_stats(chunks: list[Chunk], segments: list[TranscriptSegment]) -> dict[str, Any]:
    n = len(chunks)
    assigned = {j for c in chunks for j in c.segment_ids}
    multi = sum(1 for s in segments if sum(s.segment_id in c.segment_ids for c in chunks) > 1)
    durations = [c.duration for c in chunks]
    return {
        "n_chunks": n,
        "min_chunk_s": round(min(durations), 3) if durations else None,
        "max_chunk_s": round(max(durations), 3) if durations else None,
        "chunks_with_speech": sum(1 for c in chunks if c.segment_ids),
        "frac_chunks_with_speech": round(sum(1 for c in chunks if c.segment_ids) / n, 4) if n else 0,
        "segments_unassigned": len(segments) - len(assigned),
        "segments_in_two_or_more_chunks": multi,
    }
