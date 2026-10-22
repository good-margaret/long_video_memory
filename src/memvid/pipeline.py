"""Stage 1 orchestration: each step reads its inputs from the derived directory,
writes its outputs atomically, and records hashes and statistics in the manifest.

Derived layout (data/derived/<video_id>/):

    video_info.json        Step 2  ffprobe metadata + source hash
    audio.wav              Step 3  16 kHz mono PCM
    transcript.jsonl       Step 3  TranscriptSegment per line
    asr_info.json          Step 3  model, language, timing
    chunks.jsonl           Step 3  Chunk per line
    frames/index.jsonl     Step 3  FrameRecord per line
    frames/chunk_XXXX/*.jpg
    manifest.json          all     hashes, config hashes, statistics, frozen stages
"""

from __future__ import annotations

import logging
from pathlib import Path

from memvid.config import ProjectConfig
from memvid.errors import MemvidError, PipelineStateError
from memvid.io import read_json, read_jsonl, write_json, write_jsonl
from memvid.manifest import (
    Manifest,
    load_manifest,
    register_file,
    register_tree,
    save_manifest,
    verify_manifest,
)
from memvid.preprocess import asr as asr_mod
from memvid.preprocess.audio import extract_audio
from memvid.preprocess.chunking import build_chunks, chunk_stats
from memvid.preprocess.frames import frame_stats, sample_frames
from memvid.schemas import Chunk, FrameRecord, TranscriptSegment, VideoInfo
from memvid.video.probe import probe_video

log = logging.getLogger(__name__)

TREE_PATTERNS = {"frames": "*.jpg"}
MIN_DURATION_S = 1200.0  # the brief asks for a video of 20-60+ minutes


def video_duration(cfg: ProjectConfig) -> float:
    """Duration of the probed video in seconds (requires `memvid probe`)."""
    path = cfg.derived_dir / "video_info.json"
    if not path.is_file():
        raise PipelineStateError("video_info.json missing: run `memvid probe` first")
    return float(read_json(path)["duration_s"])


class Pipeline:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.d = cfg.derived_dir
        self.d.mkdir(parents=True, exist_ok=True)

    # ----- paths ---------------------------------------------------------
    @property
    def video_info_path(self) -> Path:
        return self.d / "video_info.json"

    @property
    def audio_path(self) -> Path:
        return self.d / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.d / "transcript.jsonl"

    @property
    def asr_info_path(self) -> Path:
        return self.d / "asr_info.json"

    @property
    def chunks_path(self) -> Path:
        return self.d / "chunks.jsonl"

    @property
    def frames_dir(self) -> Path:
        return self.d / "frames"

    @property
    def frames_index_path(self) -> Path:
        return self.frames_dir / "index.jsonl"

    # ----- manifest helpers ---------------------------------------------
    def manifest(self) -> Manifest:
        return load_manifest(self.d, self.cfg.video.video_id)

    def _refuse_if_frozen(self, m: Manifest, *names: str, force: bool) -> None:
        if "preprocess" in m.frozen_stages and not force and any(n in m.artifacts for n in names):
            raise MemvidError(
                f"Derived data for '{self.cfg.video.video_id}' is frozen; refusing to overwrite "
                f"{', '.join(names)}. Use --force only if you intend to invalidate all results."
            )

    # ----- loaders -------------------------------------------------------
    def video_info(self) -> VideoInfo:
        if not self.video_info_path.is_file():
            raise PipelineStateError("video_info.json missing: run `memvid probe` first")
        return VideoInfo.model_validate(read_json(self.video_info_path))

    def transcript(self) -> list[TranscriptSegment]:
        if not self.transcript_path.is_file():
            raise PipelineStateError("transcript.jsonl missing: run `memvid transcribe` first")
        return read_jsonl(self.transcript_path, TranscriptSegment)

    def chunks(self) -> list[Chunk]:
        if not self.chunks_path.is_file():
            raise PipelineStateError("chunks.jsonl missing: run `memvid chunk` first")
        return read_jsonl(self.chunks_path, Chunk)

    def frames(self) -> list[FrameRecord]:
        if not self.frames_index_path.is_file():
            raise PipelineStateError("frames/index.jsonl missing: run `memvid sample-frames` first")
        return read_jsonl(self.frames_index_path, FrameRecord)

    # ----- Step 2 --------------------------------------------------------
    def probe(self, *, compute_hash: bool = True, force: bool = False) -> VideoInfo:
        m = self.manifest()
        self._refuse_if_frozen(m, "video_info", force=force)
        src = self.cfg.video_path
        if not src.is_file():
            raise MemvidError(f"Video not found: {src} (place it in {self.cfg.paths.raw_dir}/)")
        info = probe_video(src, compute_hash=compute_hash)
        if info.duration_s < MIN_DURATION_S:
            log.warning("the video is %.0f s long; the brief asks for at least %.0f min",
                        info.duration_s, MIN_DURATION_S / 60)
        write_json(self.video_info_path, info)
        register_file(m, self.d, "video_info", self.video_info_path, step="probe")
        m.stats["video"] = {
            "duration_s": round(info.duration_s, 2),
            "resolution": f"{info.width}x{info.height}",
            "fps": round(info.fps, 3),
            "video_codec": info.video_codec,
            "has_audio": info.has_audio,
            "source_sha256": info.sha256,
        }
        save_manifest(self.d, m)
        return info

    # ----- Step 3 --------------------------------------------------------
    def extract_audio(self, *, force: bool = False) -> Path:
        m = self.manifest()
        self._refuse_if_frozen(m, "audio", force=force)
        info = self.video_info()
        if not info.has_audio:
            raise MemvidError("The video has no audio stream")
        extract_audio(self.cfg.video_path, self.audio_path, self.cfg.audio,
                      video_duration_s=info.duration_s)
        register_file(m, self.d, "audio", self.audio_path, step="extract_audio",
                      config_hash=self.cfg.section_hash("audio"))
        save_manifest(self.d, m)
        return self.audio_path

    def transcribe(self, *, force: bool = False) -> list[TranscriptSegment]:
        m = self.manifest()
        self._refuse_if_frozen(m, "transcript", "asr_info", force=force)
        if not self.audio_path.is_file():
            raise PipelineStateError("audio.wav missing: run `memvid extract-audio` first")
        segments, info = asr_mod.transcribe(self.audio_path, self.cfg.asr)
        write_jsonl(self.transcript_path, segments)
        write_json(self.asr_info_path, info)
        h = self.cfg.section_hash("asr", "audio")
        register_file(m, self.d, "transcript", self.transcript_path, step="transcribe", config_hash=h)
        register_file(m, self.d, "asr_info", self.asr_info_path, step="transcribe", config_hash=h)
        m.stats["asr"] = {
            **asr_mod.transcript_stats(
                segments, self.video_info().duration_s, self.cfg.asr.suspicious_no_speech_prob
            ),
            "language": info.language,
            "language_probability": info.language_probability,
            "model_size": info.model_size,
            "device": info.device,
            "compute_type": info.compute_type,
            "real_time_factor": info.real_time_factor,
        }
        save_manifest(self.d, m)
        return segments

    def chunk(self, *, allow_missing_transcript: bool = False, force: bool = False) -> list[Chunk]:
        m = self.manifest()
        self._refuse_if_frozen(m, "chunks", force=force)
        info = self.video_info()
        if self.transcript_path.is_file():
            segments = self.transcript()
        elif allow_missing_transcript:
            log.warning("no transcript: chunks will have empty transcript_text")
            segments = []
        else:
            raise PipelineStateError(
                "transcript.jsonl missing: run `memvid transcribe` "
                "(or pass --allow-missing-transcript for a visual-only smoke test)"
            )
        chunks = build_chunks(info.duration_s, segments, self.cfg.chunking)
        write_jsonl(self.chunks_path, chunks)
        register_file(m, self.d, "chunks", self.chunks_path, step="chunk",
                      config_hash=self.cfg.section_hash("chunking"))
        m.stats["chunks"] = {**chunk_stats(chunks, segments),
                             "chunk_seconds": self.cfg.chunking.chunk_seconds,
                             "transcript_available": bool(segments) or self.transcript_path.is_file()}
        save_manifest(self.d, m)
        return chunks

    def sample_frames(self, *, progress: bool = True, force: bool = False) -> list[FrameRecord]:
        m = self.manifest()
        self._refuse_if_frozen(m, "frames", "frames_index", force=force)
        info = self.video_info()
        chunks = self.chunks()
        records = sample_frames(
            self.cfg.video_path, chunks, self.frames_dir, self.d, self.cfg.frames,
            fps=info.fps, progress=progress,
        )
        write_jsonl(self.frames_index_path, records)
        h = self.cfg.section_hash("frames", "chunking")
        register_tree(m, self.d, "frames", self.frames_dir, "*.jpg", step="sample_frames", config_hash=h)
        register_file(m, self.d, "frames_index", self.frames_index_path, step="sample_frames",
                      config_hash=h)
        m.stats["frames"] = {**frame_stats(records, info.fps),
                             "frames_per_chunk": self.cfg.frames.frames_per_chunk}
        save_manifest(self.d, m)
        return records

    def preprocess(self, *, skip_asr: bool = False, progress: bool = True, force: bool = False) -> None:
        """Steps 2 (probe) and 3 end to end."""
        info = self.probe(force=force)
        if info.has_audio:
            self.extract_audio(force=force)
            if not skip_asr:
                self.transcribe(force=force)
        elif not skip_asr:
            raise MemvidError("Video has no audio stream; rerun with --skip-asr")
        self.chunk(allow_missing_transcript=skip_asr, force=force)
        self.sample_frames(progress=progress, force=force)

    # ----- freezing ------------------------------------------------------
    def verify(self) -> list[str]:
        return verify_manifest(self.manifest(), self.d, TREE_PATTERNS)

    def freeze(self) -> Manifest:
        required = ["video_info", "audio", "transcript", "chunks", "frames", "frames_index"]
        m = self.manifest()
        missing = [r for r in required if r not in m.artifacts]
        if missing:
            raise PipelineStateError(f"Cannot freeze, artifacts missing: {missing}")
        problems = self.verify()
        if problems:
            raise MemvidError("Cannot freeze, verification failed:\n  " + "\n  ".join(problems))
        if "preprocess" not in m.frozen_stages:
            m.frozen_stages.append("preprocess")
        save_manifest(self.d, m)
        return m
