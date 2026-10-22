"""Write LaTeX table fragments into report/generated/.

The report includes each fragment with \\IfFileExists, so it compiles before any
data exist and fills in automatically once `memvid export-tex` has been run.
Numbers in the report therefore always come from the code, never from copying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memvid.config import ProjectConfig
from memvid.env_check import EnvironmentReport
from memvid.io import atomic_write_text, read_json
from memvid.manifest import MANIFEST_NAME, load_manifest
from memvid.timeutils import format_timestamp

_ESC = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


class Tex(str):
    """A string that is already valid LaTeX and must not be escaped."""


def tex_escape(value: Any) -> str:
    if isinstance(value, Tex):
        return str(value)
    return "".join(_ESC.get(ch, ch) for ch in str(value))


def _fmt(v: Any) -> str:
    if isinstance(v, Tex):
        return str(v)
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return tex_escape(v)


def kv_table(rows: list[tuple[str, Any]], caption: str, label: str) -> str:
    body = "\n".join(f"{tex_escape(k)} & {_fmt(v)} \\\\" for k, v in rows)
    return (
        "\\begin{table}[ht]\n\\centering\\small\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        "\\begin{tabular}{@{}ll@{}}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def config_table(cfg: ProjectConfig) -> str:
    rows = [
        ("Audio sample rate (Hz) / channels", f"{cfg.audio.sample_rate} / {cfg.audio.channels}"),
        ("ASR model", cfg.asr.model_size),
        ("ASR beam size", cfg.asr.beam_size),
        ("ASR VAD filter", cfg.asr.vad_filter),
        ("ASR condition on previous text", cfg.asr.condition_on_previous_text),
        ("ASR language", cfg.asr.language or "auto-detect"),
        ("ASR temperature fallback", cfg.asr.temperature_fallback),
        (Tex(r"Chunk length $L$ (s)"), cfg.chunking.chunk_seconds),
        (Tex(r"Min.\ last-chunk fraction $\alpha$"), cfg.chunking.min_last_chunk_fraction),
        (Tex(r"Min.\ segment overlap $\delta$ (s)"), cfg.chunking.transcript_min_overlap_s),
        (Tex(r"Frames per chunk $N$"), cfg.frames.frames_per_chunk),
        ("Frame max side (px)", cfg.frames.max_side),
        ("JPEG quality", cfg.frames.jpeg_quality),
        ("Random seed", cfg.seed),
    ]
    return kv_table(rows, "Stage~1 preprocessing parameters (generated from the configuration).",
                    "tab:stage1-params")


def environment_table(rep: EnvironmentReport) -> str:
    rows: list[tuple[str, Any]] = [
        ("memvid", rep.memvid_version),
        ("Python", rep.python),
        ("Platform", rep.platform),
    ]
    rows += [(b.name, b.version or "missing") for b in rep.binaries]
    rows += [(p.name, p.version) for p in rep.packages if p.version]
    rows += [("GPU", f"{g.name} ({g.memory_gb} GB)") for g in rep.gpus] or [("GPU", "none")]
    if rep.cuda_version:
        rows.append(("CUDA (torch)", rep.cuda_version))
    return kv_table(rows, "Software and hardware environment.", "tab:environment")


def preprocessing_table(stats: dict[str, dict[str, Any]]) -> str:
    v, a, c, f = (stats.get(k, {}) for k in ("video", "asr", "chunks", "frames"))
    rows: list[tuple[str, Any]] = []
    if v:
        rows += [
            ("Video duration", format_timestamp(v["duration_s"])),
            ("Resolution / frame rate", f"{v['resolution']} @ {v['fps']} fps"),
        ]
    if a:
        rows += [
            ("ASR segments", a.get("n_segments")),
            ("Words transcribed", a.get("n_words")),
            ("Speech coverage (fraction of duration)", a.get("speech_coverage")),
            ("Mean / median segment length (s)",
             f"{_fmt(a.get('mean_segment_s'))} / {_fmt(a.get('median_segment_s'))}"),
            ("Longest gap without speech (s)", a.get("max_gap_s")),
            ("Suspicious segments (high no-speech prob.)", a.get("n_suspicious")),
            ("Detected language (probability)",
             f"{a.get('language')} ({_fmt(a.get('language_probability'))})"),
            ("ASR real-time factor", a.get("real_time_factor")),
        ]
    if c:
        rows += [
            ("Chunks", c.get("n_chunks")),
            ("Chunk length min / max (s)", f"{_fmt(c.get('min_chunk_s'))} / {_fmt(c.get('max_chunk_s'))}"),
            ("Chunks containing speech", c.get("frac_chunks_with_speech")),
            ("Segments in two or more chunks", c.get("segments_in_two_or_more_chunks")),
        ]
    if f:
        rows += [
            ("Frames sampled", f.get("n_frames")),
            ("Frame size", f.get("image_size")),
            ("Max / mean frame timing error (s)",
             f"{_fmt(f.get('max_timing_error_s'))} / {_fmt(f.get('mean_timing_error_s'))}"),
        ]
    return kv_table(rows, "Stage~1 preprocessing results (generated from the manifest).",
                    "tab:stage1-results")


def export_all(cfg: ProjectConfig, env_report_path: Path | None = None) -> list[Path]:
    out_dir = cfg.report_generated_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def put(name: str, text: str) -> None:
        p = out_dir / name
        atomic_write_text(p, "% Generated by `memvid export-tex`. Do not edit.\n" + text)
        written.append(p)

    put("stage1_params.tex", config_table(cfg))

    env_path = env_report_path or (cfg.runs_dir / "environment.json")
    if env_path.is_file():
        put("environment.tex", environment_table(EnvironmentReport.model_validate(read_json(env_path))))

    # Read-only access: do not instantiate Pipeline here, it creates the derived directory.
    if (cfg.derived_dir / MANIFEST_NAME).is_file():
        m = load_manifest(cfg.derived_dir, cfg.video.video_id)
        if m.stats:
            put("stage1_results.tex", preprocessing_table(m.stats))
    return written
