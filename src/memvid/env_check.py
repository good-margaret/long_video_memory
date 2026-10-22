"""Step 1: environment check.

Collects everything a reader needs to reproduce the results (Python, OS, ffmpeg,
package versions, GPU) and verifies that the requirements of Stage 1 are met.
The report is saved as JSON and exported to the LaTeX report.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata

from pydantic import BaseModel, Field

from memvid import __version__

# (distribution name, import role, required for which stage)
PACKAGES: list[tuple[str, str, bool]] = [
    ("pydantic", "core", True),
    ("PyYAML", "core", True),
    ("numpy", "core", True),
    ("pillow", "core", True),
    ("tqdm", "core", True),
    ("faster-whisper", "asr (Step 3)", False),
    ("ctranslate2", "asr backend", False),
    ("torch", "models (Stages 3-5)", False),
    ("transformers", "models (Stages 3-5)", False),
    ("accelerate", "models (Stages 3-5)", False),
    ("sentence-transformers", "memory embeddings", False),
    ("matplotlib", "analysis figures", False),
    ("gradio", "demo", False),
]
MIN_PYTHON = (3, 10)


class PackageStatus(BaseModel):
    name: str
    role: str
    required: bool
    version: str | None


class BinaryStatus(BaseModel):
    name: str
    path: str | None
    version: str | None


class GPUInfo(BaseModel):
    source: str
    name: str
    memory_gb: float | None = None


class EnvironmentReport(BaseModel):
    memvid_version: str = __version__
    python: str
    python_ok: bool
    platform: str
    machine: str
    binaries: list[BinaryStatus]
    packages: list[PackageStatus]
    cuda_available: bool | None = None
    cuda_version: str | None = None
    gpus: list[GPUInfo] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _binary(name: str) -> BinaryStatus:
    path = shutil.which(name)
    version = None
    if path:
        try:
            out = subprocess.run(
                [path, "-version"], capture_output=True, text=True, timeout=10, check=False
            ).stdout
            m = re.search(rf"{name} version (\S+)", out)
            version = m.group(1) if m else (out.splitlines()[0] if out else None)
        except (OSError, subprocess.TimeoutExpired):
            version = None
    return BinaryStatus(name=name, path=path, version=version)


def _gpus_from_torch() -> tuple[bool | None, str | None, list[GPUInfo]]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None, None, []
    available = bool(torch.cuda.is_available())
    gpus: list[GPUInfo] = []
    if available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                GPUInfo(source="torch", name=props.name, memory_gb=round(props.total_memory / 2**30, 1))
            )
    return available, getattr(torch.version, "cuda", None), gpus


def _gpus_from_nvidia_smi() -> list[GPUInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                mem = round(float(parts[1]) / 1024, 1)
            except ValueError:
                mem = None
            gpus.append(GPUInfo(source="nvidia-smi", name=parts[0], memory_gb=mem))
    return gpus


def collect_environment() -> EnvironmentReport:
    py_ok = sys.version_info[:2] >= MIN_PYTHON
    binaries = [_binary("ffmpeg"), _binary("ffprobe")]
    packages = [
        PackageStatus(name=n, role=r, required=req, version=_version(n)) for n, r, req in PACKAGES
    ]
    cuda, cuda_version, gpus = _gpus_from_torch()
    if not gpus:
        gpus = _gpus_from_nvidia_smi()

    problems: list[str] = []
    warnings: list[str] = []
    if not py_ok:
        problems.append(f"Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required")
    for b in binaries:
        if b.path is None:
            problems.append(f"system binary '{b.name}' not found (install ffmpeg)")
    for p in packages:
        if p.required and p.version is None:
            problems.append(f"required package '{p.name}' missing")
    if _version("faster-whisper") is None:
        warnings.append("faster-whisper missing: `memvid transcribe` unavailable (pip install -e '.[asr]')")
    if not gpus:
        warnings.append("no GPU detected: ASR will run on CPU (slow for large-v3); VLM stages need a GPU")

    return EnvironmentReport(
        python=platform.python_version(),
        python_ok=py_ok,
        platform=platform.platform(),
        machine=platform.machine(),
        binaries=binaries,
        packages=packages,
        cuda_available=cuda,
        cuda_version=cuda_version,
        gpus=gpus,
        problems=problems,
        warnings=warnings,
    )


def format_report(rep: EnvironmentReport) -> str:
    lines = [
        f"memvid {rep.memvid_version} | Python {rep.python} | {rep.platform}",
        "",
        "System binaries:",
    ]
    for b in rep.binaries:
        lines.append(f"  {'OK ' if b.path else 'MISSING'} {b.name:<10} {b.version or ''}")
    lines.append("Python packages:")
    for p in rep.packages:
        mark = "OK " if p.version else ("MISSING" if p.required else "absent ")
        lines.append(f"  {mark} {p.name:<15} {p.version or '-':<12} ({p.role})")
    lines.append("GPU:")
    if rep.gpus:
        for g in rep.gpus:
            lines.append(f"  {g.name} ({g.memory_gb} GB, via {g.source})")
    else:
        lines.append("  none detected")
    for w in rep.warnings:
        lines.append(f"WARNING: {w}")
    for p in rep.problems:
        lines.append(f"PROBLEM: {p}")
    lines.append("RESULT: " + ("environment OK" if rep.ok else "environment NOT OK"))
    return "\n".join(lines)
