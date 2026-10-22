"""Provenance manifest for derived artifacts.

data/derived/<video_id>/manifest.json records, for every artifact, its hash, the
step that produced it, and the hash of the config sections that determined it.
`memvid verify` recomputes the hashes; this is how "frozen" inputs are enforced:
if an artifact changes after it was frozen, every downstream result is suspect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from memvid import __version__
from memvid.io import read_json, sha256_file, sha256_tree, write_json

MANIFEST_NAME = "manifest.json"


class ArtifactRecord(BaseModel):
    name: str
    path: str  # relative to the derived directory
    kind: str = "file"  # "file" or "tree"
    sha256: str
    size_bytes: int
    step: str
    config_hash: str | None = None
    created_at: str
    package_version: str = __version__


class Manifest(BaseModel):
    video_id: str
    frozen_stages: list[str] = Field(default_factory=list)  # e.g. ["preprocess", "perception"]
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    stats: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(derived_dir: Path, video_id: str) -> Manifest:
    path = derived_dir / MANIFEST_NAME
    if path.is_file():
        return Manifest.model_validate(read_json(path))
    return Manifest(video_id=video_id)


def save_manifest(derived_dir: Path, manifest: Manifest) -> None:
    write_json(derived_dir / MANIFEST_NAME, manifest)


def register_file(
    manifest: Manifest,
    derived_dir: Path,
    name: str,
    path: Path,
    *,
    step: str,
    config_hash: str | None = None,
) -> ArtifactRecord:
    rec = ArtifactRecord(
        name=name,
        path=path.resolve().relative_to(derived_dir.resolve()).as_posix(),
        kind="file",
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        step=step,
        config_hash=config_hash,
        created_at=_now(),
    )
    manifest.artifacts[name] = rec
    return rec


def register_tree(
    manifest: Manifest,
    derived_dir: Path,
    name: str,
    directory: Path,
    pattern: str,
    *,
    step: str,
    config_hash: str | None = None,
) -> ArtifactRecord:
    files = sorted(p for p in directory.rglob(pattern) if p.is_file())
    rec = ArtifactRecord(
        name=name,
        path=directory.resolve().relative_to(derived_dir.resolve()).as_posix(),
        kind="tree",
        sha256=sha256_tree(files, directory),
        size_bytes=sum(p.stat().st_size for p in files),
        step=step,
        config_hash=config_hash,
        created_at=_now(),
    )
    manifest.artifacts[name] = rec
    return rec


def verify_manifest(manifest: Manifest, derived_dir: Path, tree_patterns: dict[str, str]) -> list[str]:
    """Return a list of human-readable problems (empty list == all artifacts intact)."""
    problems: list[str] = []
    for name, rec in sorted(manifest.artifacts.items()):
        target = derived_dir / rec.path
        if not target.exists():
            problems.append(f"{name}: missing ({rec.path})")
            continue
        if rec.kind == "file":
            digest = sha256_file(target)
        else:
            pattern = tree_patterns.get(name, "*")
            files = sorted(p for p in target.rglob(pattern) if p.is_file())
            digest = sha256_tree(files, target)
        if digest != rec.sha256:
            problems.append(f"{name}: hash mismatch ({rec.path})")
    return problems
