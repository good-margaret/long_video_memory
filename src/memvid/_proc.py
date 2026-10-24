"""Thin wrapper around subprocess for calling ffmpeg / ffprobe."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence

from memvid.errors import ExternalToolError, MissingDependencyError


def require_binary(name: str) -> str:
    """Return the absolute path of a system binary or raise a clear error."""
    override = os.environ.get(f"MEMVID_{name.upper()}")
    path = shutil.which(override or name)
    if path is None:
        raise MissingDependencyError(
            f"'{name}' was not found on PATH. Install ffmpeg (which provides ffmpeg and "
            f"ffprobe) with your system package manager, e.g. `apt install ffmpeg` or "
            f"`brew install ffmpeg`."
        )
    return path


def run(cmd: Sequence[str], *, binary_stdout: bool = False) -> subprocess.CompletedProcess:
    """Run a command, raising ExternalToolError with stderr on failure."""
    executable = require_binary(cmd[0])
    proc = subprocess.run([executable, *cmd[1:]], capture_output=True, text=not binary_stdout, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr if isinstance(proc.stderr, str) else proc.stderr.decode(errors="replace")
        raise ExternalToolError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n{stderr.strip()[-2000:]}"
        )
    return proc
