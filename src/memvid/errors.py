"""Exception hierarchy. All package errors derive from MemvidError so the CLI can
report them cleanly without a traceback."""


class MemvidError(Exception):
    """Base class for all memvid errors."""


class ConfigError(MemvidError):
    """Invalid or inconsistent configuration."""


class MissingDependencyError(MemvidError):
    """An optional Python package or a system binary is not available."""


class ExternalToolError(MemvidError):
    """An external process (ffmpeg, ffprobe) failed."""


class PipelineStateError(MemvidError):
    """A pipeline step was run before the step it depends on."""
