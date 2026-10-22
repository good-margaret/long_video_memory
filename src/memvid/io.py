"""JSON / JSONL persistence and file hashing.

All writes are atomic (temporary file + rename) so an interrupted run never
leaves a half-written artifact that a later step might silently consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _to_jsonable(record: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    return record


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: str | Path, obj: BaseModel | dict[str, Any], *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(_to_jsonable(obj), indent=indent, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict[str, Any]]) -> int:
    """Write records one per line. Returns the number of records written."""
    lines = [json.dumps(_to_jsonable(r), ensure_ascii=False) for r in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def read_jsonl(path: str | Path, model: type[T] | None = None) -> list[Any]:
    """Read a JSONL file, optionally validating every line against a model."""
    rows = list(iter_jsonl(path))
    if model is None:
        return rows
    return [model.model_validate(r) for r in rows]


def sha256_file(path: str | Path, *, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(block_size):
            h.update(block)
    return h.hexdigest()


def sha256_tree(paths: Iterable[str | Path], root: str | Path) -> str:
    """Order-independent digest of many files: hash of sorted (relpath, filehash) pairs."""
    root = Path(root)
    entries = sorted(
        (Path(p).resolve().relative_to(root.resolve()).as_posix(), sha256_file(p)) for p in paths
    )
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(f"{rel}\t{digest}\n".encode())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
