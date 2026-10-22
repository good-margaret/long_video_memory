"""Progress bars (tqdm), switched off with --quiet."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress_iter(items: Iterable[T], *, desc: str, enabled: bool = True) -> Iterator[T]:
    return iter(tqdm(items, desc=desc, disable=not enabled))
