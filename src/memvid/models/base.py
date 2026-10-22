"""Interfaces shared by all backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class GenerativeModel(ABC):
    """A text generator that optionally accepts images (a VLM) and an output JSON schema.

    Implementations must be deterministic for identical inputs (greedy decoding), because
    every stage that calls them caches and freezes its outputs.
    """

    name: str = "model"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Return the generated text (the assistant turn only)."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}
