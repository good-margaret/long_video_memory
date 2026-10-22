"""Robust extraction of one JSON object from model output.

Small models wrap JSON in prose or code fences, or append text after it. We take the
first balanced {...} block (string-aware), then parse it strictly. No repair heuristics
beyond that: malformed output goes back to the model with the parser error (see
`generate_validated`).
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from memvid.models.base import GenerativeModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class JSONExtractionError(ValueError):
    pass


def extract_json_block(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise JSONExtractionError("no '{' in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise JSONExtractionError("unbalanced braces in model output")


def parse_json_object(text: str) -> dict[str, Any]:
    block = extract_json_block(text)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        raise JSONExtractionError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise JSONExtractionError("top-level JSON value is not an object")
    return obj


REPAIR_SUFFIX = (
    "\n\nYour previous answer could not be used:\n{error}\n"
    "Previous answer:\n{previous}\n\n"
    "Reply again with ONLY one JSON object that satisfies the schema. No prose."
)


class GenerationResult(BaseModel):
    ok: bool
    value: Any = None
    raw: list[str]
    error: str | None = None
    attempts: int


def generate_validated(
    model: GenerativeModel,
    prompt: str,
    schema: type[T],
    *,
    images=None,
    max_new_tokens: int | None = None,
    repair_attempts: int = 1,
) -> GenerationResult:
    """Generate, extract JSON, validate against `schema`; on failure send the error back
    up to `repair_attempts` times. Never raises on model errors: callers log the failure."""
    json_schema = schema.model_json_schema()
    raws: list[str] = []
    current = prompt
    error = None
    for attempt in range(repair_attempts + 1):
        text = model.generate(current, images=images, json_schema=json_schema,
                              max_new_tokens=max_new_tokens)
        raws.append(text)
        try:
            obj = parse_json_object(text)
            value = schema.model_validate(obj)
            return GenerationResult(ok=True, value=value, raw=raws, attempts=attempt + 1)
        except (JSONExtractionError, ValidationError) as exc:
            error = str(exc)[:1500]
            log.debug("invalid model output (attempt %d): %s", attempt + 1, error)
            current = prompt + REPAIR_SUFFIX.format(error=error, previous=text[:3000])
    return GenerationResult(ok=False, raw=raws, error=error, attempts=repair_attempts + 1)
