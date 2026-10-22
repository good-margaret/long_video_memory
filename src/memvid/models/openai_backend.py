"""Any OpenAI-compatible chat-completions server, e.g. `vllm serve Qwen/Qwen3-VL-4B-Instruct`.

Uses only the standard library. Images are sent as base64 JPEG data URLs. With
`guided_json: true` the request carries `response_format: json_schema`, which vLLM
enforces with constrained decoding, removing JSON parse failures.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any

from memvid.config import ModelSpec
from memvid.errors import ConfigError, ExternalToolError
from memvid.models.base import GenerativeModel


def _data_url(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class OpenAICompatibleModel(GenerativeModel):
    def __init__(self, spec: ModelSpec):
        if not spec.base_url:
            raise ConfigError("models.*.base_url is required for the openai backend")
        self.spec = spec
        self.name = spec.model_id
        self.url = spec.base_url.rstrip("/") + "/chat/completions"
        self.api_key = os.environ.get(spec.api_key_env, "EMPTY")

    def build_request(self, prompt, images, json_schema, max_new_tokens) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _data_url(im)}} for im in images or []
        ]
        content.append({"type": "text", "text": prompt})
        body: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens or self.spec.max_new_tokens,
        }
        if json_schema is not None and self.spec.guided_json:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }
        return body

    def generate(self, prompt, *, images=None, json_schema=None, max_new_tokens=None) -> str:
        body = self.build_request(prompt, images, json_schema, max_new_tokens)
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.spec.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ExternalToolError(
                f"{self.url} returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ExternalToolError(f"cannot reach {self.url}: {exc.reason}") from exc
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ExternalToolError(f"unexpected response from {self.url}: {str(data)[:300]}") from exc

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "backend": "openai", "base_url": self.spec.base_url,
                "guided_json": self.spec.guided_json}
