"""Local Hugging Face models (optional extra: pip install -e '.[models]').

Vision-language models (e.g. Qwen3-VL) are loaded with AutoModelForImageTextToText and
their AutoProcessor; text-only models fall back to AutoModelForCausalLM + AutoTokenizer.
Decoding is greedy (do_sample=False) so outputs are reproducible for fixed inputs.
Images are resized by the caller (perception.max_pixels), so the visual-token budget does
not depend on processor defaults, which differ between model versions.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from memvid.config import ModelSpec
from memvid.errors import MissingDependencyError
from memvid.models.base import GenerativeModel

log = logging.getLogger(__name__)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _import():
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError(
            "torch/transformers are not installed. Install the models extra: "
            "pip install -e '.[models]' (see README for the CUDA build of torch)."
        ) from exc
    return torch, transformers


def _dtype_kwargs(torch, spec: ModelSpec) -> dict[str, Any]:
    if spec.dtype == "auto":
        return {"dtype": "auto"}
    return {"dtype": getattr(torch, spec.dtype)}


class TransformersModel(GenerativeModel):
    def __init__(self, spec: ModelSpec):
        torch, tf = _import()
        self.spec = spec
        self.name = spec.model_id
        self._torch = torch
        t0 = time.perf_counter()
        kwargs = {**_dtype_kwargs(torch, spec), "device_map": spec.device_map}
        self.is_vlm = True
        try:
            self.model = tf.AutoModelForImageTextToText.from_pretrained(spec.model_id, **kwargs)
            self.processor = tf.AutoProcessor.from_pretrained(spec.model_id)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        except (ValueError, KeyError, OSError) as exc:
            if isinstance(exc, OSError) and "does not appear to have" not in str(exc):
                raise
            log.info("%s is not an image-text model (%s); loading as causal LM",
                     spec.model_id, type(exc).__name__)
            self.is_vlm = False
            self.model = tf.AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
            self.processor = tf.AutoTokenizer.from_pretrained(spec.model_id)
            self.tokenizer = self.processor
        self.model.eval()
        log.info("loaded %s in %.1fs (vlm=%s)", spec.model_id, time.perf_counter() - t0, self.is_vlm)

    def generate(self, prompt, *, images=None, json_schema=None, max_new_tokens=None) -> str:
        images = images or []
        if images and not self.is_vlm:
            raise ValueError(f"{self.name} is text-only but images were given")
        if self.is_vlm:
            content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        n = max_new_tokens or self.spec.max_new_tokens
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=n, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return _THINK_RE.sub("", text).strip()

    def describe(self) -> dict[str, Any]:
        tf_version = __import__("transformers").__version__
        return {"name": self.name, "backend": "transformers", "vlm": self.is_vlm,
                "transformers": tf_version, "dtype": self.spec.dtype}


def load_hf_tokenizer(model_id: str):
    _, tf = _import()
    return tf.AutoTokenizer.from_pretrained(model_id)

