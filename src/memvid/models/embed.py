"""Text embeddings and token counting.

Embeddings are L2-normalised, so the inner product is the cosine similarity.
* SentenceTransformerEmbedder: the real one (optional extra `memory`).
* HashingEmbedder: deterministic feature hashing of words and character trigrams; no
  model, no download. Used by the smoke test and where sentence-transformers is unavailable. Its
  similarities are lexical, so thresholds tuned for one embedder do not transfer to the
  other; runs record which embedder was used.

Token counting defines the memory budget (report, Eq. budget):
|M|_tau = sum over records of the token count of the record's serialised line.
* HFTokenCounter: tokenizer of the answering model (the honest unit).
* ApproxTokenCounter: deterministic estimate (one token per 4 characters of each word,
  one per punctuation mark), for environments without the tokenizer.
"""

from __future__ import annotations

import math
import re
import zlib
from functools import lru_cache

import numpy as np

from memvid.errors import MissingDependencyError

_WORD = re.compile(r"\w+|[^\w\s]")


class Embedder:
    name = "embedder"
    dim = 0

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, rows unit norm
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 31) & 1 == 0 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError(
                "sentence-transformers is not installed: pip install -e '.[memory]', "
                "or set models.embedder: hashing"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
            for t, v in zip(missing, vecs, strict=True):
                self._cache[t] = v.astype(np.float32)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])


class TokenCounter:
    name = "tokens"

    def count(self, text: str) -> int:
        raise NotImplementedError


class ApproxTokenCounter(TokenCounter):
    name = "approx"

    def count(self, text: str) -> int:
        return _approx(text)


@lru_cache(maxsize=200_000)
def _approx(text: str) -> int:
    return sum(max(1, math.ceil(len(p) / 4)) for p in _WORD.findall(text))


class HFTokenCounter(TokenCounter):
    def __init__(self, tokenizer, name: str):
        self.tokenizer = tokenizer
        self.name = name
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        n = self._cache.get(text)
        if n is None:
            n = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(self._cache) < 500_000:
                self._cache[text] = n
        return n


def truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Longest word prefix of `text` with at most max_tokens tokens (binary search)."""
    if counter.count(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])
