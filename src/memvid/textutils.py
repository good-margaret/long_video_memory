"""Word-level text helpers shared by retrieval, memory, scoring and the test models."""

from __future__ import annotations

import re

STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "are", "was", "were",
    "be", "it", "its", "this", "that", "with", "into", "onto", "from", "for", "by", "as",
    "his", "her", "their", "he", "she", "they", "has", "have", "had", "then", "there",
    "what", "which", "who", "when", "where", "how", "did", "does", "do", "many", "much",
    "after", "before", "video",
})
_WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> list[str]:
    """Lower-cased words without stopwords, in order of appearance."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def adds_information(new: str, old: str) -> bool:
    """True if `new` contains a content word that `old` lacks."""
    return bool(set(content_words(new)) - set(content_words(old)))
