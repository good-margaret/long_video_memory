"""Answer scoring.

Order of methods per answer type (the first that applies decides):
* number         : parse a number (digits or English words up to twenty) from both answers;
                   correct iff |pred - ref| <= rel_tol * max(1, |ref|).
* yes_no         : first yes/no token of the prediction equals the reference.
* multiple_choice: letter of the prediction (or the text of an option) equals the reference.
* short_text     : normalised exact match against the answer or any accepted answer, or
                   every content word of a reference contained in the prediction; otherwise
                   the LLM judge if enabled (its agreement with manual scoring is reported).
"""

from __future__ import annotations

import re
import string
from typing import Literal

from pydantic import BaseModel

from memvid.models.base import GenerativeModel
from memvid.textutils import content_words

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "zero": 0, "none": 0, "no": 0, "one": 1, "a single": 1, "single": 1, "two": 2, "a pair": 2,
    "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "a dozen": 12, "dozen": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_NEGATION = {"not", "no", "never", "neither", "nor", "isnt", "wasnt", "didnt", "doesnt"}

Method = Literal["number", "yes_no", "choice", "exact", "containment", "judge", "none"]


class Score(BaseModel):
    correct: bool
    method: Method


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    words = [w for w in text.replace("-", " ").split() if w not in _ARTICLES]
    return " ".join(words)


_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b")


def parse_number(text: str) -> float | None:
    """First number in the text, ignoring timestamps such as 00:12 or 1:02:03."""
    t = _TIMESTAMP.sub(" ", text.lower().replace(",", ""))
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if m:
        return float(m.group(0))
    for phrase in sorted(_NUMBER_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            return float(_NUMBER_WORDS[phrase])
    return None


def parse_yes_no(text: str) -> str | None:
    m = re.search(r"\b(yes|no|true|false)\b", text.lower())
    if not m:
        return None
    return "yes" if m.group(1) in ("yes", "true") else "no"


def parse_choice(text: str, options: list[str]) -> str | None:
    """Option letter of the answer. Option text is matched first, so an answer such as
    "a drill" is not read as letter A; a letter counts only when it stands alone
    ("B", "(b)", "B)", "B.", "B: ...")."""
    t = text.strip()
    norm = normalize_text(t)
    for i, opt in enumerate(options):
        if normalize_text(opt) == norm:
            return chr(65 + i)
    m = re.match(r"^\(?([A-Za-z])(?:\)|\.|:|$)", t) or re.match(r"^\(([A-Za-z])\)", t)
    if m and 0 <= ord(m.group(1).upper()) - 65 < len(options):
        return m.group(1).upper()
    matches = [i for i, opt in enumerate(options) if normalize_text(opt) and normalize_text(opt) in norm]
    if len(matches) == 1:  # exactly one option's text appears in a longer answer
        return chr(65 + matches[0])
    return None


def _content(text: str) -> set[str]:
    return set(content_words(normalize_text(text)))


def score_answer(
    pred: str,
    reference: str,
    answer_type: str,
    *,
    accepted: list[str] | None = None,
    options: list[str] | None = None,
    rel_tol: float = 0.05,
    judge: GenerativeModel | None = None,
    question: str = "",
) -> Score:
    pred = pred or ""
    if answer_type == "number":
        p, r = parse_number(pred), parse_number(reference)
        if p is None or r is None:
            return Score(correct=False, method="number")
        return Score(correct=abs(p - r) <= rel_tol * max(1.0, abs(r)), method="number")
    if answer_type == "yes_no":
        ref = parse_yes_no(reference)
        return Score(correct=ref is not None and parse_yes_no(pred) == ref, method="yes_no")
    if answer_type == "multiple_choice":
        return Score(correct=parse_choice(pred, options or []) == reference.strip().upper(),
                     method="choice")
    refs = [reference, *(accepted or [])]
    np_ = normalize_text(pred)
    if any(np_ == normalize_text(r) for r in refs):
        return Score(correct=True, method="exact")
    pc = _content(pred)
    negated = bool(_NEGATION & set(normalize_text(pred).split()))
    for r in refs:
        rc = _content(r)
        # containment: the reference is in the answer and the answer is not a list of
        # alternatives or a negation ("not raw wood, white" must not count)
        if rc and rc <= pc and not negated and len(pc - rc) <= max(2, len(rc)):
            return Score(correct=True, method="containment")
    if judge is not None:
        from memvid.models.jsonutil import generate_validated
        from memvid.prompts import JUDGE

        class _Verdict(BaseModel):
            correct: bool

        res = generate_validated(
            judge, JUDGE.format(question=question, reference=" | ".join(refs), candidate=pred),
            _Verdict, max_new_tokens=32,
        )
        return Score(correct=bool(res.ok and res.value.correct), method="judge")
    return Score(correct=False, method="none")
