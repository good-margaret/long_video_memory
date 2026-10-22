"""Timestamps (parsing, formatting) and time-interval arithmetic.

The question set is written by hand, so timestamps accept "hh:mm:ss(.f)", "mm:ss(.f)"
or plain seconds.
"""

from __future__ import annotations


def parse_timestamp(value: str | int | float) -> float:
    """Convert a timestamp to seconds.

    Accepted forms: 75, 75.5, "75.5", "01:15", "01:15.5", "1:01:15", "1:01:15.25".

    >>> parse_timestamp("1:02:03.5")
    3723.5
    >>> parse_timestamp("02:03")
    123.0
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError("Boolean is not a valid timestamp")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Empty timestamp")
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid timestamp '{value}': too many ':' separators")
        try:
            numbers = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp '{value}'") from exc
        if any(n < 0 for n in numbers):
            raise ValueError(f"Invalid timestamp '{value}': negative component")
        if len(numbers) > 1 and any(n >= 60 for n in numbers[1:]):
            raise ValueError(f"Invalid timestamp '{value}': minutes/seconds must be < 60")
        seconds = 0.0
        for n in numbers:
            seconds = seconds * 60.0 + n
    if seconds < 0:
        raise ValueError(f"Negative timestamp: {value}")
    return seconds


def format_timestamp(seconds: float, *, decimals: int = 1) -> str:
    """Format seconds as hh:mm:ss.f (hours always shown).

    >>> format_timestamp(3723.5)
    '01:02:03.5'
    """
    if seconds < 0:
        raise ValueError("Negative duration")
    scale = 10**decimals
    total = round(seconds * scale)
    hours, rem = divmod(total, 3600 * scale)
    minutes, rem = divmod(rem, 60 * scale)
    secs = rem / scale
    width = 2 + (decimals + 1 if decimals > 0 else 0)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:0{width}.{decimals}f}"


Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Union of intervals as a sorted list of disjoint intervals."""
    out: list[Interval] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def total_length(intervals: list[Interval]) -> float:
    return sum(b - a for a, b in merge_intervals(intervals))


def overlap_length(x: list[Interval], y: list[Interval]) -> float:
    """Length of the intersection of two unions of intervals."""
    return sum(max(0.0, min(b, d) - max(a, c))
               for a, b in merge_intervals(x) for c, d in merge_intervals(y))


def tiou(x: list[Interval], y: list[Interval]) -> float:
    inter = overlap_length(x, y)
    union = total_length(x) + total_length(y) - inter
    return inter / union if union > 0 else 0.0
