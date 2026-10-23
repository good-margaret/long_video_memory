"""Compare the frozen Whisper transcript with the provided-subtitle condition."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from jiwer import process_words

from memvid.io import read_jsonl, write_json
from memvid.schemas import TranscriptSegment
from memvid.timeutils import merge_intervals


def words(text: str) -> list[str]:
    text = re.sub(r"\b(?:speaker|music)\s*:\s*", " ", text.lower())
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def coverage(segments: list[TranscriptSegment]) -> float:
    return sum(end - start for start, end in merge_intervals([(s.start, s.end) for s in segments]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asr", default="data/experiments/F8_ASR/KbahC-QCKU8/transcript.jsonl"
    )
    parser.add_argument(
        "--srt", default="data/experiments/F8_SRT/KbahC-QCKU8/transcript.jsonl"
    )
    parser.add_argument("--output", default="runs/input_budget_experiment/transcript_comparison.json")
    args = parser.parse_args()

    asr = read_jsonl(args.asr, TranscriptSegment)
    srt = read_jsonl(args.srt, TranscriptSegment)
    hypothesis = words(" ".join(segment.text for segment in asr))
    reference = words(" ".join(segment.text for segment in srt))
    comparison = process_words(" ".join(reference), " ".join(hypothesis))
    result = {
        "note": "SRT is a subtitle reference, not guaranteed verbatim ground truth.",
        "wer_against_srt": round(comparison.wer, 4),
        "substitutions": comparison.substitutions,
        "deletions": comparison.deletions,
        "insertions": comparison.insertions,
        "asr": {
            "segments": len(asr),
            "words": len(hypothesis),
            "coverage_s": round(coverage(asr), 2),
            "most_common_words": Counter(hypothesis).most_common(20),
        },
        "srt": {
            "segments": len(srt),
            "words": len(reference),
            "coverage_s": round(coverage(srt), 2),
            "most_common_words": Counter(reference).most_common(20),
        },
    }
    write_json(Path(args.output), result)
    print(result)


if __name__ == "__main__":
    main()
