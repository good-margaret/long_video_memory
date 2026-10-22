"""Pick a video from LVBench (https://github.com/zai-org/LVBench) that suits the brief, then
download it and prepare its config and a draft question set.

    python scripts/lvbench.py list                  # rank candidates from the LVBench metadata
    python scripts/lvbench.py list --check 10       # also ask YouTube: still online? speech?
    python scripts/lvbench.py fetch <key>           # download + config + draft questions

`list` ranks the 103 LVBench videos by how much material they offer for the brief, using
LVBench's own questions as evidence (the metadata says nothing else about the content):
  ent    questions about entities (entity recognition)       -> objects and people to track
  evt    questions about events (event understanding, temporal grounding)
  long   questions spanning >= 5 min, or summaries            -> relations between parts
  order  questions with before/after/first/last/again/change   -> order and changes over time
  quart  quartiles of the video that questions point into (0-4), early = in the first 10 min
  score  ent + evt + 2*long + 2*order + quart + 2*[early] + category prior
The category prior (selfmedia 3, tv 2, documentary 1, live 1, cartoon 0, sport 0) favours
vlogs and series, where the same people and objects recur and objects change state. The
ranking is a shortlist, not a verdict: watch the candidate before committing to it.
`--check` matters: some LVBench videos are now private or region-locked. "speech=yes" means
YouTube generated automatic captions; "no?" means it did not (often non-English speech).

`fetch` writes, for video id lvb_<key>:
  data/raw/lvb_<key>.mp4                       the video (<= 720p, with audio; needs ffmpeg)
  configs/lvb_<key>.yaml                       configs/default.yaml with this video
  data/annotations/lvb_<key>/questions.yaml    LVBench questions converted to the memvid
                                               format (a draft: types are mapped
                                               automatically and object_change questions
                                               must be written by hand)

Requirements: the memvid package (pip install -e .) and yt-dlp (pip install yt-dlp) for
`--check` and the download; if YouTube refuses the download, update yt-dlp and install
deno (the JavaScript runtime yt-dlp uses for YouTube). LVBench is licensed CC BY-NC-SA 4.0
(academic use only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

from memvid.questions.schema import Question
from memvid.timeutils import format_timestamp, parse_timestamp

ROOT = Path(__file__).resolve().parents[1]
META_URL = "https://huggingface.co/datasets/THUDM/LVBench/resolve/main/video_info.meta.jsonl"
META_PATH = ROOT / "data" / "raw" / "lvbench" / "video_info.meta.jsonl"
PRIOR = {"selfmedia": 3, "tv": 2, "documentary": 1, "live": 1, "cartoon": 0, "sport": 0}
LONG_SPAN_S = 300.0
EARLY_S = 600.0
ORDER_WORDS = re.compile(r"\b(before|after|first|last|again|later|earlier|finally|change[sd]?|changing|"
                         r"turns? into|becomes?|became|anymore|still)\b", re.I)


# ----------------------------------------------------------------------------- metadata
def load_meta(refresh: bool = False) -> list[dict]:
    if refresh or not META_PATH.is_file():
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {META_URL}", file=sys.stderr)
        urllib.request.urlretrieve(META_URL, META_PATH)
    with open(META_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_question(text: str) -> tuple[str, list[str]]:
    """LVBench embeds the options in the question: 'stem\\n(A) ...\\n(B) ...'."""
    parts = re.split(r"\n\(([A-Z])\)\s*", text)
    return parts[0].strip(), [p.strip() for p in parts[2::2]]


def interval(time_reference: str, duration_s: float) -> tuple[float, float] | None:
    """'MM:SS-MM:SS' (minutes may exceed 59) -> seconds, clipped to the video; a point
    reference becomes a 2 s window. None if the reference is missing or outside the video."""
    try:
        a, b = (parse_timestamp(x) for x in time_reference.split("-"))
    except ValueError:
        return None
    a, b = min(a, b), min(max(a, b), duration_s)
    if a >= duration_s:
        return None
    if b - a < 1.0:
        a, b = max(0.0, a - 1.0), min(duration_s, a + 1.0)
    return a, b


def profile(row: dict) -> dict:
    duration = row["video_info"]["duration_minutes"] * 60
    ent = evt = long_ = order = early = 0
    quartiles = set()
    for q in row["qa"]:
        types = set(q["question_type"])
        iv = interval(q["time_reference"], duration)
        ent += "entity recognition" in types
        evt += bool(types & {"event understanding", "temporal grounding"})
        long_ += "summarization" in types or (iv is not None and iv[1] - iv[0] >= LONG_SPAN_S)
        order += bool(ORDER_WORDS.search(split_question(q["question"])[0]))
        if iv is not None:
            quartiles.add(min(3, int(4 * iv[0] / duration)))
            early += iv[0] < EARLY_S
    score = (ent + evt + 2 * long_ + 2 * order + len(quartiles) + 2 * (early > 0)
             + PRIOR.get(row["type"], 0))
    return {"key": row["key"], "type": row["type"], "minutes": duration / 60, "qa": len(row["qa"]),
            "ent": ent, "evt": evt, "long": long_, "order": order, "quart": len(quartiles),
            "early": early, "score": score}


# ----------------------------------------------------------------------------- YouTube
def _yt_dlp():
    try:
        import yt_dlp  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("yt-dlp is not installed: pip install yt-dlp")
    return yt_dlp


class _Silent:
    """yt-dlp logger that drops its messages; failures still arrive as exceptions."""

    def debug(self, _msg: str) -> None:
        pass

    warning = error = debug


def youtube_info(key: str) -> dict:
    """Title, availability and a speech proxy (YouTube made automatic captions)."""
    yt_dlp = _yt_dlp()
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "logger": _Silent()}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={key}", download=False)
    except Exception as exc:  # removed, private, region-locked, ...
        return {"online": False, "error": re.sub(r"^ERROR: \[youtube\] \S+ ", "", str(exc).splitlines()[0])[:70]}
    langs = sorted(info.get("automatic_captions") or {})
    return {"online": True, "title": info.get("title", ""), "minutes_yt": (info.get("duration") or 0) / 60,
            "speech": bool(langs), "language": info.get("language") or (langs[0] if langs else None)}


def download(key: str, out: Path, max_height: int) -> Path:
    yt_dlp = _yt_dlp()
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b",
        "format_sort": ["vcodec:h264", "acodec:m4a"],  # fast, exact ffmpeg seeking; plays everywhere
        "merge_output_format": "mp4",  # merging video and audio needs ffmpeg on PATH
        "outtmpl": str(out.with_suffix("")) + ".%(ext)s",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={key}"])
    if not out.is_file():
        sys.exit(f"download finished but {out} is missing; check the yt-dlp output above")
    return out


# ----------------------------------------------------------------------------- outputs
def write_config(video_id: str, force: bool) -> Path:
    path = ROOT / "configs" / f"{video_id}.yaml"
    if path.exists() and not force:
        print(f"kept existing {path.relative_to(ROOT).as_posix()}")
        return path
    text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    text = re.sub(r"(?m)^(  video_id:\s*)\S+", rf"\g<1>{video_id}", text, count=1)
    text = re.sub(r"(?m)^(  filename:\s*)\S+", rf"\g<1>{video_id}.mp4", text, count=1)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def memvid_type(lvbench_types: list[str], span_s: float) -> str:
    """Best-effort mapping onto the memvid question types (review every one)."""
    types = set(lvbench_types)
    if "summarization" in types or span_s >= LONG_SPAN_S:
        return "cross_segment"
    if types & {"event understanding", "temporal grounding", "reasoning"}:
        return "past_event"
    return "attribute_recall"  # entity recognition, key information retrieval


def convert_questions(row: dict) -> tuple[list[dict], list[str]]:
    duration = row["video_info"]["duration_minutes"] * 60
    out, skipped = [], []
    for q in row["qa"]:
        iv = interval(q["time_reference"], duration)
        stem, options = split_question(q["question"])
        if iv is None:
            skipped.append(f"uid {q['uid']}: time reference '{q['time_reference']}' unusable")
            continue
        rec = {
            "id": f"lvb{q['uid']}", "question": stem, "answer": q["answer"], "answer_type": "multiple_choice",
            "options": options, "type": memvid_type(q["question_type"], iv[1] - iv[0]),
            "gt_intervals": [[format_timestamp(iv[0], decimals=0), format_timestamp(iv[1], decimals=0)]],
            "modality": "both", "notes": f"LVBench uid {q['uid']} ({', '.join(q['question_type'])}); type auto-mapped",
        }
        try:
            Question.model_validate(rec)
        except ValueError as exc:
            skipped.append(f"uid {q['uid']}: {str(exc).splitlines()[-1]}")
            continue
        out.append(rec)
    return out, skipped


def write_questions(row: dict, video_id: str) -> Path:
    qs, skipped = convert_questions(row)
    folder = ROOT / "data" / "annotations" / video_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "questions.yaml"
    if path.exists():
        path = folder / "questions.lvbench.yaml"  # never overwrite hand-written questions
    header = ("# DRAFT converted from LVBench (CC BY-NC-SA 4.0) by scripts/lvbench.py.\n"
              "# Review every question: `type` was mapped automatically and `modality` is unknown.\n"
              "# The brief also needs object_change questions (none in LVBench) and the plan needs\n"
              "# 60-80 questions in total: add your own before `memvid questions-freeze`.\n")
    body = yaml.safe_dump({"video_id": video_id, "annotator": "LVBench (converted; review)", "questions": qs},
                          sort_keys=False, allow_unicode=True, width=100)
    path.write_text(header + body, encoding="utf-8", newline="\n")
    counts: dict[str, int] = {}
    for q in qs:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    print(f"{len(qs)} questions converted {counts}; {len(skipped)} skipped")
    for s in skipped:
        print(f"  skipped {s}")
    return path


# ----------------------------------------------------------------------------- commands
def cmd_list(args) -> None:
    rows = [profile(r) for r in load_meta(args.refresh)]
    rows = [r for r in rows if args.min_minutes <= r["minutes"] <= args.max_minutes
            and (not args.types or r["type"] in args.types)]
    rows.sort(key=lambda r: (-r["score"], r["minutes"]))
    rows = rows[: args.top]
    head = f"{'#':>2} {'key':<12} {'type':<11} {'min':>4} {'qa':>3} {'ent':>3} {'evt':>3} {'long':>4} " \
           f"{'order':>5} {'quart':>5} {'early':>5} {'score':>5}"
    print(head + ("  youtube" if args.check else ""))
    for i, r in enumerate(rows, 1):
        line = (f"{i:>2} {r['key']:<12} {r['type']:<11} {r['minutes']:>4.0f} {r['qa']:>3} {r['ent']:>3} "
                f"{r['evt']:>3} {r['long']:>4} {r['order']:>5} {r['quart']:>5} {r['early']:>5} {r['score']:>5}")
        if args.check and i <= args.check:
            y = youtube_info(r["key"])
            line += (f"  online, speech={'yes' if y['speech'] else 'no?'} ({y['language']}), "
                     f"{y['minutes_yt']:.0f} min: {y['title'][:50]}" if y["online"] else f"  OFFLINE: {y['error']}")
        print(line)
    print("\nwatch a candidate: https://www.youtube.com/watch?v=<key>; then: python scripts/lvbench.py fetch <key>")


def cmd_fetch(args) -> None:
    row = next((r for r in load_meta() if r["key"] == args.key), None)
    if row is None:
        sys.exit(f"{args.key} is not an LVBench video key (see `python scripts/lvbench.py list`)")
    video_id = f"lvb_{args.key}"  # YouTube ids may start with '-', which memvid ids may not
    if not args.no_download:
        video = ROOT / "data" / "raw" / f"{video_id}.mp4"
        if video.is_file():
            print(f"kept existing {video.relative_to(ROOT).as_posix()}")
        else:
            download(args.key, video, args.max_height)
    cfg = write_config(video_id, args.force)
    questions = write_questions(row, video_id)
    c = cfg.relative_to(ROOT).as_posix()
    print(f"\nconfig:    {c}\nquestions: {questions.relative_to(ROOT).as_posix()}\n"
          f"next:      memvid preprocess -c {c}   (then see README, section 3)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Choose and fetch an LVBench video for memvid")
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="rank LVBench videos for the brief")
    ls.add_argument("--min-minutes", type=float, default=20.0, help="the brief asks for 20-60+ min")
    ls.add_argument("--max-minutes", type=float, default=70.0,
                    help="perception cost grows with length (about 180 VLM calls per hour)")
    ls.add_argument("--types", nargs="*", choices=sorted(PRIOR), help="only these categories")
    ls.add_argument("--top", type=int, default=15)
    ls.add_argument("--check", type=int, default=0, metavar="N",
                    help="ask YouTube about the top N: online? speech (automatic captions)? title")
    ls.add_argument("--refresh", action="store_true", help="download the metadata again")
    ls.set_defaults(func=cmd_list)
    fe = sub.add_parser("fetch", help="download a video, write its config and draft questions")
    fe.add_argument("key", help="LVBench key (the YouTube id)")
    fe.add_argument("--max-height", type=int, default=720, help="frames are resized to 768 px anyway")
    fe.add_argument("--no-download", action="store_true", help="only write the config and questions")
    fe.add_argument("--force", action="store_true", help="overwrite an existing config")
    fe.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
