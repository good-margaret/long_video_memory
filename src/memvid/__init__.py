"""memvid: budgeted, structured memory for question answering over long videos.

Stage 1  environment, video probing, preprocessing (audio, ASR, chunks, frames)
Stage 2  evaluation set (questions, split, freezing)
Stage 3  perception: per-chunk structured observations from a VLM
Stage 4  memory organisations C1-C4, retrieval, answering
Stage 5  experiments, evaluation, statistics, audits, demo
"""

__version__ = "1.5.1"
