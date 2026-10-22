# memvid: Memory for Long Videos

Research code for a course project on **budgeted, structured memory for question
answering over long (20–60+ min) videos**. A vision-language model (VLM) observes the
video chunk by chunk; a memory module stores, merges, updates, compresses and forgets
information under a fixed token budget; questions are answered from the memory alone,
with citations that point back to the source clips.

The repository contains the complete pipeline, an interactive demo and a LaTeX report
whose tables and figures are generated from the results.

| Stage | What it does | Main commands |
|---|---|---|
| 1 | Environment check, video probing, preprocessing (audio, ASR, chunks, frames) | `check-env` `probe` `preprocess` `freeze` |
| 2 | Evaluation set: questions, dev/test split, freezing | `questions-validate` `questions-split` `questions-freeze` |
| 3 | Perception: one structured observation per chunk from a VLM, audited | `perceive` `perception-audit` `perception-freeze` |
| 4 | Memory organisations C1–C4, retrieval, answering | `budgets` `build` `answer` `runs` |
| 5 | All runs, scoring, failure attribution, audits, figures, demo | `run-all` `analyze` `audit-*` `attribution` `demo` |

The research questions (mapped to the project brief) and the method of every stage are in
[`report/`](report/) (`cd report && latexmk -pdf main.tex` builds `main.pdf`).

---

## 1. Installation

### 1.1 System requirements

* Python **3.10+**
* **ffmpeg** and **ffprobe** on `PATH` (`sudo apt install ffmpeg`, `brew install ffmpeg`)
* For real runs: an NVIDIA GPU with about 16 GB or more for Qwen3-VL-4B in bf16 (an
  estimate: about 9 GB of weights plus activations for 8 frames per call); with less
  memory use Qwen3-VL-2B (set `models.*.model_id`).
* LaTeX (`latexmk`, `pdflatex`) for the report.

### 1.2 Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1. PyTorch with CUDA first (pick the command for your CUDA version at pytorch.org), e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Everything else
pip install -e ".[all]"
```

| Extra | Packages | Needed for |
|---|---|---|
| *(core)* | pydantic, PyYAML, NumPy, Pillow, tqdm | all stages with the stand-in models (smoke test) |
| `asr` | faster-whisper | `transcribe` |
| `models` | torch, transformers ≥ 4.57, accelerate | local Qwen3-VL (perception, LLM roles) |
| `memory` | sentence-transformers | embeddings for entity resolution and retrieval |
| `analysis` | matplotlib, pandas | figures, `notebooks/analysis.ipynb` |
| `demo` | gradio, matplotlib | `memvid demo` |
| `dev` | ruff | linting |
| `all` | all of the above | a full run |

### 1.3 Check

```bash
memvid check-env     # versions, GPU; saves runs/environment.json
```
Then run the smoke test below: it exercises every stage without a GPU.

---

## 2. Five-minute smoke test (no GPU, no real video)

Runs **every stage** on a synthetic video with a deterministic rule-based stand-in
for the models (`backend: heuristic`). It tests the plumbing only; its numbers mean
nothing and every generated table says so.

```bash
C=configs/smoke_test.yaml
memvid make-test-video --out data/raw/smoke_test.mp4 --duration 90
memvid preprocess -c $C --skip-asr
memvid questions-split -c $C && memvid questions-freeze -c $C
memvid perceive -c $C && memvid perception-freeze -c $C
memvid run-all -c $C
memvid analyze -c $C
memvid demo -c $C                  # needs the demo extra
```

---

## 3. Full workflow on the real video

All commands take `-c <config>` (default `configs/default.yaml`) and are run from the
project root. Each step checks that its inputs exist and, where it matters, that they
are frozen. `memvid status` shows where you are.

### Stage 1: data

1. Choose a video of 20–60+ minutes with speech, objects or people that reappear in
   distant parts, and objects whose state changes. Put it in `data/raw/`; set
   `video.video_id` and `video.filename` in `configs/default.yaml`.

   **Or take one from [LVBench](https://github.com/zai-org/LVBench)** (`pip install yt-dlp`):
   ```bash
   python scripts/lvbench.py list --check 10   # ranked shortlist; is it still on YouTube? speech?
   python scripts/lvbench.py fetch <key>       # video, configs/lvb_<key>.yaml, draft questions
   ```
   Watch the candidate before committing to it, and use `-c configs/lvb_<key>.yaml` below.
   The draft `questions.yaml` holds LVBench's ~15 multiple-choice questions with their time
   references; review their types and add your own (object changes above all) to reach 60–80.
2. `memvid preprocess` (probe, audio, Whisper transcript, 20 s chunks, 8 frames per
   chunk; warns if the video is shorter than 20 minutes).
3. `memvid freeze`.

### Stage 2: questions

1. Write 60–80 questions in `data/annotations/<id>/questions.yaml`
   (template: `templates/questions.yaml`) **before building any memory**. The brief
   requires `past_event`, `object_change` and `cross_segment` questions; `attribute_recall`
   and `speech` are optional additions.
2. `memvid questions-validate` until there are no errors (read the warnings).
3. `memvid questions-split` (≈15 dev questions, stratified by type).
4. `memvid questions-freeze`: from now on any change to the questions or the split is detected.

### Stage 3: perception

1. `memvid perceive` (resumable; `--limit 10` for a first look). Check
   `data/derived/<id>/perception_info.json` for parse failures.
2. `memvid perception-audit`, label the 20 sampled chunks (contact sheets provided),
   `memvid perception-audit-summary`.
3. **Gate:** false-entity rate < 15 % → `memvid perception-freeze`.

### Stages 4–5: memory, experiments, analysis

1. Tune on the dev split if needed (thresholds in `memory.resolution`, weights,
   prompts), e.g. `memvid build --condition C3 --budget B2` then
   `memvid answer --run runs/<id>/C3_B2 --split dev`. Each run stores a signature of
   exactly the inputs it depends on, so a configuration change rebuilds only the affected
   runs and answers (tuning C3 never rebuilds C1, C2, C4 or the budgets); nothing stale
   is reused. Then leave the config alone.
2. `memvid run-all` builds C4, resolves budgets B1–B3, and builds and answers C1–C3 at
   every budget plus the four C3 ablations (14 runs) on the test split.
3. `memvid analyze` scores all runs, builds the failure-attribution sheet, draws figures
   and writes `report/generated/results_*.tex`. The analysis is descriptive (95 %
   intervals, paired McNemar comparisons).
4. Manual labels (each `make` command writes a CSV in `runs/<id>/audits/`):
   * `memvid audit-merges` → fill `correct` → `memvid audit-merges-summary`
   * `memvid audit-states` → fill `correct` → `memvid audit-states-summary`
   * `memvid audit-judge` → fill `manual_correct` → `memvid audit-judge-summary`
   * `runs/<id>/audits/attribution.csv`: confirm `proposed_cause` in `final_cause`,
     mark `perception_has_evidence = n` where the observations never had the evidence.
5. `memvid analyze` again (labels are kept and used), then
   `cd report && latexmk -pdf main.tex`.
6. `memvid demo`: inspect memory over time, ask questions, open supporting clips.

### Using vLLM instead of local transformers

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --max-model-len 32768 --limit-mm-per-prompt '{"image": 8}'
```
(the `--limit-mm-per-prompt` syntax differs between vLLM versions; see its docs),
then in the config set, for `models.vlm` and `models.llm`:
`backend: openai`, `base_url: http://localhost:8000/v1`, `guided_json: true`.
Requests carry the JSON schema, so decoding is constrained to it and parse failures
should largely disappear (check `parse_failures` in `perception_info.json`).

---

## 4. Project structure

```
memvid/
├── pyproject.toml              dependencies and extras, tool config
├── configs/
│   ├── default.yaml            the experiment (all stages)
│   └── smoke_test.yaml         synthetic video + heuristic models
├── templates/questions.yaml    question file template
├── scripts/lvbench.py          pick, download and prepare a video from LVBench
├── src/memvid/
│   ├── cli.py                  `memvid` entry point: one small function per command
│   ├── config.py               typed configuration (unknown keys are errors)
│   ├── prompts.py              every prompt, versioned
│   ├── textutils.py            stopwords, content words (shared)
│   ├── timeutils.py            timestamps and interval arithmetic (shared)
│   ├── pipeline.py, video/, preprocess/      Stage 1: probe, audio, ASR, chunks, frames
│   ├── questions/                            Stage 2: schema, split, freeze
│   ├── perception/                           Stage 3: observation schema, VLM run, audit
│   ├── memory/                               Stage 4
│   │   ├── records.py          record types, one-line serialisation, provenance overlap
│   │   ├── store.py            event-sourced store, token count, vector index, replay
│   │   ├── base.py             interface shared by all memory organisations
│   │   ├── baselines.py        C1 rolling summary, C2/C4 flat log
│   │   ├── structured.py       C3 structured memory
│   │   ├── systems.py          conditions and ablations → memory objects
│   │   ├── retrieval.py        hybrid retrieval under the allowance B_R
│   │   ├── answer.py           answering with citations → video time
│   │   └── persist.py          SQLite run files
│   ├── experiment.py           budgets, runs, answering, staleness   Stage 5
│   ├── evaluation/             scoring, metrics, audits, attribution, stats, plots, analysis
│   ├── models/                 transformers / OpenAI-compatible / heuristic backends, embeddings
│   ├── demo/                   demo logic (backend.py) and Gradio UI (app.py)
│   └── reporting/              LaTeX tables
├── notebooks/analysis.ipynb    exploring the results
├── data/raw/                   videos (git-ignored)
├── data/derived/<id>/          machine-generated, hashed in manifest.json (git-ignored)
├── data/annotations/<id>/      human-written: questions, split (committed)
├── runs/<id>/                  memory runs, answers, audits, analysis (git-ignored)
└── report/                     LaTeX report (main.pdf is built, not committed); generated/ and
                                figures/ are written by memvid
```

---

## 5. Outputs

| Path | Content |
|---|---|
| `data/derived/<id>/` | Stage 1 artifacts, `observations.jsonl`, `perception_info.json`, `manifest.json` (hashes, statistics, frozen stages) |
| `data/annotations/<id>/split.json` | dev/test assignment |
| `runs/<id>/budgets.json` | budgets in tokens and the C4 size they derive from |
| `runs/<id>/<cond>_<budget>/memory.sqlite` | final records, operation log, entity-resolution log, size trace, chunk bounds |
| `runs/<id>/<cond>_<budget>/answers_<split>.jsonl` | answers, citations, retrieved records, intervals |
| `runs/<id>/<cond>_<budget>/eval_<split>.jsonl` | per-question scores, grounding, survival |
| `runs/<id>/audits/` | audit sheets and their summaries, attribution sheet |
| `runs/<id>/analysis/` | `summary.json`, figures |

The SQLite run file can be inspected directly, e.g.
`sqlite3 runs/<id>/C3_B2/memory.sqlite "select chunk_id,label,record_id,reason from ops where label='compress'"`.

---

## 6. Design decisions (details in the report)

* **Controlled comparison.** All conditions read the same frozen observations, use the
  same budget unit (answerer tokens of the serialised memory), the same retrieval
  allowance and the same answerer.
* **Memory-agnostic perception.** The VLM never sees the memory; linking observations
  is the memory's job.
* **Event sourcing and provenance.** Every mutation is logged; replaying the log
  reproduces the memory at any chunk. Every record keeps the chunks it came from,
  through merges and compression, so grounding and survival are measurable.
* **Structured memory (C3).** Name-similarity entity resolution with an LLM-adjudicated
  ambiguous band; debounced state changes (two agreeing observations); event
  deduplication; window compression by the LLM, then priority-based deletion. Entities
  and state changes are never compressed.
* **Honest evaluation.** Paired tests, bootstrap intervals, manual audits of every
  subjective step, and a failure attribution to the first stage that lost the evidence.

---

## 7. Development

```bash
ruff check src scripts
```

The heuristic backend lets the whole pipeline run without a GPU: the smoke test (section 2)
is the end-to-end check after a change.

## License

MIT (code). Videos are not distributed; use footage you are allowed to use.
