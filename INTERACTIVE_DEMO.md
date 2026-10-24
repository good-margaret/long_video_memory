# Interactive memory notebook

Clone the demo branch or import its notebook into Kaggle:

```bash
git clone --branch codex/interactive-notebook-demo --single-branch https://github.com/good-margaret/long_video_memory.git
cd long_video_memory
```

Open `notebooks/interactive_memory_demo.ipynb`. Its first cell also clones this
branch automatically when running outside a checkout, then installs dependencies.
The setup cell discovers working Unix FFmpeg/FFprobe or downloads managed binaries
into `.cache/media-tools/`. Windows uses managed tools to avoid incompatible
Anaconda executables. `MEMVID_FFMPEG` and `MEMVID_FFPROBE` are configured automatically;
no author-specific paths, administrator installation or global PATH edits are required.
The notebook pins Transformers 5.2.0, the version used for the local model check;
it does not change the research package's broader model dependency requirement.

The notebook uses ipywidgets 8, not a public Gradio server. It prepares a browser
compatible H264/AAC copy, transcribes it with Whisper, samples eight frames per
20-second chunk, perceives observations, and builds only C3 with an explicit
token budget. Original files are never overwritten. Positive audio offset delays
the separate soundtrack. Missing trailing audio is padded with silence.

## Optional YouTube input

Set `DOWNLOAD_YOUTUBE = True` and edit `YOUTUBE_URL`. The downloader saves a video
with audio under `data/youtube/raw/`; the default cap is 720p. Filenames include
the video ID and quality cap, so changing the cap cannot silently reuse a lower
resolution. `.source.json` stores source URL, resolution cap, actual height,
duration and downloader version. Verified matching downloads can be reused offline.
Set `DOWNLOAD_YOUTUBE = False` to provide local or `/kaggle/input/...` files instead.

The notebook extra includes `yt-dlp[default]` and its matching EJS component.
Supported Node/Deno is discovered automatically; official portable Deno is
downloaded into `.cache/js-runtime/` if necessary. Current YouTube extraction
requires this runtime: see the [official yt-dlp setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
YouTube may require authentication or block Kaggle IPs. In that case upload media
downloaded on your own computer. Browser cookies and credentials are never read automatically.

## Record ID legend

| Prefix | Type | Meaning |
| --- | --- | --- |
| E | entity | Person, object, animal, place, attributes and aliases |
| V | event | Visual event, speech-summary event, or compressed event summary |
| S | state_change | Confirmed old-to-new attribute transition |
| R | relation | Subject–predicate–object fact referring to entity IDs |
| N | note | Free-text summary/log entry, mainly in C1/C2/C4 |

V does not mean visual-only. S does not mean speech: speech records are events
with modality=speech and therefore use V IDs. The number is a per-type creation
counter within a run, not time or a chunk index. Updating a record can keep its ID.

Profiles:

| Profile | Vision | Text / memory / QA | ASR | Execution |
| --- | --- | --- | --- | --- |
| local_small | SmolVLM-500M-Instruct | Qwen3-0.6B | tiny.en / int8 | CPU / float32; first 180 seconds |
| kaggle | Qwen3-VL-4B-Instruct | same checkpoint | large-v3-turbo / float16 | two T4 / float16; set duration to None |

These are different model conditions, not accuracy-equivalent runs. The local
profile tests integration. Kaggle validates the full video; accuracy still needs
separate evaluation against the fixed question dataset. First use downloads model
weights. Set HF_HOME to control their cache location.

SmolVLM-500M is used for plain visual descriptions, not the complex observation
JSON task that failed in the initial real-model check. A separate Qwen3-0.6B
stage converts those captions and Whisper text into the existing observation
schema. Captions are saved and resumable in visual_captions.jsonl. Kaggle's
Qwen3-VL profile perceives frames and speech directly. No human-written captions
or rule-generated observations substitute for model output.

The local SmolVLM processor explicitly disables image splitting and caps its
resize at 384 pixels, rather than silently upscaling each frame to 2048 pixels.
Both overrides are stored in ModelSpec and the session YAML. CPU processing of
4K AV1 inputs can still be slow; preparation prefers libdav1d when available and
otherwise uses the software libaom-av1 decoder.

## Inspecting evidence

Select a chunk to reconstruct memory from SQLite's operation journal. Answers
retrieve from that snapshot only. The source selector opens clips for any visible
record. Answer clips use only valid cited IDs, retain separated source intervals,
and do not pad past the selected memory time. Scores, token usage, citations and
generation errors remain visible. Clips show the provenance of a memory record;
they do not certify that the model's interpretation is correct.

Preview is limited to ten seconds. Long supporting spans are split into pieces
of at most 30 seconds, retaining their entire coverage. The clip selector encodes
and embeds only the selected piece; it does not embed a 40-minute source video
into notebook output. Programmatic ask() returns lazy clip plans: use
materialize_clip(item) to create a file. clips(..., lazy=False) exports immediately.

Memory timestamps refer to prepared.mp4; clip labels also show the corresponding
time in the original input. An insufficient answer is not presented as grounded;
invalid citations and malformed output produce explicit statuses. Failed visual
perception blocks memory building, including the upstream transcript-only fallback.
Transcript-floor and rescue provenance remain in derived/observations.jsonl.

## Saving and continuing

Each session is fingerprinted by input paths, file sizes and modification times,
range, profile and budget. This is a cache identity, not a cryptographic hash of
the input media contents. Keep the complete session directory. It includes media,
YAML configuration, transcript, frames, observations, memory.sqlite, processing.log,
stage markers, question history and clips. Build stages run in separate processes
so model allocations are released between stages. Completed stages are skipped;
perception resumes from the existing partial observations. An interrupted memory
stage rebuilds C3 from saved observations, rather than reloading partial internal
entity-resolution state. Interrupting the notebook terminates the active worker.

`NotebookDemo.open(session_dir)` opens saved memory without original inputs or
preprocessing. It still needs prepared.mp4 for clips and the configured model
weights for new answers. Opening a copied session relocates its media and artifact
paths in memory; the saved YAML remains an audit of the original run.

For Kaggle clone this branch, install the notebook dependencies and FFmpeg, enable
Internet and T4 ×2, select profile kaggle and an output path in /kaggle/working.
If widgets are not visible, enable the notebook frontend's widget support; ordinary
tables alone do not prove callbacks or video playback work.

The portable project ZIP is an alternative to cloning the branch: extract its
long_video_memory folder under /kaggle/working, then import the notebook into
Kaggle. Its first cell recognizes that location. Supply the video and optional
audio through /kaggle/input. A separately supplied demo_session ZIP can be
extracted and opened with NotebookDemo.open() to inspect the completed local run.

## Local integration result

The first 180 seconds were processed with real Whisper tiny.en, SmolVLM-500M,
Qwen3-0.6B and MiniLM embeddings: 9 chunks, 72 frames, 18 final C3 records,
663 tokens out of a 2000-token budget. Asking where they are travelling returned
Arctic with citation V1 and a verified 20-second H264/AAC clip. Saved-memory reload
and prefix isolation were checked. No heuristic backend was used.

This is not an accuracy result. The tiny models repeated descriptions, invented
some attributes and omitted some visual details or relationships. Raw captions
and structured observations remain visible for inspection. The full-video,
large-model Kaggle run and manual frontend verification there are still pending.

## Validation

Run `python -m pytest tests/test_notebook_demo.py tests/test_notebook_bootstrap.py -q` for temporal isolation,
invalid citations, empty memory, disjoint evidence, clipping at the visible
boundary and both model profiles. Media and real-model checks are additional;
unit tests deliberately do not download weights or stand in for model accuracy.
Bootstrap tests cover managed Windows tools, validated Unix tools, safe archive
extraction, JS runtime fallback, merged-audio checks, download caching and English UI text.

## Recovering failed perception chunks

If perception completes but the demo rejects some chunk IDs, all observations and
the partial checkpoint are already saved. Update this branch with `git pull --ff-only`
and call `demo.build()` again in the same kernel with unchanged settings. Successful
cached chunks are retained; only failed or transcript-only fallback chunks are
retried. Worker processes import the updated code without restarting the kernel.

Selective retries use the same frames, image resolution and transcript, with a
compact JSON prompt and an output limit of at least 2048 tokens. This recovery
strategy and retried IDs are recorded in perception_info.json. Before retrying,
small observation/manifest files are backed up under derived/demo/recovery_backups/.
The actual derived root is the session's derived directory; media is not duplicated.
An unmatched or missing checkpoint causes an explicit error rather than a full-video
restart. Do not use --force or edit the model/frame configuration to recover an existing run.

If a retry still fails, its failed status is retained and memory construction stays
blocked. Subsequent retries retain newly successful chunks as well. This does not
silently replace missing visual observations with transcript-only records.
