"""Gradio UI (optional extra `demo`). Start with `memvid demo -c configs/default.yaml`."""

from __future__ import annotations

from typing import Any

from memvid.config import ProjectConfig
from memvid.demo import backend as B
from memvid.errors import MissingDependencyError
from memvid.pipeline import video_duration
from memvid.timeutils import format_timestamp


def build_app(cfg: ProjectConfig):
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingDependencyError("gradio is not installed: pip install -e '.[demo]'") from exc

    runs = B.runs(cfg)
    if not runs:
        raise RuntimeError("no runs found: build memories first (memvid run-all / memvid build)")
    choices = [r["run"] for r in runs]
    paths = {r["run"]: r["path"] for r in runs}
    duration = video_duration(cfg)
    default = next((c for c in choices if c.startswith("C3_")), choices[0])

    def on_chunk(run: str, chunk: int):
        groups = B.memory_at(paths[run], int(chunk))
        tables = [B.record_rows(groups.get(k, [])) for k in
                  ("entity", "event", "state_change", "relation", "note")]
        bounds = B.chunk_bounds(paths[run], int(chunk))
        n = sum(len(v) for v in groups.values())
        header = (f"**After chunk {int(chunk)}** ({format_timestamp(bounds[0], decimals=0)}-"
                  f"{format_timestamp(bounds[1], decimals=0)}): {n} records")
        return (header, *tables, B.ops_at(paths[run], int(chunk)),
                B.timeline_figure(paths[run], int(chunk), duration))

    def on_run(run: str):
        n = B.n_chunks(paths[run])
        return gr.update(maximum=max(0, n - 1), value=n - 1), *on_chunk(run, n - 1)

    def on_ask(run: str, question: str):
        if not question or not question.strip():
            return "Type a question.", [], [], None
        res = B.ask(cfg, paths[run], question.strip())
        a = res["answer"]
        cited = res["cited"]
        rows = [[("* " if r.id in cited else "") + r.id, r.kind, r.line()] for r in res["retrieved"]]
        iv = ", ".join(f"{format_timestamp(x, decimals=0)}-{format_timestamp(y, decimals=0)}"
                       for x, y in a.cited_intervals) or "none"
        md = (f"### {a.answer or '(no answer)'}\n\nconfidence: **{a.confidence}** · cited: "
              f"{', '.join(a.cited_ids) or 'none'} · support: {iv}"
              + (f"\n\ninvalid citations: {', '.join(a.invalid_citations)}" if a.invalid_citations else ""))
        clip = None
        if a.cited_ids:
            clip = str(B.clip_for_record(cfg, paths[run], a.cited_ids[0]))
        return md, rows, a.cited_ids, clip

    def on_open(run: str, record_id: str):
        if not record_id:
            return None
        return str(B.clip_for_record(cfg, paths[run], record_id.lstrip("* ").strip()))

    def on_compare(question: str):
        out = []
        for r in choices:
            if r.startswith(("C1_", "C2_", "C3_", "C4_")) and "-" not in r:
                res = B.ask(cfg, paths[r], question)
                out.append([r, res["answer"].answer, ", ".join(res["answer"].cited_ids),
                            res["answer"].confidence])
        return out

    with gr.Blocks(title=f"memvid: {cfg.video.video_id}") as app:
        gr.Markdown(f"# Memory for long videos: `{cfg.video.video_id}`")
        run_dd = gr.Dropdown(choices, value=default, label="Run (condition_budget)")
        with gr.Tabs():
            with gr.Tab("Memory over time"):
                chunk = gr.Slider(0, B.n_chunks(paths[default]) - 1, step=1,
                                  value=B.n_chunks(paths[default]) - 1, label="After chunk")
                header = gr.Markdown()
                plot = gr.Plot(label="Records on the video timeline")
                with gr.Tabs():
                    tabs: list[Any] = []
                    for name in ("Entities", "Events", "State changes", "Relations", "Notes"):
                        with gr.Tab(name):
                            tabs.append(gr.Dataframe(headers=B.RECORD_HEADERS, wrap=True))
                ops = gr.Dataframe(headers=B.OP_HEADERS, label="Operations at this chunk", wrap=True)
            with gr.Tab("Ask"):
                q = gr.Textbox(label="Question", placeholder="What color was the board before it was painted?")
                go = gr.Button("Answer from memory", variant="primary")
                ans = gr.Markdown()
                retrieved = gr.Dataframe(headers=["id (* = cited)", "kind", "record"], wrap=True,
                                         label="Retrieved records")
                cited = gr.Dropdown([], label="Open the clip of a cited record")
                video = gr.Video(label="Supporting clip")
            with gr.Tab("Compare conditions"):
                q2 = gr.Textbox(label="Question")
                go2 = gr.Button("Ask every condition")
                cmp_ = gr.Dataframe(headers=["run", "answer", "cited", "confidence"], wrap=True)
        outs = [header, *tabs, ops, plot]
        run_dd.change(on_run, run_dd, [chunk, *outs])
        chunk.release(on_chunk, [run_dd, chunk], outs)
        app.load(on_chunk, [run_dd, chunk], outs)

        def ask_and_choices(run, question):
            md, rows, ids, clip = on_ask(run, question)
            return md, rows, gr.update(choices=ids, value=ids[0] if ids else None), clip

        go.click(ask_and_choices, [run_dd, q], [ans, retrieved, cited, video])
        cited.change(on_open, [run_dd, cited], video)
        go2.click(on_compare, q2, cmp_)
    return app
