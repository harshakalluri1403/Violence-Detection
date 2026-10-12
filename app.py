#!/usr/bin/env python
"""Gradio web demo: upload a video, get an annotated copy, a violence timeline
and a list of incidents.

    python app.py                       # http://127.0.0.1:7860
    python app.py --share               # temporary public link
"""
import argparse
import os
import tempfile

import gradio as gr

from vdet.stream import ViolenceDetector
from vdet.viz import plot_timeline, render_annotated

DET = None


def analyze(video, threshold, explain, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a video first.")
    progress(0, desc="Scoring windows")
    tl = DET.analyze(video, batch=16, keep_above=threshold if explain else None,
                     progress=lambda f: progress(0.7 * f, desc="Scoring windows"))
    events = tl.events(on=threshold, off=max(threshold - 0.2, 0.05))
    cams = None
    if explain and events:
        progress(0.75, desc="Grad-CAM")
        from vdet.explain import cams_for_timeline
        cams = cams_for_timeline(DET, tl)
    progress(0.8, desc="Rendering")
    out = os.path.join(tempfile.mkdtemp(), "annotated.mp4")
    render_annotated(video, out, tl, events, threshold, cams)
    fig = plot_timeline(tl, events, threshold)
    pct = lambda p: f"{min(p, 0.99):.0%}"  # noqa: E731  (never display certainty)
    rows = [[f"{e.start:.1f}", f"{e.end:.1f}", f"{e.end - e.start:.1f}", pct(e.peak)] for e in events]
    peak = float(tl.smoothed().max()) if len(tl.prob) else 0.0
    if events:
        summary = (f"### Violence detected: {len(events)} incident(s)\n"
                   f"Longest {max(e.end - e.start for e in events):.1f} s · peak probability {pct(peak)} · "
                   f"video length {tl.duration:.1f} s")
    else:
        summary = f"### No violence detected\nPeak probability {pct(peak)} · video length {tl.duration:.1f} s"
    return out, fig, rows, summary


def build_ui(examples):
    with gr.Blocks(title="Violence detection") as demo:
        gr.Markdown("# Violence detection in video\nVideoMAE fine-tuned on four violence datasets "
                    "(CCTV, movies, sports, street footage). It scores 3.2 s sliding windows and "
                    "turns them into incidents with start and end times.")
        with gr.Row():
            with gr.Column(scale=1):
                vid = gr.Video(label="Input video", sources=["upload"])
                thr = gr.Slider(0.3, 0.95, value=0.6, step=0.05, label="Alarm threshold")
                exp = gr.Checkbox(value=True, label="Show where (Grad-CAM heatmaps during incidents)")
                btn = gr.Button("Analyze", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=vid)
            with gr.Column(scale=1):
                summary = gr.Markdown()
                out_vid = gr.Video(label="Annotated", show_label=False)  # the label badge would cover the status bar
        plot = gr.Plot(label="Timeline")
        table = gr.Dataframe(headers=["start (s)", "end (s)", "duration (s)", "peak"], label="Incidents")
        btn.click(analyze, [vid, thr, exp], [out_vid, plot, table, summary])
    return demo


def main():
    global DET
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/vdet_base.pt")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    DET = ViolenceDetector(args.checkpoint)
    ex_dir = "examples"
    examples = sorted(os.path.join(ex_dir, f) for f in os.listdir(ex_dir)
                      if f.endswith((".mp4", ".avi"))) if os.path.isdir(ex_dir) else []
    build_ui(examples).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
