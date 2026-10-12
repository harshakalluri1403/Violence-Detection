#!/usr/bin/env python
"""README assets from held-out long-bench videos: a close-up GIF (a CCTV scene,
then a street scene, each normal -> fight -> alarm with Grad-CAM), the full
annotated MP4s and the timeline plot.

    python scripts/make_assets.py --checkpoint runs/vmae_b_mixed/model.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_longbench import gt_events  # noqa: E402
from vdet.explain import cams_for_timeline  # noqa: E402
from vdet.stream import ViolenceDetector  # noqa: E402
from vdet.viz import plot_timeline, render_annotated, video_to_gif  # noqa: E402


def rank_videos(bench, lb_dir, model, source):
    """Long-bench videos of one source with >= 2 fights, best-separated first."""
    from sklearn.metrics import roc_auc_score
    scored = []
    for meta_path in sorted(glob.glob(os.path.join(bench, f"{source}_*.json"))):
        name = os.path.basename(meta_path)[:-5]
        z = os.path.join(lb_dir, "timelines", model, name + ".npz")
        if not os.path.exists(z):
            continue
        gts = gt_events(json.load(open(meta_path))["segments"])
        z = np.load(z)
        tc = z["t_end"] - 1.6
        y = np.array([any(s <= t < e for s, e in gts) for t in tc])
        if len(gts) >= 2 and y.any() and not y.all():
            scored.append((roc_auc_score(y, z["prob"]), name))
    return [n for _, n in sorted(scored, reverse=True)]


def scene(det, bench, name, out_dir, threshold):
    src = os.path.join(bench, name + ".mp4")
    gts = gt_events(json.load(open(os.path.join(bench, name + ".json")))["segments"])
    tl = det.analyze(src, keep_above=threshold)
    events = tl.events(on=threshold, off=threshold - 0.2)
    cams = cams_for_timeline(det, tl)
    mp4 = os.path.join(out_dir, f"demo_{name}.mp4")
    render_annotated(src, mp4, tl, events, threshold, cams, max_width=640)
    # A fight that is preceded by >= 3 s of normal footage and was detected.
    start = None
    for s, e in gts:
        if s >= 3 and any(ev.start <= e and ev.end >= s for ev in events):
            start = s
            break
    start = gts[0][0] if start is None else start
    return mp4, tl, events, gts, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/vmae_b_mixed/model.pt")
    ap.add_argument("--bench", default="data/longbench")
    ap.add_argument("--lb", default="results/longbench")
    ap.add_argument("--model-name", default="vmae_b_mixed")
    ap.add_argument("--cctv", default=None, help="long-bench video for the CCTV scene (default: best)")
    ap.add_argument("--street", default=None, help="long-bench video for the street scene (default: best)")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    det = ViolenceDetector(args.checkpoint)

    cctv = args.cctv or rank_videos(args.bench, args.lb, args.model_name, "rwf")[0]
    street = args.street or rank_videos(args.bench, args.lb, args.model_name, "rlvs")[0]
    segs = []
    for name in (cctv, street):
        mp4, tl, events, gts, start = scene(det, args.bench, name, args.out, args.threshold)
        segs.append((mp4, max(0.0, start - 3.0), 9.0))
        print(f"{name}: fights {[(round(a, 1), round(b, 1)) for a, b in gts]} | alarms "
              f"{[(round(e.start, 1), round(e.end, 1)) for e in events]} | clip from {start - 3:.1f}s")
        if name == cctv:
            plot_timeline(tl, events, args.threshold, gt=gts, path=os.path.join(args.out, "demo_timeline.png"),
                          title="Violence probability on a 2-minute held-out CCTV montage (grey = real fights)")
    size = video_to_gif(segs, os.path.join(args.out, "demo_single.gif"), fps=10, width=560)
    print(f"demo_single.gif: {size / 1e6:.1f} MB from {cctv} + {street}")


if __name__ == "__main__":
    main()
