#!/usr/bin/env python
"""Evaluate trained models as *streaming detectors* on the long-video benchmark.

Reports, per source (CCTV = RWF-2000, mixed web video = RLVS):
  * frame-level ROC-AUC / AP of the smoothed score
  * alarm precision / recall and false alarms per hour at the default threshold
  * live alarm latency: seconds from fight start to the first alarm a live
    system would raise (window end time, trailing average of 3 windows)
  * an operating curve (recall vs false alarms/hour) over thresholds
  * throughput (seconds of video processed per wall-clock second)

    python scripts/eval_longbench.py --models vmae_b_mixed,vmae_s_mixed,vmae_b_hockey
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdet.metrics import event_metrics  # noqa: E402
from vdet.stream import Timeline, ViolenceDetector, extract_events  # noqa: E402

THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def gt_events(segs):
    ev = []
    for s in segs:
        if s["label"] != 1:
            continue
        if ev and abs(s["start"] - ev[-1][1]) < 1e-3:
            ev[-1][1] = s["end"]
        else:
            ev.append([s["start"], s["end"]])
    return [tuple(e) for e in ev]


def live_delays(tl, gts, on, k=3):
    causal = np.convolve(np.pad(tl.prob, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")
    out = []
    for s, e in gts:
        m = (tl.t_end >= s) & (tl.t_end <= e + tl.window_sec) & (causal >= on)
        out.append(float(tl.t_end[m][0] - s) if m.any() else None)
    return out


def evaluate_source(tls, gts_all, durations, window_sec, default_on=0.6):
    ys, ss = [], []
    for tl, gts in zip(tls, gts_all):
        grid = np.arange(0, tl.duration, 0.2)
        y = np.zeros(len(grid), int)
        for s, e in gts:
            y[(grid >= s) & (grid < e)] = 1
        ys.append(y)
        ss.append(np.interp(grid, tl.t_center, tl.smoothed()) if len(tl.prob) else np.zeros(len(grid)))
    y, s = np.concatenate(ys), np.concatenate(ss)
    out = {"frame_auc": float(roc_auc_score(y, s)), "frame_ap": float(average_precision_score(y, s)),
           "hours": float(sum(durations) / 3600)}
    violent_time = sum(e - s for gts in gts_all for s, e in gts)

    def at(on):
        preds, gts_flat, delays = [], [], []
        offset = 0.0
        for tl, gts, dur in zip(tls, gts_all, durations):
            ev = tl.events(on=on, off=max(on - 0.2, 0.05))
            preds += [(e.start + offset, e.end + offset) for e in ev]
            gts_flat += [(a + offset, b + offset) for a, b in gts]
            delays += live_delays(tl, gts, on)
            offset += dur + 1000.0  # keep videos apart
        m = event_metrics(preds, gts_flat, sum(durations), violent_time, tol=window_sec / 2)
        hit = [d for d in delays if d is not None]
        m["live_recall"] = len(hit) / max(len(delays), 1)
        m["live_median_delay_s"] = float(np.median(hit)) if hit else float("nan")
        return m

    out["default"] = {"threshold": default_on, **at(default_on)}
    out["curve"] = [{"threshold": float(t), **at(t)} for t in THRESHOLDS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/longbench")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", default="vmae_b_mixed")
    ap.add_argument("--window", type=float, default=3.2)
    ap.add_argument("--stride", type=float, default=0.4)
    ap.add_argument("--out", default="results/longbench")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(args.bench, "*.mp4")))
    videos = [v for v in videos if not v.endswith(".tmp.mp4")]
    summary = {}
    for name in args.models.split(","):
        ckpt = os.path.join(args.runs, name, "model.pt")
        if not os.path.exists(ckpt):
            print("missing", ckpt)
            continue
        det = ViolenceDetector(ckpt, window_sec=args.window, stride_sec=args.stride)
        tl_dir = os.path.join(args.out, "timelines", name)
        os.makedirs(tl_dir, exist_ok=True)
        per_src, wall, vid_sec = {}, 0.0, 0.0
        for v in videos:
            meta = json.load(open(v.replace(".mp4", ".json")))
            cache = os.path.join(tl_dir, os.path.basename(v).replace(".mp4", ".npz"))
            tl = None
            if os.path.exists(cache):
                try:
                    z = np.load(cache)
                    tl = Timeline(z["t_end"], z["prob"], args.window, float(z["duration"]), float(z["fps"]))
                except Exception:  # truncated by a preemption mid-write: recompute
                    os.remove(cache)
            if tl is None:
                t0 = time.time()
                tl = det.analyze(v, batch=32)
                wall += time.time() - t0
                vid_sec += tl.duration
                with open(cache + ".tmp", "wb") as f:  # atomic: a preemption never leaves half a file
                    np.savez(f, t_end=tl.t_end, prob=tl.prob, duration=tl.duration, fps=tl.fps)
                os.replace(cache + ".tmp", cache)
            src = os.path.basename(v).split("_")[0]
            per_src.setdefault(src, ([], [], []))
            per_src[src][0].append(tl)
            per_src[src][1].append(gt_events(meta["segments"]))
            per_src[src][2].append(meta["duration"])
        res = {src: evaluate_source(*vals, args.window) for src, vals in per_src.items()}
        if wall > 0:
            res["realtime_factor"] = vid_sec / wall
        summary[name] = res
        for src, r in res.items():
            if isinstance(r, dict):
                d = r["default"]
                print(f"{name:22s} {src:5s} AUC {r['frame_auc']:.3f} AP {r['frame_ap']:.3f} | "
                      f"alarm P {d['event_precision']:.2f} R {d['event_recall']:.2f} "
                      f"FA/h {d['false_alarms_per_hour']:.1f} | live delay {d['live_median_delay_s']:.1f}s")
        del det
    prev = os.path.join(args.out, "summary.json")
    if os.path.exists(prev):
        old = json.load(open(prev))
        old.update(summary)
        summary = old
    with open(prev, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
