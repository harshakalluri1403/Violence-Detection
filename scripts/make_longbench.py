#!/usr/bin/env python
"""Build a long-video benchmark by stitching held-out *test* clips.

Clip datasets only answer "is this 2-5 s clip violent?". A deployed system
watches hours of footage and must say *when*. Each benchmark video here is
~2 minutes: 24 test clips from one source dataset, ~25 % of them violent,
re-timed to 25 fps at 640x360. Ground truth is the time span of every violent
clip, so we can measure alarm precision/recall, false alarms per hour and
detection delay. No clip used here was seen in training.

    python scripts/make_longbench.py --index data/index.csv --out data/longbench
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FPS_OUT, SIZE_OUT, MAX_CLIP_SEC = 25, (640, 360), 10.0


def read_resized(path, size):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not 5 <= fps <= 61:
        fps = 25.0
    return frames, fps


def build_video(job):
    out_path, clips = job
    if os.path.exists(out_path) and os.path.exists(out_path.replace(".mp4", ".json")):
        return out_path
    vw = cv2.VideoWriter(out_path + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, SIZE_OUT)
    t, segs = 0.0, []
    for c in clips:
        frames, fps = read_resized(c["path"], SIZE_OUT)
        if len(frames) < 2:
            continue
        dur = len(frames) / fps
        if dur > MAX_CLIP_SEC:  # keep the middle MAX_CLIP_SEC seconds of long clips
            n_keep = int(MAX_CLIP_SEC * fps)
            s = (len(frames) - n_keep) // 2
            frames, dur = frames[s:s + n_keep], MAX_CLIP_SEC
        n_out = max(1, int(round(dur * FPS_OUT)))
        for i in np.linspace(0, len(frames) - 1, n_out).round().astype(int):
            vw.write(frames[i])
        segs.append({"start": round(t, 3), "end": round(t + n_out / FPS_OUT, 3),
                     "label": int(c["label"]), "clip_id": c["clip_id"]})
        t += n_out / FPS_OUT
    vw.release()
    os.replace(out_path + ".tmp.mp4", out_path)
    with open(out_path.replace(".mp4", ".json"), "w") as f:
        json.dump({"duration": t, "segments": segs}, f, indent=1)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data/longbench")
    ap.add_argument("--sources", default="rwf,rlvs")
    ap.add_argument("--videos-per-source", type=int, default=15)
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--p-violent", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for src in args.sources.split(","):
        te = df[(df.split == "test") & (df.dataset == src)]
        vio, nor = te[te.label == 1], te[te.label == 0]
        for v in range(args.videos_per_source):
            k = int(np.clip(rng.binomial(args.clips, args.p_violent), 1, args.clips - 1))
            pick = pd.concat([vio.sample(k, random_state=int(rng.integers(1 << 31))),
                              nor.sample(args.clips - k, random_state=int(rng.integers(1 << 31)))])
            pick = pick.iloc[rng.permutation(len(pick))]
            clips = pick[["path", "label", "clip_id"]].to_dict("records")
            jobs.append((os.path.join(args.out, f"{src}_{v:02d}.mp4"), clips))
    with Pool(min(6, os.cpu_count())) as pool:
        for p in pool.imap_unordered(build_video, jobs):
            print("built", p, flush=True)


if __name__ == "__main__":
    main()
