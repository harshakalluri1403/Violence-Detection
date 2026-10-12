#!/usr/bin/env python
"""Index the raw datasets and extract 32 frames per clip (one-time).

    python scripts/prepare_data.py --raw ~/vd/raw --out data
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdet.data import discover_videos, extract_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    df = discover_videos(os.path.expanduser(args.raw))
    print(df.groupby(["dataset", "split", "label"]).size().unstack(fill_value=0))
    frames_root = os.path.join(args.out, "frames")
    jobs = [(p, os.path.join(frames_root, cid)) for p, cid in zip(df.path, df.clip_id)]
    with Pool(args.workers) as pool:
        metas = list(tqdm(pool.imap(extract_clip, jobs, chunksize=2), total=len(jobs)))

    ok = np.array([m is not None for m in metas])
    if (~ok).any():
        print(f"dropping {(~ok).sum()} unreadable clips:", df.path[~ok].tolist()[:10])
    meta = pd.DataFrame([m or {} for m in metas], index=df.index)
    df = pd.concat([df, meta], axis=1)[ok]
    df["duration"] = df.n_frames / df.fps.replace(0, np.nan)
    df.to_csv(os.path.join(args.out, "index.csv"), index=False)
    print(df.groupby("dataset").agg(clips=("clip_id", "size"), violent=("label", "mean"),
                                    dur_s=("duration", "median"), fps=("fps", "median"),
                                    h=("height", "median"), w=("width", "median")))


if __name__ == "__main__":
    main()
