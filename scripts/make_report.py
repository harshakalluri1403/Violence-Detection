#!/usr/bin/env python
"""Collect every run into results/summary.json, markdown tables and figures.

    python scripts/make_report.py --runs runs --out results
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vdet.data import DATASET_NAMES, DATASETS  # noqa: E402
from vdet.viz import INK, INK2, MUTED, SERIES, SURFACE, apply_style  # noqa: E402

apply_style(plt)
TEST_LABELS = {"hockey": "Hockey", "movies": "Movies", "rwf": "RWF-2000\n(CCTV)", "rlvs": "Real-Life\nViolence"}
TRAIN_LABELS = {"hockey": "Hockey", "rwf": "RWF-2000", "rlvs": "Real-Life V.", "mixed": "All four"}

MAIN_METHODS = [  # (run, label)
    ("vgg_lstm_hockey", "VGG16+LSTM on Hockey (original repo approach)"),
    ("clip_zeroshot", "CLIP ViT-L/14 zero-shot (threshold set on validation clips)"),
    ("vmae_b_hockey", "VideoMAE-B on Hockey only"),
    ("vgg_lstm_mixed", "VGG16+LSTM on all four datasets"),
    ("vmae_s_mixed", "VideoMAE-S on all four (lite, 22M)"),
    ("vmae_b_mixed", "**VideoMAE-B on all four (ours, 86M)**"),
]


def load_runs(runs_dir):
    out = {}
    for p in glob.glob(os.path.join(runs_dir, "*", "metrics.json")):
        out[os.path.basename(os.path.dirname(p))] = json.load(open(p))["results"]
    return out


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def table_main(R):
    head = "| Method | " + " | ".join(DATASET_NAMES[d] for d in DATASETS) + " | Mean |\n"
    head += "|---|" + "---:|" * (len(DATASETS) + 1) + "\n"
    rows = []
    for run, label in MAIN_METHODS:
        if run not in R:
            continue
        accs = [R[run].get(d, {}).get("acc") for d in DATASETS]
        aucs = [R[run].get(d, {}).get("auc") for d in DATASETS]
        cells = [f"{fmt(a)} <sub>({fmt(u, False)})</sub>" for a, u in zip(accs, aucs)]
        mean = np.nanmean([a for a in accs if a is not None])
        rows.append(f"| {label} | " + " | ".join(cells) + f" | **{fmt(mean)}** |")
    return head + "\n".join(rows) + "\n"


def table_lodo(R):
    head = "| Held-out dataset | VGG16+LSTM | VideoMAE-B | VideoMAE-B trained *with* it (reference) |\n|---|---:|---:|---:|\n"
    rows = []
    for d in DATASETS:
        a = R.get(f"vgg_lstm_lodo_{d}", {}).get(d, {})
        b = R.get(f"vmae_b_lodo_{d}", {}).get(d, {})
        c = R.get("vmae_b_mixed", {}).get(d, {})
        if not b:
            continue
        rows.append(f"| {DATASET_NAMES[d]} | {fmt(a.get('acc'))} <sub>({fmt(a.get('auc'), False)})</sub> | "
                    f"**{fmt(b.get('acc'))}** <sub>({fmt(b.get('auc'), False)})</sub> | "
                    f"{fmt(c.get('acc'))} <sub>({fmt(c.get('auc'), False)})</sub> |")
    return head + "\n".join(rows) + "\n"


def table_ablation(R):
    a, b = R.get("vmae_b_mixed_nocctv"), R.get("vmae_b_mixed")
    if not (a and b):
        return ""
    head = "| CCTV augmentation | " + " | ".join(DATASET_NAMES[d] for d in DATASETS) + " |\n"
    head += "|---|" + "---:|" * len(DATASETS) + "\n"
    rows = [f"| off | " + " | ".join(fmt(a[d]["acc"]) for d in DATASETS) + " |",
            f"| on (default) | " + " | ".join(fmt(b[d]["acc"]) for d in DATASETS) + " |"]
    return head + "\n".join(rows) + "\n"


def table_longbench(L):
    names = {"vmae_b_mixed": "VideoMAE-B, all four (ours)", "vmae_s_mixed": "VideoMAE-S, all four (lite)",
             "vmae_b_hockey": "VideoMAE-B, Hockey only", "vmae_b_mixed_nocctv": "VideoMAE-B, all four, no CCTV aug"}
    src_names = {"rwf": "CCTV (RWF-2000)", "rlvs": "Web video (RLVS)"}
    head = ("| Model | Footage | Frame AUC | Alarm precision | Alarm recall | False alarms / hour | "
            "Live alarm delay |\n|---|---|---:|---:|---:|---:|---:|\n")
    rows = []
    for m, label in names.items():
        if m not in L:
            continue
        for src in ["rwf", "rlvs"]:
            r = L[m].get(src)
            if not r:
                continue
            d = r["default"]
            rows.append(f"| {label} | {src_names[src]} | {r['frame_auc']:.3f} | {fmt(d['event_precision'])}% | "
                        f"{fmt(d['event_recall'])}% | {d['false_alarms_per_hour']:.1f} | "
                        f"{d['live_median_delay_s']:.1f} s |")
    return head + "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_main(R, path):
    runs = [("vgg_lstm_hockey", "VGG16+LSTM, Hockey-trained (original approach)"),
            ("clip_zeroshot", "CLIP zero-shot (threshold set on validation clips)"),
            ("vmae_b_hockey", "VideoMAE-B, Hockey-trained"),
            ("vmae_b_mixed", "VideoMAE-B, all four datasets (ours)")]
    runs = [r for r in runs if r[0] in R]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    n, width = len(runs), 0.8 / max(len(runs), 1)
    x = np.arange(len(DATASETS))
    for i, (run, label) in enumerate(runs):
        vals = [100 * R[run][d]["acc"] for d in DATASETS]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.88, color=SERIES[i], label=label,
                      edgecolor=SURFACE, linewidth=1)
        if run == "vmae_b_mixed":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
    ax.axhline(50, color=MUTED, lw=1)
    ax.text(0.5, 51, "chance", color=MUTED, fontsize=8, ha="center", va="bottom")  # in the gap between groups
    ax.set_xticks(x, [TEST_LABELS[d] for d in DATASETS])
    ax.set_ylim(0, 106)  # bars start at zero: a short bar is a real failure, not a cropped one
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy on each held-out test set", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cross(R, path):
    fams = [("vgg_lstm", "VGG16+LSTM (original approach)"), ("vmae_b", "VideoMAE-B (ours)")]
    trains = ["hockey", "rwf", "rlvs", "mixed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    for ax, (fam, title) in zip(axes, fams):
        M = np.full((len(trains), len(DATASETS)), np.nan)
        for i, t in enumerate(trains):
            r = R.get(f"{fam}_{t}")
            if r:
                M[i] = [100 * r[d]["acc"] for d in DATASETS]
        ax.imshow(M, cmap=cmap, vmin=50, vmax=100, aspect="auto")
        for i in range(len(trains)):
            for j in range(len(DATASETS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if M[i, j] > 78 else INK)
        ax.set_xticks(range(len(DATASETS)), [TEST_LABELS[d] for d in DATASETS], fontsize=8)
        ax.set_yticks(range(len(trains)), [TRAIN_LABELS[t] for t in trains], fontsize=8)
        ax.set_xlabel("tested on")
        ax.set_ylabel("trained on")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Cross-dataset accuracy (%): does a model trained on one kind of footage work on another?",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_operating(L, path):
    models = [("vmae_b_mixed", "VideoMAE-B, all four (ours)"), ("vmae_s_mixed", "VideoMAE-S, all four (lite)"),
              ("vmae_b_hockey", "VideoMAE-B, Hockey only")]
    models = [m for m in models if m[0] in L]
    srcs = [("rwf", "CCTV footage (RWF-2000)"), ("rlvs", "Web video (RLVS)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150, sharey=True)
    for ax, (src, title) in zip(axes, srcs):
        for i, (m, label) in enumerate(models):
            curve = L[m].get(src, {}).get("curve", [])
            if not curve:
                continue
            fa = [max(c["false_alarms_per_hour"], 0.1) for c in curve]
            rec = [100 * c["event_recall"] for c in curve]
            ax.plot(fa, rec, color=SERIES[i], lw=2, marker="o", ms=4, label=label)
            d = L[m][src]["default"]
            ax.plot([max(d["false_alarms_per_hour"], 0.1)], [100 * d["event_recall"]], "o", ms=9,
                    mfc=SERIES[i], mec=SURFACE, mew=2)
        ours = L.get("vmae_b_mixed", {}).get(src, {}).get("curve", [])
        for c in ours:  # say which threshold each end of our curve is
            if c["threshold"] == 0.95:
                ax.annotate("threshold 0.95", (max(c["false_alarms_per_hour"], 0.1), 100 * c["event_recall"]),
                            textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ticks = [3, 10, 30, 100, 300]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(3, 300)
        ax.set_xlabel("false alarms per hour of normal footage (log scale)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(50, 101)
    axes[0].set_ylabel("incidents detected (%)")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Alarm operating curves on 2-minute stitched videos", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.905, "Each curve sweeps the alarm threshold from 0.3 (right) to 0.95 (left); "
             "large dot = default 0.6. Up and to the left is better.", ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))  # leave room for the title + subtitle
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_timeline_example(bench_dir, lb_dir, path, model="vmae_b_mixed", video="rwf_03"):
    from vdet.stream import Timeline
    from vdet.viz import plot_timeline
    from scripts.eval_longbench import gt_events
    z = os.path.join(lb_dir, "timelines", model, f"{video}.npz")
    meta = os.path.join(bench_dir, f"{video}.json")
    if not (os.path.exists(z) and os.path.exists(meta)):
        return False
    z = np.load(z)
    tl = Timeline(z["t_end"], z["prob"], 3.2, float(z["duration"]), float(z["fps"]))
    gt = gt_events(json.load(open(meta))["segments"])
    plot_timeline(tl, tl.events(0.6, 0.4), 0.6, gt=gt, path=path,
                  title=f"Streaming output on a 2-minute CCTV montage ({video}) — grey = real fights")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bench", default="data/longbench")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    R = load_runs(args.runs)
    lb_path = os.path.join(args.out, "longbench", "summary.json")
    L = json.load(open(lb_path)) if os.path.exists(lb_path) else {}

    md = ["## Clip classification — accuracy % (ROC-AUC) on each test set\n", table_main(R),
          "\n## Leave-one-dataset-out — accuracy on a dataset never seen in training\n", table_lodo(R)]
    if L:
        md += ["\n## Long-video streaming benchmark\n", table_longbench(L)]
    abl = table_ablation(R)
    if abl:
        md += ["\n## Ablation: CCTV-style augmentation — accuracy %\n", abl]
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"clip": R, "longbench": L}, f, indent=1)

    if all(r in R for r in ["vgg_lstm_hockey", "vmae_b_mixed"]):
        fig_main(R, os.path.join(fig_dir, "accuracy_by_dataset.png"))
    if any(k.startswith("vmae_b_") for k in R):
        fig_cross(R, os.path.join(fig_dir, "cross_dataset.png"))
    if L:
        fig_operating(L, os.path.join(fig_dir, "operating_curves.png"))
        fig_timeline_example(args.bench, os.path.join(args.out, "longbench"),
                             os.path.join(fig_dir, "timeline_example.png"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
