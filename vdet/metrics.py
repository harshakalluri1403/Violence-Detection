"""Clip-level and timeline-level metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "ap": float(average_precision_score(y, p)) if both else float("nan"),
        "ece": expected_calibration_error(y, p),
    }


# --------------------------------------------------------------------------- #
# Timeline (long-video) metrics
# --------------------------------------------------------------------------- #
def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def event_metrics(pred_events, gt_events, duration, violent_time, tol=1.6, min_inside=0.5):
    """Event-level matching on one or more videos.

    pred_events / gt_events: lists of (start, end) in seconds. Ground-truth
    events are widened by `tol` seconds on each side (a sliding window cannot
    localise more precisely than about half its length). An alarm is *correct*
    if at least `min_inside` of its duration lies inside the widened ground
    truth, otherwise it is a false alarm — so an alarm that stays on through
    normal footage is penalised rather than counted as a hit. A ground-truth
    event is detected if a correct alarm overlaps it.
    """
    wide = [(s - tol, e + tol) for s, e in gt_events]
    correct = []
    for p in pred_events:
        inside = sum(_overlap(p, g) for g in wide)
        correct.append(inside >= min_inside * max(p[1] - p[0], 1e-9))
    good = [p for p, c in zip(pred_events, correct) if c]
    tp_pred, fp = len(good), len(pred_events) - len(good)
    hits, delays = 0, []
    for g, gw in zip(gt_events, wide):
        ov = [p for p in good if _overlap(p, gw) > 0]
        if ov:
            hits += 1
            delays.append(max(0.0, min(p[0] for p in ov) - g[0]))
    normal_hours = max(duration - violent_time, 1e-9) / 3600.0
    return {
        "event_precision": tp_pred / max(len(pred_events), 1),
        "event_recall": hits / max(len(gt_events), 1),
        "false_alarms": int(fp),
        "false_alarms_per_hour": fp / normal_hours,
        "median_delay_s": float(np.median(delays)) if delays else float("nan"),
        "n_pred": len(pred_events),
        "n_gt": len(gt_events),
    }
