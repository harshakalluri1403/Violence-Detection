"""Fast unit tests (CPU, no downloads): python -m pytest -q"""
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdet.data import augment_clip  # noqa: E402
from vdet.metrics import binary_metrics, event_metrics  # noqa: E402
from vdet.models import (VideoClassifier, layer_decay_param_groups, load_checkpoint,  # noqa: E402
                         save_checkpoint)
from vdet.stream import SlidingWindowScorer, extract_events  # noqa: E402


def tiny_model(num_frames=4, image_size=32):
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    cfg = VideoMAEConfig(image_size=image_size, patch_size=16, num_frames=num_frames, tubelet_size=2,
                         hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         num_labels=1, use_mean_pooling=True)
    return VideoClassifier(VideoMAEForVideoClassification(cfg)).eval()


# ---------------------------------------------------------------- alarms ---- #
def test_hysteresis_opens_high_and_closes_low():
    t = np.arange(0, 20, 0.4)
    s = np.where((t >= 5) & (t < 10), 0.9, 0.1)
    s[(t >= 7) & (t < 7.5)] = 0.5  # dips below `on` but stays above `off`: still one event
    ev = extract_events(t, s, on=0.6, off=0.4, pad=0.2)
    assert len(ev) == 1
    assert ev[0].start == pytest.approx(5.0, abs=0.05)  # first window >= 5 s is 5.2, minus pad
    assert ev[0].end == pytest.approx(9.8, abs=0.05)
    assert ev[0].peak == pytest.approx(0.9)


def test_close_events_merge_and_blips_drop():
    t = np.arange(0, 30, 0.4)
    s = np.full_like(t, 0.1)
    s[(t >= 5) & (t < 8)] = 0.9
    s[(t >= 8.8) & (t < 12)] = 0.9    # 0.8 s gap -> merged with the first
    s[(t >= 20) & (t < 20.3)] = 0.9   # single-window blip -> shorter than min_dur
    ev = extract_events(t, s, on=0.6, off=0.4, min_dur=0.8, merge_gap=1.5, pad=0.2)
    assert len(ev) == 1 and ev[0].start <= 5.0 + 1e-9 and ev[0].end > 11


def test_always_on_alarm_is_a_false_alarm_not_a_hit():
    m = event_metrics([(0, 600)], [(100, 105), (300, 305)], duration=600, violent_time=10)
    assert m["event_recall"] == 0 and m["false_alarms"] == 1


def test_well_placed_alarms_are_hits():
    gt = [(100, 105), (300, 305)]
    m = event_metrics([(99, 106), (301, 307), (450, 452)], gt, duration=600, violent_time=10)
    assert m["event_recall"] == 1 and m["event_precision"] == pytest.approx(2 / 3)
    assert m["false_alarms_per_hour"] == pytest.approx(1 / (590 / 3600))


def test_binary_metrics_perfect_and_calibrated():
    m = binary_metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert m["acc"] == 1 and m["auc"] == 1 and m["ece"] == pytest.approx(0)


# ---------------------------------------------------------------- stream ---- #
class Constant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, x):
        self.calls.append(tuple(x.shape))
        return torch.full((x.shape[0],), 2.0) + self.w


def test_scorer_window_cadence():
    model = Constant()
    sc = SlidingWindowScorer(model, num_frames=16, window_sec=3.2, stride_sec=0.4, batch=4, size=32)
    out, fps = [], 25
    frame = np.zeros((48, 64, 3), np.uint8)
    for i in range(10 * fps):
        out += sc.push(frame, i / fps)
    out += sc.flush()
    t = np.array([o[0] for o in out])
    assert t[0] == pytest.approx(3.0, abs=0.05)            # first full window after 16 samples at 5 fps
    assert np.allclose(np.diff(t), 0.4, atol=0.05)          # one decision every stride
    assert all(shape[1:] == (16, 3, 32, 32) for shape in model.calls)
    assert all(p == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item()) for _, p in out)


# ----------------------------------------------------------- data / model ---- #
def test_augment_clip_shape_and_range():
    random.seed(0)
    for _ in range(20):  # exercise the random greyscale / low-res / blur / noise branches
        x = torch.randint(0, 256, (16, 3, 64, 80), dtype=torch.uint8)
        y = augment_clip(x, size=32, cctv=True)
        assert y.shape == (16, 3, 32, 32) and y.min() >= 0 and y.max() <= 1


def test_layer_decay_head_fastest_embedding_slowest():
    groups = layer_decay_param_groups(tiny_model(), weight_decay=0.05, layer_decay=0.5)
    scales = sorted({g["lr_scale"] for g in groups})
    assert scales[-1] == 1.0 and scales[0] == pytest.approx(0.5 ** 3)
    assert all(g["weight_decay"] == 0 for g in groups if all(p.ndim == 1 for p in g["params"]))


def test_checkpoint_roundtrip(tmp_path):
    m = tiny_model()
    x = torch.randn(2, 4, 3, 32, 32)
    path = tmp_path / "m.pt"
    save_checkpoint(m, str(path), {"num_frames": 4})
    m2, meta = load_checkpoint(str(path))
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-2)
    assert meta["num_frames"] == 4
