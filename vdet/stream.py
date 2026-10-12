"""Sliding-window violence scoring for video files, webcams and RTSP streams.

A video is treated as a stream: frames are subsampled to
num_frames / window_sec fps, and every `stride_sec` the most recent
`num_frames` samples form one window that the model scores. Scores are then
smoothed and turned into events with hysteresis thresholds, which is what a
real alerting system needs (one alert per fight, not one per frame).
"""
import collections
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .models import load_checkpoint


@dataclass
class Event:
    start: float
    end: float
    peak: float
    mean: float

    def to_dict(self):
        return {k: round(float(v), 3) for k, v in asdict(self).items()}


@dataclass
class Timeline:
    t_end: np.ndarray        # end time of each scored window (s) — when a live system would know
    prob: np.ndarray         # raw violence probability per window
    window_sec: float
    duration: float
    fps: float
    kept: dict = field(default_factory=dict)  # t_end -> uint8 window frames (for Grad-CAM)

    @property
    def t_center(self):
        return np.clip(self.t_end - self.window_sec / 2, 0, None)

    def smoothed(self, k=3):
        if len(self.prob) < k or k <= 1:
            return self.prob.copy()
        pad = np.pad(self.prob, (k // 2, k - 1 - k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    def score_at(self, t, k=3):
        if len(self.prob) == 0:
            return 0.0
        return float(np.interp(t, self.t_center, self.smoothed(k)))

    def events(self, on=0.6, off=0.4, min_dur=0.8, merge_gap=1.5, k=3):
        return extract_events(self.t_center, self.smoothed(k), on, off, min_dur, merge_gap,
                              pad=(self.t_end[1] - self.t_end[0]) / 2 if len(self.t_end) > 1 else 0.2)


def extract_events(times, scores, on=0.6, off=0.4, min_dur=0.8, merge_gap=1.5, pad=0.2):
    """Hysteresis: open when score >= on, close when it falls below off."""
    events, start, vals = [], None, []
    for t, s in zip(times, scores):
        if start is None and s >= on:
            start, vals = t, [s]
        elif start is not None:
            if s < off:
                events.append([start - pad, prev_t + pad, vals])
                start = None
            else:
                vals.append(s)
        prev_t = t
    if start is not None:
        events.append([start - pad, times[-1] + pad, vals])
    merged = []
    for e in events:
        if merged and e[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = e[1]
            merged[-1][2] += e[2]
        else:
            merged.append(e)
    return [Event(max(0.0, s), e, float(np.max(v)), float(np.mean(v)))
            for s, e, v in merged if e - s >= min_dur]


class SlidingWindowScorer:
    """Feed (frame, timestamp) pairs; get back (window_end_time, probability).

    batch=1 gives the lowest latency (live use); larger batches are faster for
    files.
    """

    def __init__(self, model, num_frames=16, window_sec=3.2, stride_sec=0.4, batch=8,
                 device="cpu", size=224, keep_above=None, keep_max=400):
        self.model, self.T, self.window = model, num_frames, window_sec
        self.dt = window_sec / num_frames
        self.stride = stride_sec
        self.batch, self.device, self.size = batch, device, size
        self.dtype = next(model.parameters()).dtype
        self.buf = collections.deque(maxlen=num_frames)
        self.next_sample = 0.0
        self.last_window = -1e9
        self.pending = []
        self.keep_above, self.keep_max, self.kept = keep_above, keep_max, {}
        self.mean = MEAN.view(1, 1, 3, 1, 1).to(device, self.dtype)
        self.std = STD.view(1, 1, 3, 1, 1).to(device, self.dtype)

    def push(self, frame_bgr, t):
        out = []
        if t + 1e-6 >= self.next_sample:
            self.buf.append(cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA))
            self.next_sample += self.dt
            if self.next_sample < t:
                self.next_sample = t + self.dt
            if len(self.buf) == self.T and t - self.last_window >= self.stride - 1e-6:
                self.pending.append((t, np.stack(self.buf)))
                self.last_window = t
                if len(self.pending) >= self.batch:
                    out = self._run()
        return out

    def flush(self):
        return self._run() if self.pending else []

    def to_tensor(self, windows):
        """uint8 (B,T,H,W,3) BGR -> normalised (B,T,3,H,W) on device."""
        x = torch.from_numpy(np.ascontiguousarray(windows[..., ::-1])).to(self.device)
        x = x.permute(0, 1, 4, 2, 3).to(self.dtype) / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _run(self):
        times = [t for t, _ in self.pending]
        wins = np.stack([w for _, w in self.pending])
        self.pending = []
        probs = torch.sigmoid(self.model(self.to_tensor(wins)).float()).cpu().numpy()
        if self.keep_above is not None:
            for t, w, p in zip(times, wins, probs):
                if p >= self.keep_above and len(self.kept) < self.keep_max:
                    self.kept[t] = w
        return list(zip(times, probs.tolist()))


class ViolenceDetector:
    """High-level API: load a checkpoint once, analyse files or build live scorers."""

    def __init__(self, checkpoint, device=None, window_sec=3.2, stride_sec=0.4, half=True, int8=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        if half and self.device == "cuda":
            self.model.half()
        self.int8 = int8 and self.device == "cpu"
        if self.int8:  # ~1.4x faster on CPU, probabilities move by < 0.01; no Grad-CAM
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear},
                                                                    dtype=torch.qint8)
        self.num_frames = self.meta.get("num_frames", 16)
        self.window_sec, self.stride_sec = window_sec, stride_sec

    def scorer(self, batch=1, keep_above=None):
        return SlidingWindowScorer(self.model, self.num_frames, self.window_sec, self.stride_sec,
                                   batch=batch, device=self.device, keep_above=keep_above)

    def analyze(self, path, batch=16, keep_above=None, progress=None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(fps) or fps <= 1:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sc = self.scorer(batch, keep_above)
        res, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res += sc.push(frame, i / fps)
            i += 1
            if progress and total and i % 50 == 0:
                progress(min(i / total, 1.0))
        cap.release()
        res += sc.flush()
        if not res and sc.buf:  # clip shorter than one window: score what we have
            frames = list(sc.buf)
            idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
            sc.pending = [(i / fps, np.stack([frames[j] for j in idx]))]
            res = sc.flush()
        t_end = np.array([r[0] for r in res], dtype=float)
        prob = np.array([r[1] for r in res], dtype=float)
        return Timeline(t_end, prob, self.window_sec, i / fps, fps, sc.kept)
