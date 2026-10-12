"""Dataset discovery, one-time frame extraction, and the training/eval clip dataset.

Every clip, whatever its source length or resolution, is reduced once to
STORE_FRAMES uniformly spaced JPEG frames of STORE_SIZE x STORE_SIZE. Training
then reads 16 of those 32 frames per step, which keeps the data loader cheap
and lets the whole corpus (~3 GB) sit in the page cache.
"""
import glob
import math
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset

cv2.setNumThreads(0)

DATASETS = ["hockey", "movies", "rwf", "rlvs"]
DATASET_NAMES = {
    "hockey": "Hockey Fights",
    "movies": "Movies Fights",
    "rwf": "RWF-2000 (CCTV)",
    "rlvs": "Real-Life Violence",
}
STORE_FRAMES = 32
STORE_SIZE = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEO_EXT = (".avi", ".mp4", ".mpg", ".mpeg", ".mov", ".mkv", ".wmv")


# --------------------------------------------------------------------------- #
# Discovery & splits
# --------------------------------------------------------------------------- #
def _label_from_path(dataset, path):
    p = path.replace("\\", "/")
    name = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    if dataset == "hockey":
        return 1 if name.startswith("fi") else 0 if name.startswith("no") else None
    if dataset == "movies":
        return {"fights": 1, "nofights": 0}.get(parent)
    if dataset == "rwf":
        return {"fight": 1, "nonfight": 0}.get(parent)
    if dataset == "rlvs":
        return {"violence": 1, "nonviolence": 0}.get(parent)
    return None


def discover_videos(raw_root, seed=0):
    """Index every labelled clip under raw_root/<dataset>/ and assign splits.

    RWF-2000 keeps its official train/val split (val is our test set). The
    others have no official split, so they get stratified 5-fold ids; fold 0 is
    the default test fold.
    """
    from sklearn.model_selection import StratifiedKFold

    rows = []
    for ds in DATASETS:
        seen = set()
        for path in sorted(glob.glob(os.path.join(raw_root, ds, "**", "*"), recursive=True)):
            if not path.lower().endswith(VIDEO_EXT):
                continue
            label = _label_from_path(ds, path)
            if label is None:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            parts = path.replace("\\", "/").lower().split("/")
            split = ("test" if "val" in parts else "train") if ds == "rwf" else None
            key = (label, stem.lower(), split)
            if key in seen:  # the RLVS archive ships every clip twice
                continue
            seen.add(key)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
            rows.append(dict(dataset=ds, clip_id=f"{ds}/{split or 'all'}/{label}_{safe}",
                             path=path, label=label, split=split))
    df = pd.DataFrame(rows)
    df["fold"] = -1
    for ds in DATASETS:
        if ds == "rwf":
            continue
        m = (df.dataset == ds).to_numpy()
        if not m.any():
            continue
        sub = df[m]
        folds = np.zeros(len(sub), dtype=int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(sub)), sub.label)):
            folds[te] = k
        df.loc[m, "fold"] = folds
        df.loc[m, "split"] = np.where(folds == 0, "test", "train")
    return df.reset_index(drop=True)


def apply_test_fold(df, fold):
    """Re-derive train/test for the fold-based datasets (RWF keeps its split)."""
    df = df.copy()
    m = df.dataset != "rwf"
    df.loc[m, "split"] = np.where(df.loc[m, "fold"] == fold, "test", "train")
    return df


# --------------------------------------------------------------------------- #
# One-time frame extraction
# --------------------------------------------------------------------------- #
def read_video_frames(path, size=None):
    """Decode every frame (BGR). Optionally squash-resize to size x size."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if size is not None:
            f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, fps


def extract_clip(args):
    """Worker: write STORE_FRAMES jpgs for one clip; returns metadata or None."""
    path, out_dir = args
    done_marker = os.path.join(out_dir, f"{STORE_FRAMES - 1:02d}.jpg")
    cap = cv2.VideoCapture(path)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    if os.path.exists(done_marker) and os.path.exists(os.path.join(out_dir, "meta.txt")):
        with open(os.path.join(out_dir, "meta.txt")) as f:
            n, fps = f.read().split()
        return dict(n_frames=int(n), fps=float(fps), height=h, width=w)
    frames, fps = read_video_frames(path, STORE_SIZE)
    if len(frames) < 2:
        return None
    os.makedirs(out_dir, exist_ok=True)
    idx = np.linspace(0, len(frames) - 1, STORE_FRAMES).round().astype(int)
    for j, i in enumerate(idx):
        cv2.imwrite(os.path.join(out_dir, f"{j:02d}.jpg"), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"{len(frames)} {fps}")
    return dict(n_frames=len(frames), fps=fps, height=h, width=w)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def _random_box(h, w, scale=(0.45, 1.0), ratio=(3 / 4, 4 / 3)):
    for _ in range(10):
        area = random.uniform(*scale) * h * w
        r = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        bw, bh = int(round(math.sqrt(area * r))), int(round(math.sqrt(area / r)))
        if 0 < bw <= w and 0 < bh <= h:
            return random.randint(0, h - bh), random.randint(0, w - bw), bh, bw
    return 0, 0, h, w


def augment_clip(x, size=224, cctv=True):
    """x: uint8 (T,3,H,W). Same spatial/photometric params for every frame.

    The CCTV branch simulates what real surveillance feeds look like and
    clean YouTube/broadcast clips don't: IR/greyscale night footage, low
    resolution, soft focus and sensor noise.
    """
    _, _, h, w = x.shape
    top, left, bh, bw = _random_box(h, w)
    x = TF.resized_crop(x, top, left, bh, bw, [size, size], antialias=True)
    if random.random() < 0.5:
        x = x.flip(-1)
    x = x.float() / 255.0
    x = TF.adjust_brightness(x, random.uniform(0.7, 1.3))
    x = TF.adjust_contrast(x, random.uniform(0.7, 1.3))
    x = TF.adjust_saturation(x, random.uniform(0.7, 1.3))
    if cctv:
        if random.random() < 0.2:
            x = TF.rgb_to_grayscale(x, num_output_channels=3)
        if random.random() < 0.25:
            lo = random.randint(56, 128)
            x = TF.resize(TF.resize(x, [lo, lo], antialias=True), [size, size], antialias=True)
        if random.random() < 0.2:
            x = TF.gaussian_blur(x, [5, 5], [random.uniform(0.3, 1.5)] * 2)
        if random.random() < 0.2:
            x = x + torch.randn_like(x) * random.uniform(0.01, 0.05)
    return x.clamp(0, 1)


def normalize(x):
    """float (T,3,H,W) in [0,1] -> ImageNet-normalised."""
    return (x - MEAN) / STD


def frames_to_tensor(frames_bgr, size=224):
    """List of BGR uint8 frames (any size) -> normalised (T,3,size,size) float."""
    arr = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames_bgr])
    x = torch.from_numpy(np.ascontiguousarray(arr[..., ::-1])).permute(0, 3, 1, 2).float() / 255.0
    return normalize(x)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    """Reads pre-extracted frames. Returns (clip (T,3,S,S), label, dataset_idx)."""

    def __init__(self, df, frames_root, num_frames=16, train=False, size=224, cctv_aug=True):
        self.ids = df.clip_id.tolist()
        self.labels = df.label.astype(float).tolist()
        self.ds_idx = [DATASETS.index(d) for d in df.dataset]
        self.root = frames_root
        self.num_frames = num_frames
        self.train = train
        self.size = size
        self.cctv_aug = cctv_aug

    def __len__(self):
        return len(self.ids)

    def _indices(self):
        T = STORE_FRAMES
        if self.train:  # random temporal extent => robustness to clip speed / window length
            span = random.randint(T // 2, T)
            start = random.randint(0, T - span)
            return np.linspace(start, start + span - 1, self.num_frames).round().astype(int)
        return np.linspace(0, T - 1, self.num_frames).round().astype(int)

    def __getitem__(self, i):
        d = os.path.join(self.root, self.ids[i])
        frames = [cv2.imread(os.path.join(d, f"{j:02d}.jpg")) for j in self._indices()]
        x = np.stack(frames)[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)
        if self.train:
            x = augment_clip(x, self.size, self.cctv_aug)
        else:
            x = TF.resize(x, [self.size, self.size], antialias=True).float() / 255.0
        return normalize(x), torch.tensor(self.labels[i]), self.ds_idx[i]
