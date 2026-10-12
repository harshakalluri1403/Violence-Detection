#!/usr/bin/env python
"""Baselines evaluated on exactly the same splits as the VideoMAE runs.

    python scripts/baselines.py vgg_lstm   # the original project's approach
    python scripts/baselines.py clip       # zero-shot CLIP ViT-L/14, no training

vgg_lstm re-implements the Kaggle VGG16 -> LSTM notebook (frozen ImageNet
VGG16 fc2 features per frame, LSTM(512) -> Dense(1024) -> Dense(50)), but with
16 evenly spaced frames and proper ImageNet preprocessing. It trains one model
per training configuration in CONFIGS so it can be compared cell-by-cell with
the transformer.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import split_train_val  # noqa: E402
from vdet.data import DATASETS, ClipDataset  # noqa: E402
from vdet.metrics import binary_metrics  # noqa: E402

CONFIGS = {
    "hockey": ["hockey"],
    "rwf": ["rwf"],
    "rlvs": ["rlvs"],
    "mixed": DATASETS,
    "lodo_hockey": ["movies", "rwf", "rlvs"],
    "lodo_movies": ["hockey", "rwf", "rlvs"],
    "lodo_rwf": ["hockey", "movies", "rlvs"],
    "lodo_rlvs": ["hockey", "movies", "rwf"],
}


def write_run(run_dir, name, train_sets, df, probs, val=None, threshold=0.5):
    os.makedirs(run_dir, exist_ok=True)
    results, rows = {}, []
    if val is not None:
        results["val"] = val
    for ds in DATASETS:
        te = df[(df.split == "test") & (df.dataset == ds)]
        p = probs[te.index.to_numpy()]
        results[ds] = binary_metrics(te.label, p, threshold)
        rows.append(pd.DataFrame({"clip_id": te.clip_id, "dataset": ds, "label": te.label, "prob": p}))
    pd.concat(rows).to_csv(os.path.join(run_dir, "preds.csv"), index=False)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"meta": {"name": name, "train_sets": train_sets, "threshold": threshold},
                   "results": results}, f, indent=2)
    print(name, {k: round(v["acc"], 4) for k, v in results.items()})


# --------------------------------------------------------------------------- #
# VGG16 + LSTM
# --------------------------------------------------------------------------- #
@torch.no_grad()
def vgg_features(df, frames, device, cache):
    if os.path.exists(cache):
        return np.load(cache)
    from torchvision.models import VGG16_Weights, vgg16
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)
    fc2 = nn.Sequential(vgg.features, vgg.avgpool, nn.Flatten(), *list(vgg.classifier[:5])).half()
    loader = DataLoader(ClipDataset(df, frames), batch_size=8, num_workers=7)
    out = []
    for x, _, _ in loader:
        b, t = x.shape[:2]
        f = fc2(x.flatten(0, 1).to(device).half()).view(b, t, -1)
        out.append(f.cpu().numpy())
    feats = np.concatenate(out).astype(np.float16)
    np.save(cache, feats)
    return feats


class LSTMHead(nn.Module):
    """Same layers as the original notebook: LSTM(512) -> Dense(1024, relu) ->
    Dense(50, sigmoid) -> output (BCE on one logit instead of MSE on softmax)."""

    def __init__(self, d_in=4096):
        super().__init__()
        self.lstm = nn.LSTM(d_in, 512, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 50), nn.Sigmoid(),
                                 nn.Linear(50, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.mlp(h[:, -1]).squeeze(-1)


def train_lstm(feats, tr, va, device, epochs=60, seed=0):
    torch.manual_seed(seed)
    model = LSTMHead(feats.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    Xtr = torch.from_numpy(feats[tr.index.to_numpy()]).float().to(device)
    ytr = torch.tensor(tr.label.to_numpy(), dtype=torch.float32, device=device)
    Xva = torch.from_numpy(feats[va.index.to_numpy()]).float().to(device)
    best, best_state = -1, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva)).cpu().numpy()
        auc = binary_metrics(va.label, pv)["auc"]
        if auc > best:  # model selection on the validation split only
            best = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def run_vgg_lstm(args, df, device):
    feats = vgg_features(df, args.frames, device, os.path.join(os.path.dirname(args.index), "vgg16_fc2.npy"))
    X = torch.from_numpy(feats).float()
    for cfg, sets in CONFIGS.items():
        run_dir = os.path.join(args.runs, f"vgg_lstm_{cfg}")
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            continue
        tr, va = split_train_val(df, sets, 0)
        model = train_lstm(feats, tr, va, device)
        with torch.no_grad():
            probs = torch.cat([torch.sigmoid(model(X[i:i + 256].to(device))).cpu()
                               for i in range(0, len(X), 256)]).numpy()
        val = binary_metrics(va.label, probs[va.index.to_numpy()])
        write_run(run_dir, f"vgg_lstm_{cfg}", sets, df, probs, val)


# --------------------------------------------------------------------------- #
# CLIP zero-shot
# --------------------------------------------------------------------------- #
VIOLENT = ["a photo of people fighting", "a photo of a violent fight", "people punching and kicking each other",
           "a CCTV image of a fight", "a brawl between people", "a person assaulting another person"]
NORMAL = ["a photo of people doing everyday activities", "a photo of people walking calmly",
          "a CCTV image of a normal scene", "people playing sports", "people talking to each other",
          "a peaceful crowd"]


@torch.no_grad()
def run_clip(args, df, device):
    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(name).to(device).half().eval()
    tok = CLIPTokenizer.from_pretrained(name)

    def text_emb(prompts):
        t = tok(prompts, padding=True, return_tensors="pt").to(device)
        e = model.get_text_features(**t)
        e = getattr(e, "pooler_output", e)
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)
        return e / e.norm()

    T = torch.stack([text_emb(VIOLENT), text_emb(NORMAL)])
    imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    clip_mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)
    # Score every clip: the test sets for the results, the validation clips to pick
    # a threshold (CLIP's raw 0.5 cut-off is arbitrary; AUC is unaffected).
    loader = DataLoader(ClipDataset(df, args.frames, num_frames=8), batch_size=8, num_workers=3)
    probs = []
    for x, _, _ in loader:
        b, t = x.shape[:2]
        x = x.flatten(0, 1) * imnet_std + imnet_mean  # undo ImageNet norm
        x = ((x - clip_mean) / clip_std).to(device).half()
        e = model.get_image_features(pixel_values=x)
        e = getattr(e, "pooler_output", e)
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.view(b, t, -1).mean(1)
        e = e / e.norm(dim=-1, keepdim=True)
        probs.append((100 * e @ T.T).float().softmax(-1)[:, 0].cpu())
    p_all = torch.cat(probs).numpy()
    _, va = split_train_val(df, DATASETS, 0)
    pv, yv = p_all[va.index.to_numpy()], va.label.to_numpy()
    cands = np.unique(pv)
    th = float(cands[np.argmax([((pv >= t) == yv).mean() for t in cands])])
    write_run(os.path.join(args.runs, "clip_zeroshot"), "clip_zeroshot", [], df, p_all,
              val=binary_metrics(yv, pv, th), threshold=th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["vgg_lstm", "clip"])
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    df = pd.read_csv(args.index)
    device = "cuda"
    if args.which == "vgg_lstm":
        run_vgg_lstm(args, df, device)
    else:
        run_clip(args, df, device)


if __name__ == "__main__":
    main()
