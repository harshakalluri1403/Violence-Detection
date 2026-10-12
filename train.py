#!/usr/bin/env python
"""Fine-tune a Kinetics-pretrained VideoMAE for violence detection.

    python train.py --name vmae_b_mixed --train-sets hockey,movies,rwf,rlvs

The run directory gets model.pt (fp16 weights), metrics.json, preds.csv and a
per-epoch log. Training resumes from runs/<name>/last.pt if it exists, so a
preempted spot VM can simply re-run the same command.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from vdet.data import DATASETS, ClipDataset, apply_test_fold
from vdet.metrics import binary_metrics
from vdet.models import build_model, layer_decay_param_groups, save_checkpoint


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--train-sets", default="hockey,movies,rwf,rlvs")
    ap.add_argument("--test-sets", default="hockey,movies,rwf,rlvs")
    ap.add_argument("--fold", type=int, default=0, help="test fold for hockey/movies/rlvs")
    ap.add_argument("--arch", default="base", help="base | small | any HF VideoMAE id")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup epochs")
    ap.add_argument("--no-cctv-aug", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N optimizer steps")
    return ap.parse_args()


def split_train_val(df, train_sets, seed):
    pool = df[(df.split == "train") & df.dataset.isin(train_sets)]
    strat = pool.dataset + "_" + pool.label.astype(str)
    return train_test_split(pool, test_size=0.1, stratify=strat, random_state=seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logit = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(probs).numpy()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.join(args.runs, args.name)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print(f"[{args.name}] already finished, skipping")
        return
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    df = pd.read_csv(args.index)
    if args.fold:
        df = apply_test_fold(df, args.fold)
    train_sets = args.train_sets.split(",")
    test_sets = [s for s in args.test_sets.split(",") if s]
    tr, va = split_train_val(df, train_sets, args.seed)
    print(f"[{args.name}] train={len(tr)} val={len(va)} sets={train_sets}")

    train_ds = ClipDataset(tr, args.frames, train=True, cctv_aug=not args.no_cctv_aug)
    if len(train_sets) > 1:
        # Each dataset is drawn with probability ~ sqrt(size) so the 200-clip
        # Movies set is not drowned out by the 2000-clip ones.
        counts = tr.dataset.value_counts()
        w = tr.dataset.map(lambda d: counts[d] ** -0.5).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
        train_loader = DataLoader(train_ds, args.bs, sampler=sampler, num_workers=args.workers,
                                  pin_memory=True, drop_last=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers,
                                  pin_memory=True, drop_last=True, persistent_workers=True)
    eval_kw = dict(batch_size=args.bs * 2, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(ClipDataset(va, args.frames), **eval_kw)

    model = build_model(args.arch).to(device)
    if args.grad_ckpt:
        model.net.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(layer_decay_param_groups(model, args.wd, args.layer_decay), lr=args.lr)

    steps_per_epoch = len(train_loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(steps_per_epoch * args.warmup))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total_steps - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    start_epoch, step = 0, 0
    last_path = os.path.join(run_dir, "last.pt")
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch, step = state["epoch"], state["step"]
        print(f"[{args.name}] resumed at epoch {start_epoch}")

    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            (loss / args.accum).backward()
            loss_sum += loss.item() * len(y)
            seen += len(y)
            if (i + 1) % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = args.lr * g["lr_scale"] * lr_factor(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"  ep {epoch} step {step}/{total_steps} loss {loss_sum / seen:.4f} "
                          f"{seen / (time.time() - t0):.1f} clips/s", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
        val_p = predict(model, val_loader, device)
        val_m = binary_metrics(va.label, val_p)
        rec = dict(epoch=epoch + 1, step=step, train_loss=loss_sum / max(seen, 1),
                   clips_per_s=seen / (time.time() - t0), val=val_m)
        print(f"[{args.name}] epoch {epoch + 1}/{args.epochs} loss {rec['train_loss']:.4f} "
              f"val acc {val_m['acc']:.4f} auc {val_m['auc']:.4f}", flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "step": step}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        if args.max_steps and step >= args.max_steps:
            break

    # ---- final evaluation on every held-out test set --------------------- #
    results = {"val": binary_metrics(va.label, predict(model, val_loader, device))}
    pred_rows = []
    for ds in test_sets:
        te = df[(df.split == "test") & (df.dataset == ds)]
        if te.empty:
            continue
        p = predict(model, DataLoader(ClipDataset(te, args.frames), **eval_kw), device)
        results[ds] = binary_metrics(te.label, p)
        pred_rows.append(pd.DataFrame({"clip_id": te.clip_id, "dataset": ds, "label": te.label, "prob": p}))
        print(f"[{args.name}] test {ds:7s} acc {results[ds]['acc']:.4f} auc {results[ds]['auc']:.4f}")
    if pred_rows:
        pd.concat(pred_rows).to_csv(os.path.join(run_dir, "preds.csv"), index=False)
    meta = {"arch": args.arch, "num_frames": model.num_frames, "train_sets": train_sets,
            "name": args.name, "fold": args.fold}
    save_checkpoint(model, os.path.join(run_dir, "model.pt"), meta)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    if os.path.exists(last_path):
        os.remove(last_path)


if __name__ == "__main__":
    main()
