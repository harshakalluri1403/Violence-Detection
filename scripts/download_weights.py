#!/usr/bin/env python
"""Download the trained checkpoints from the GitHub release into checkpoints/.

    python scripts/download_weights.py            # base + small
    python scripts/download_weights.py --only base
"""
import argparse
import os
import sys
import urllib.request

RELEASE = "https://github.com/harshakalluri1403/Violence-Detection/releases/download/v2.0.0"
WEIGHTS = {"base": "vdet_base.pt", "small": "vdet_small.pt"}


def fetch(url, dst):
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {os.path.basename(dst)}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    os.replace(tmp, dst)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(WEIGHTS))
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for key, name in WEIGHTS.items():
        if args.only and key != args.only:
            continue
        dst = os.path.join(args.dir, name)
        if os.path.exists(dst):
            print(f"  {name}: already present")
            continue
        fetch(f"{RELEASE}/{name}", dst)


if __name__ == "__main__":
    main()
