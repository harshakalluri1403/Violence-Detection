#!/usr/bin/env bash
# One-time VM setup: Python deps + the four public violence datasets from Kaggle.
# Safe to re-run: finished downloads are skipped.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

pip3 install --user -q --upgrade transformers opencv-python-headless scikit-learn pandas \
    kaggle matplotlib tqdm imageio-ffmpeg

RAW="${RAW:-$HOME/vd/raw}"
mkdir -p "$RAW"

dl() {  # dl <local-name> <kaggle-slug>
    local name=$1 slug=$2
    if [ -f "$RAW/$name/.done" ]; then echo "[skip] $name"; return; fi
    mkdir -p "$RAW/$name"
    echo "[download] $slug -> $RAW/$name"
    kaggle datasets download -d "$slug" -p "$RAW/$name" --unzip -q
    touch "$RAW/$name/.done"
}

dl hockey yassershrief/hockey-fight-vidoes
dl movies naveenk903/movies-fight-detection-dataset
dl rlvs   mohamedmustafa/real-life-violence-situations-dataset
dl rwf    vulamnguyen/rwf2000
du -sh "$RAW"/*
echo SETUP_DONE
