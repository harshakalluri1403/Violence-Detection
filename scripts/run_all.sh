#!/usr/bin/env bash
# Full experiment pipeline (runs on the GPU VM). Every step skips work that is
# already finished, so after a spot preemption just run it again.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ALL=hockey,movies,rwf,rlvs
EP_MULTI=${EP_MULTI:-12}
EP_SINGLE=${EP_SINGLE:-15}

[ -f data/index.csv ] || python3 scripts/prepare_data.py --raw "$HOME/vd/raw" --out data || exit 1

run() { python3 train.py "$@" || echo "FAILED: $*"; }

# 1. Main model + one model per dataset (cross-dataset generalisation matrix)
run --name vmae_b_mixed  --train-sets $ALL   --epochs "$EP_MULTI"
run --name vmae_b_hockey --train-sets hockey --epochs "$EP_SINGLE"
run --name vmae_b_rwf    --train-sets rwf    --epochs "$EP_SINGLE"
run --name vmae_b_rlvs   --train-sets rlvs   --epochs "$EP_SINGLE"

# 2. Baselines on identical splits
[ -f runs/clip_zeroshot/metrics.json ] || python3 scripts/baselines.py clip
python3 scripts/baselines.py vgg_lstm

# 3. Long-video streaming benchmark
python3 scripts/make_longbench.py --videos-per-source 25
python3 scripts/eval_longbench.py --models vmae_b_mixed,vmae_b_hockey

# 4. Leave-one-dataset-out: accuracy on a dataset never seen in training
run --name vmae_b_lodo_rwf    --train-sets hockey,movies,rlvs --epochs "$EP_MULTI"
run --name vmae_b_lodo_rlvs   --train-sets hockey,movies,rwf  --epochs "$EP_MULTI"
run --name vmae_b_lodo_hockey --train-sets movies,rwf,rlvs    --epochs "$EP_MULTI"
run --name vmae_b_lodo_movies --train-sets hockey,rwf,rlvs    --epochs "$EP_MULTI"

# 5. Lite model for CPU / edge, and the CCTV-augmentation ablation
run --name vmae_s_mixed --arch small --train-sets $ALL --epochs "$EP_MULTI" --bs 16 --accum 1
python3 scripts/eval_longbench.py --models vmae_s_mixed
run --name vmae_b_mixed_nocctv --train-sets $ALL --epochs "$EP_MULTI" --no-cctv-aug
python3 scripts/eval_longbench.py --models vmae_b_mixed_nocctv

touch runs/ALL_DONE
echo PIPELINE_DONE
