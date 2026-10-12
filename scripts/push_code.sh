#!/usr/bin/env bash
# Copy the code (not data/runs) to the training VM: scripts/push_code.sh [zone]
# run_all.sh is skipped (bash reads a running script lazily, so overwriting it
# mid-run corrupts the pipeline); set PUSH_RUNNER=1 to include it.
set -euo pipefail
ZONE=${1:-us-central1-c}
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
EXTRA=(); [ "${PUSH_RUNNER:-0}" = 1 ] || EXTRA=(--exclude=scripts/run_all.sh)
tar czf "$TMP/code.tgz" "${EXTRA[@]}" --exclude=data --exclude=runs --exclude=out --exclude=checkpoints \
    --exclude=results --exclude=__pycache__ --exclude='*.pt' vdet scripts train.py detect.py app.py requirements.txt
gcloud compute scp --zone "$ZONE" --quiet "$(cygpath -w "$TMP/code.tgz" 2>/dev/null || echo "$TMP/code.tgz")" violence-l4:vd/code.tgz
gcloud compute ssh violence-l4 --zone "$ZONE" --quiet --command "cd ~/vd && tar xzf code.tgz && rm code.tgz && echo PUSHED"
rm -rf "$TMP"
