#!/usr/bin/env bash
# Build the full training + evaluation datasets used by the real runs.
set -euo pipefail
PYTHON=${PYTHON:-python}

"$PYTHON" -m atlas_rl.data.build_dataset --split train --n-per-env 600 \
    --difficulty-mix 1:0.15 2:0.25 3:0.30 4:0.20 5:0.10 \
    --out data/generated/train.jsonl

"$PYTHON" -m atlas_rl.data.build_dataset --split eval --n-per-env 100 \
    --difficulties 2 3 4 --out data/generated/eval.jsonl

echo "datasets ready under data/generated/"
