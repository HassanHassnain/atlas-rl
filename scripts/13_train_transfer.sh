#!/usr/bin/env bash
# RUNTIME-ONLY: train both held-out-family checkpoints for the transfer study.
set -euo pipefail
PYTHON=${PYTHON:-python}

test -f data/generated/train.jsonl || PYTHON="$PYTHON" bash scripts/01_build_datasets.sh

"$PYTHON" -m atlas_rl.training.grpo_train --config configs/grpo_no_synthesis.yaml
"$PYTHON" -m atlas_rl.training.grpo_train --config configs/grpo_no_triage.yaml
