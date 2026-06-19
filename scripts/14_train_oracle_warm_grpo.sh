#!/usr/bin/env bash
# Strong curriculum: deterministic oracle SFT followed by verifier-reward GRPO.
set -euo pipefail
PYTHON=${PYTHON:-python}

test -f data/generated/train.jsonl || PYTHON="$PYTHON" bash scripts/01_build_datasets.sh
"$PYTHON" -m atlas_rl.training.sft_rs_baseline \
    --config configs/sft_oracle_qwen3b.yaml --stage oracle
"$PYTHON" -m atlas_rl.training.sft_rs_baseline \
    --config configs/sft_oracle_qwen3b.yaml --stage train
"$PYTHON" -m atlas_rl.training.grpo_train \
    --config configs/grpo_oracle_warm_qwen3b.yaml
