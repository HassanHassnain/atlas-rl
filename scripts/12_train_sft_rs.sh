#!/usr/bin/env bash
# RUNTIME-ONLY (RTX 3090 box). SFT-from-rejection-sampling baseline.
# Stage 1 samples from the base model (slow, ~hours); stage 2 trains LoRA.
# Outputs: data/generated/sft_rs_3b.jsonl (+ .stats.json with per-env
# acceptance rates = base-model pass@8) and checkpoints/sft_rs_3b/final
set -euo pipefail
PYTHON=${PYTHON:-python}
test -f data/generated/train.jsonl || PYTHON="$PYTHON" bash scripts/01_build_datasets.sh
"$PYTHON" -m atlas_rl.training.sft_rs_baseline --config configs/sft_rs_qwen3b.yaml --stage sample
"$PYTHON" -m atlas_rl.training.sft_rs_baseline --config configs/sft_rs_qwen3b.yaml --stage train
