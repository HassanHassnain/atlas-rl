#!/usr/bin/env bash
# RUNTIME-ONLY (RTX 3090 box). Main GRPO run: Qwen2.5-3B QLoRA.
# Single GPU: just run this. Two GPUs: see the vllm section in the config —
# start `CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct`
# first, uncomment the vllm block, then CUDA_VISIBLE_DEVICES=1 bash this script.
# Output: checkpoints/grpo_3b/final
set -euo pipefail
PYTHON=${PYTHON:-python}
test -f data/generated/train.jsonl || PYTHON="$PYTHON" bash scripts/01_build_datasets.sh
"$PYTHON" -m atlas_rl.training.grpo_train --config configs/grpo_qwen3b_3090.yaml
