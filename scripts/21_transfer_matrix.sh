#!/usr/bin/env bash
# Cross-environment generalization study. RUNTIME-ONLY for real checkpoints.
# Train the subset checkpoints first (same config, dataset.envs restricted):
#   python -m atlas_rl.training.grpo_train --config configs/grpo_qwen3b_3090.yaml  # all envs
#   then two runs with dataset.envs edited per configs/transfer.yaml comments,
#   output_dir checkpoints/grpo_no_synthesis and checkpoints/grpo_no_triage.
set -euo pipefail
PYTHON=${PYTHON:-python}
"$PYTHON" -m atlas_rl.evaluation.transfer_matrix --config configs/transfer.yaml \
    --out results/transfer
