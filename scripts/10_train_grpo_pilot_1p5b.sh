#!/usr/bin/env bash
# RUNTIME-ONLY (RTX 3090 box). Pilot GRPO on Qwen2.5-1.5B — validates the loop.
# Expected: reward curve climbs within ~50 steps; per-env success telemetry
# printed every ~20 reward calls. Output: checkpoints/grpo_1p5b_pilot/final
set -euo pipefail
PYTHON=${PYTHON:-python}
test -f data/generated/train.jsonl || PYTHON="$PYTHON" bash scripts/01_build_datasets.sh
"$PYTHON" -m atlas_rl.training.grpo_train --config configs/grpo_qwen1p5b_pilot.yaml
