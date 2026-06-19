#!/usr/bin/env bash
# Reward-hacking audit — a first-class deliverable. CPU-only.
set -euo pipefail
PYTHON=${PYTHON:-python}
"$PYTHON" -m atlas_rl.rewards.hacking_audit --seeds 100 --difficulties 1 2 3 4 5 \
    --out results/audit
