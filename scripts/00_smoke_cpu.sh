#!/usr/bin/env bash
# CPU-only smoke test: proves the entire pipeline (envs -> dataset -> eval ->
# report -> audit) works before any GPU is involved. Run from repo root.
set -euo pipefail
PYTHON=${PYTHON:-python}

echo "== 1/6 unit + contract tests =="
"$PYTHON" -m pytest -q

echo "== 2/6 build a small dataset =="
"$PYTHON" -m atlas_rl.data.build_dataset --split train --n-per-env 40 \
    --out data/generated/train_smoke.jsonl
"$PYTHON" -m atlas_rl.data.build_dataset --split eval --n-per-env 10 \
    --difficulties 2 3 4 --out data/generated/eval_smoke.jsonl

echo "== 3/6 mock evaluations (oracle / noisy / format-only) =="
"$PYTHON" -m atlas_rl.evaluation.run_eval --model mock:oracle          --n-per-env 6 --difficulties 2 3 4 --out results/smoke_oracle
"$PYTHON" -m atlas_rl.evaluation.run_eval --model mock:noisy_oracle:0.6 --n-per-env 6 --difficulties 2 3 4 --out results/smoke_noisy
"$PYTHON" -m atlas_rl.evaluation.run_eval --model mock:format_only     --n-per-env 6 --difficulties 2 3 4 --out results/smoke_format

echo "== 4/6 comparison report with paired stats =="
"$PYTHON" -m atlas_rl.evaluation.report \
    --runs results/smoke_oracle results/smoke_noisy results/smoke_format \
    --baseline results/smoke_format --out results/smoke_report

echo "== 5/6 transfer matrix =="
"$PYTHON" -m atlas_rl.evaluation.transfer_matrix \
    --config configs/transfer_smoke.yaml --out results/smoke_transfer

echo "== 6/6 reward-hacking audit =="
"$PYTHON" -m atlas_rl.rewards.hacking_audit --seeds 10 --out results/smoke_audit

echo "SMOKE TEST PASSED - pipeline is healthy. See results/."
