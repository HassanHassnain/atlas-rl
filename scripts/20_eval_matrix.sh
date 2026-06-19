#!/usr/bin/env bash
# Evaluate every model in configs/eval_matrix.yaml on the same held-out
# instances, then build the headline comparison report.
# Local models are RUNTIME-ONLY; the 32B baseline needs roughly 66 GB of
# aggregate GPU memory in full bf16.
set -euo pipefail
PYTHON=${PYTHON:-python}

CFG=${1:-configs/eval_matrix.yaml}
N=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['eval']['n_per_env'])")
DIFFS=$("$PYTHON" -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('$CFG'))['eval']['difficulties'])))")
TEMP=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['eval']['temperature'])")
TOPP=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['eval'].get('top_p', 0.95))")
MAXTOK=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['eval'].get('max_new_tokens', 768))")
K=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['eval'].get('k', 1))")
SEED_OFFSET=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['eval'].get('seed_offset', 0))")
BASE=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['baseline'])")
OUT_ROOT=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG')).get('output_root', 'results'))")

RUNS=()
while IFS=$'\t' read -r name spec; do
    echo "== evaluating $name  ($spec) =="
    "$PYTHON" -m atlas_rl.evaluation.run_eval --model "$spec" \
        --n-per-env "$N" --difficulties $DIFFS --temperature "$TEMP" \
        --top-p "$TOPP" --max-new-tokens "$MAXTOK" --k "$K" \
        --seed-offset "$SEED_OFFSET" \
        --out "$OUT_ROOT/$name"
    RUNS+=("$OUT_ROOT/$name")
done < <("$PYTHON" -c "
import yaml
for m in yaml.safe_load(open('$CFG'))['models']:
    print(f\"{m['name']}\t{m['spec']}\")")

"$PYTHON" -m atlas_rl.evaluation.report --runs "${RUNS[@]}" \
    --baseline "$OUT_ROOT/$BASE" --out "$OUT_ROOT/report"
echo "Headline report: $OUT_ROOT/report/REPORT.md"
