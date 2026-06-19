# Atlas-RL

Atlas-RL is a benchmark and training pipeline for verifiable DevOps/SRE
reasoning tasks. It contains ten deterministic procedural environments,
programmatic reward functions, GRPO training, rejection-sampling SFT, and
paired held-out evaluation.

The project is designed around answers that can be checked by code rather than
graded by preference. Every environment has seeded instance generation, an
oracle, partial-credit reward, strict pass/fail scoring, and exploit canaries
for reward-hacking checks.

## Results

Headline numbers are from fixed blind seed ranges after model and configuration
selection.

| blind run | model | pass@1 [95% CI] | mean reward |
|---|---|---:|---:|
| Qwen2.5 | Qwen2.5-3B base | 23% [15, 31] | 0.303 |
| Qwen2.5 | Qwen2.5-3B stable GRPO | **80% [72, 87]** | **0.901** |
| Qwen2.5 | Qwen2.5-32B base | 81% [73, 88] | 0.889 |
| Qwen3 | Qwen3-4B base | 54% [44, 64] | 0.682 |
| Qwen3 | Qwen3-4B oracle SFT | **88% [81, 94]** | **0.939** |
| Qwen3 | Qwen3-4B stable GRPO | **88% [81, 94]** | **0.939** |
| Qwen3 | Qwen3-30B-A3B base | 67% [58, 76] | 0.808 |

The Qwen2.5 run shows a 57-point blind pass@1 gain over the 3B base model.
It does not support a claim that the trained 3B model beats the 32B dense
baseline: the final blind scores were 80% and 81%.

The Qwen3 run shows the same pipeline transferring to a newer 4B model family.
Both trained 4B checkpoints reached 88% pass@1 and beat the 30.5B-total /
3.3B-active MoE baseline on the sealed blind suite. Because that baseline
activates fewer parameters per token than the dense trained model, it is not a
larger-active-parameter comparison.

The reward-hacking audit catches 42,895 / 42,895 seeded exploit cases.

## Environments

Atlas-RL includes ten single-turn environments:

- `log_triage`: root-cause service and failure classification from logs
- `config_repair`: YAML config repair against a schema
- `ci_doctor`: CI failure triage
- `runbook_planner`: incident runbook ordering under preconditions
- `shell_golf`: one-line shell pipeline synthesis in a sandboxed VFS
- `cron_author`: natural-language scheduling to cron
- `regex_extract`: regex authoring with positive and negative examples
- `dockerfile_lint`: policy-violation detection
- `k8s_doctor`: Kubernetes manifest repair through restricted patch ops
- `semver_resolve`: dependency resolution under semver constraints

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the fast local checks:

```bash
make check
make smoke
python -m atlas_rl.demo --env shell_golf --difficulty 4
```

`make smoke` builds small generated datasets, runs mock evaluations, creates a
comparison report, builds a transfer matrix, and runs a short reward-hacking
audit. Generated outputs are ignored by Git.

## Reproduce

Build deterministic train and evaluation data:

```bash
bash scripts/01_build_datasets.sh
```

Install the training stack on a CUDA machine:

```bash
python -m pip install -r requirements-train.txt
```

Run the measured experiment entrypoints:

```bash
bash scripts/10_train_grpo_pilot_1p5b.sh
bash scripts/11_train_grpo_3b.sh
bash scripts/12_train_sft_rs.sh
bash scripts/13_train_transfer.sh
bash scripts/14_train_oracle_warm_grpo.sh
bash scripts/20_eval_matrix.sh
bash scripts/21_transfer_matrix.sh
bash scripts/22_hacking_audit.sh
```

Final blind protocols are captured in:

- `configs/final_blind_eval.yaml`
- `configs/qwen3_family/final_blind_eval.yaml`

Generated datasets, checkpoints, and reports live under `data/generated/`,
`checkpoints/`, `results/`, and `misc/`; these are intentionally excluded from
the repository.

## Design Contracts

- Train, development, audit, and final blind seed ranges are disjoint.
- `generate(seed, difficulty)` is deterministic.
- Strict success requires a parseable and semantically exact answer.
- Format reward does not count as task correctness.
- Every canary must score at most `0.25` and must not pass.
- Difficulty metadata is monotonic across levels 1-5.

## Layout

```text
atlas_rl/
  core/             protocol, types, seeding, registry
  envs/             procedural environments and verifiers
  training/         GRPO, reward adapter, rejection-sampling SFT
  evaluation/       evaluation, statistics, reports, transfer study
  rewards/          reward-hacking audit
  inference/        mock, Hugging Face, OpenAI-compatible backends
configs/            experiment configs and blind protocols
scripts/            numbered experiment entrypoints
tests/              verifier and pipeline contracts
```

## License

Atlas-RL is released under the MIT License.
