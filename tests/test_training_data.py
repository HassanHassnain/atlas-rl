"""Small, dependency-free tests for training data selection."""

import json
from collections import Counter
from types import SimpleNamespace

from atlas_rl.training.grpo_train import write_reward_stats
from atlas_rl.data.build_dataset import build_rows
from atlas_rl.training.sft_rs_baseline import build_oracle_rows, stratified_limit


def test_stratified_limit_balances_environment_coverage():
    rows = [
        {"env_id": env_id, "seed": seed}
        for env_id in ("z_env", "a_env", "m_env")
        for seed in range(10)
    ]

    selected = stratified_limit(rows, 8)
    counts = Counter(row["env_id"] for row in selected)

    assert len(selected) == 8
    assert counts == {"a_env": 3, "m_env": 3, "z_env": 2}


def test_stratified_limit_preserves_full_dataset_when_uncapped():
    rows = [{"env_id": "env", "seed": seed} for seed in range(3)]
    assert stratified_limit(rows, None) is rows
    assert stratified_limit(rows, 10) is rows


def test_oracle_rows_are_balanced_and_strict_successes():
    rows = build_rows(["ci_doctor", "cron_author"], "train", 4, difficulties=[2, 3])
    out = build_oracle_rows(rows, 4)
    assert Counter(row["env_id"] for row in out) == {"ci_doctor": 2, "cron_author": 2}
    assert all(row["messages"][-1]["role"] == "assistant" for row in out)
    assert all("<answer>" in row["messages"][-1]["content"] for row in out)


def test_reward_stats_are_persisted(tmp_path):
    stats = SimpleNamespace(
        totals=[0.2, 0.6],
        succ=[0.0, 1.0],
        flags=Counter({"parse_error": 3}),
    )

    write_reward_stats(str(tmp_path), stats)
    saved = json.loads((tmp_path / "reward_stats.json").read_text())

    assert saved == {
        "mean_reward_last512": 0.4,
        "success_last512": 0.5,
        "hack_flags": {"parse_error": 3},
        "n_reward_calls": 2,
    }
