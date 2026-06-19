"""End-to-end pipeline tests on CPU: dataset build, mock eval, report, transfer."""

import json
import os

from atlas_rl import REGISTRY
from atlas_rl.data.build_dataset import build_rows
from atlas_rl.evaluation.run_eval import evaluate, summarize
from atlas_rl.evaluation.report import build_report
from atlas_rl.evaluation.stats import bootstrap_ci, paired_bootstrap_diff
from atlas_rl.evaluation.transfer_matrix import render, run_transfer

ENVS = sorted(REGISTRY)


def test_dataset_builder_schema_and_disjointness():
    train = build_rows(ENVS[:3], "train", 8)
    evalr = build_rows(ENVS[:3], "eval", 8, difficulties=[2, 3])
    assert len(train) == 24 and len(evalr) == 24
    for r in train + evalr:
        assert r["prompt"][0]["role"] == "system"
        assert r["prompt"][1]["role"] == "user"
        assert r["env_id"] in REGISTRY
        json.dumps(r)
    train_ids = {r["instance_id"] for r in train}
    eval_ids = {r["instance_id"] for r in evalr}
    assert not train_ids & eval_ids


def test_mock_oracle_eval_is_perfect():
    res = evaluate("mock:oracle", ENVS, n_per_env=4, difficulties=[2, 4],
                   progress=False)
    s = summarize(res)
    assert s["overall"]["pass1"] == 1.0
    assert s["overall"]["mean_reward"] >= 0.99


def test_mock_format_only_eval_is_near_zero():
    res = evaluate("mock:format_only", ENVS, n_per_env=3, difficulties=[3],
                   progress=False)
    s = summarize(res)
    assert s["overall"]["pass1"] == 0.0
    assert s["overall"]["mean_reward"] <= 0.15


def test_noisy_oracle_sits_in_between():
    res = evaluate("mock:noisy_oracle:0.5", ENVS[:4], n_per_env=10,
                   difficulties=[3], progress=False)
    s = summarize(res)
    assert 0.2 <= s["overall"]["pass1"] <= 0.8


def test_report_generation(tmp_path):
    for name, spec in [("a", "mock:oracle"), ("b", "mock:format_only")]:
        res = evaluate(spec, ENVS[:3], n_per_env=3, difficulties=[2], progress=False)
        s = summarize(res)
        d = tmp_path / name
        os.makedirs(d)
        with open(d / "rows.jsonl", "w") as f:
            for r in res["rows"]:
                f.write(json.dumps(r) + "\n")
        with open(d / "summary.json", "w") as f:
            json.dump(s, f)
    md = build_report([str(tmp_path / "a"), str(tmp_path / "b")],
                      str(tmp_path / "b"))
    assert "mock:oracle" in md and "Δ pass@1" in md
    assert "+100.0 pts" in md  # oracle beats format_only by exactly 100 points
    assert "Δ reward" in md and "Per-environment mean reward" in md


def test_transfer_includes_strict_and_partial_credit():
    out = run_transfer({
        "envs": ENVS[:2],
        "n_per_env": 2,
        "difficulties": [2],
        "models": [
            {"name": "oracle", "spec": "mock:oracle", "trained_on": ENVS[:1]},
            {"name": "format", "spec": "mock:format_only", "trained_on": []},
        ],
    })
    assert out["matrix"]["oracle"]["pass1_by_env"]
    assert out["matrix"]["oracle"]["reward_by_env"]
    md = render(out["matrix"], out["envs"])
    assert "Strict pass@1" in md and "Mean verifier reward" in md


def test_stats_sanity():
    m, lo, hi = bootstrap_ci([1.0] * 10)
    assert m == lo == hi == 1.0
    m, lo, hi = bootstrap_ci([0, 1] * 20)
    assert lo < 0.5 < hi
    t = paired_bootstrap_diff([1.0] * 30, [0.0] * 30)
    assert t["mean_diff"] == 1.0 and t["p_better"] == 1.0


def test_reward_adapter_matches_direct_verify():
    from atlas_rl.training.reward_adapter import make_reward_fn

    fn = make_reward_fn()
    env = REGISTRY["cron_author"]
    insts = [env.generate(s, 2) for s in range(3)]
    completions = [env.oracle(i) for i in insts]
    rewards = fn(prompts=None, completions=completions,
                 env_id=["cron_author"] * 3, seed=[i.seed for i in insts],
                 difficulty=[2, 2, 2])
    assert all(r >= 0.99 for r in rewards)
    # message-list completions (TRL conversational format) also work
    rewards2 = fn(prompts=None,
                  completions=[[{"role": "assistant", "content": c}] for c in completions],
                  env_id=["cron_author"] * 3, seed=[i.seed for i in insts],
                  difficulty=[2, 2, 2])
    assert rewards == rewards2
