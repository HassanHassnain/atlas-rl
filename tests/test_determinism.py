"""Contract 1: byte-identical regeneration; distinct seeds -> distinct instances."""

import pytest

from atlas_rl import REGISTRY

ENVS = sorted(REGISTRY)
SEEDS = [0, 1, 7, 12345]
DIFFS = [1, 3, 5]


@pytest.mark.parametrize("env_id", ENVS)
def test_regeneration_identical(env_id):
    env = REGISTRY[env_id]
    for seed in SEEDS:
        for d in DIFFS:
            a = env.generate(seed, d)
            b = env.generate(seed, d)
            assert a.prompt == b.prompt, f"{env_id} s{seed} d{d}: prompt differs"
            assert a.ground_truth == b.ground_truth
            assert a.metadata == b.metadata


@pytest.mark.parametrize("env_id", ENVS)
def test_seeds_produce_variety(env_id):
    env = REGISTRY[env_id]
    prompts = {env.generate(s, 3).prompt for s in range(20)}
    assert len(prompts) >= 18, f"{env_id}: only {len(prompts)}/20 unique prompts"


@pytest.mark.parametrize("env_id", ENVS)
def test_train_eval_seed_disjointness(env_id):
    from atlas_rl.core.seeding import EVAL_SEED_BASE

    env = REGISTRY[env_id]
    train = {env.generate(s, 3).prompt for s in range(15)}
    evalp = {env.generate(EVAL_SEED_BASE + s, 3).prompt for s in range(15)}
    assert not train & evalp, f"{env_id}: train/eval instance collision"


def test_reserved_seed_ranges_are_disjoint():
    from atlas_rl.core.seeding import (
        AUDIT_SEED_BASE,
        EVAL_SEED_BASE,
        FINAL_TEST_SEED_BASE,
        TRAIN_SEED_BASE,
    )

    assert [TRAIN_SEED_BASE, EVAL_SEED_BASE, AUDIT_SEED_BASE, FINAL_TEST_SEED_BASE] == [
        0, 1_000_000, 2_000_000, 4_000_000
    ]


@pytest.mark.parametrize("env_id", ENVS)
def test_ground_truth_json_serializable(env_id):
    import json

    env = REGISTRY[env_id]
    inst = env.generate(3, 4)
    json.dumps(inst.to_dict())  # must not raise
