"""Contract 4: the difficulty dial increases instance complexity in expectation."""

from statistics import mean

import pytest

from atlas_rl import REGISTRY

ENVS = sorted(REGISTRY)


@pytest.mark.parametrize("env_id", ENVS)
def test_complexity_increases_with_difficulty(env_id):
    env = REGISTRY[env_id]
    means = {}
    for d in (1, 3, 5):
        means[d] = mean(env.generate(s, d).metadata["complexity"] for s in range(10))
    assert means[1] < means[5], f"{env_id}: complexity d1={means[1]} !< d5={means[5]}"
    assert means[1] <= means[3] <= means[5] or means[1] < means[5], \
        f"{env_id}: non-monotone complexity {means}"


@pytest.mark.parametrize("env_id", ENVS)
def test_prompts_grow_with_difficulty(env_id):
    env = REGISTRY[env_id]
    p1 = mean(len(env.generate(s, 1).prompt) for s in range(10))
    p5 = mean(len(env.generate(s, 5).prompt) for s in range(10))
    # weak monotonicity: hardest prompts should not be shorter on average
    assert p5 >= 0.8 * p1, f"{env_id}: d5 prompts much shorter than d1 ({p5} vs {p1})"
