"""Contract 2: oracle scores ~1.0 and strict success; degenerate answers don't."""

import pytest

from atlas_rl import REGISTRY

ENVS = sorted(REGISTRY)
CASES = [(s, d) for s in [0, 3, 11, 42, 100] for d in [1, 2, 3, 4, 5]]


@pytest.mark.parametrize("env_id", ENVS)
def test_oracle_achieves_full_reward(env_id):
    env = REGISTRY[env_id]
    for seed, d in CASES:
        inst = env.generate(seed, d)
        rb = env.verify(inst, env.oracle(inst))
        assert rb.success, (f"{env_id} s{seed} d{d}: oracle not strict-success "
                            f"(total={rb.total}, components={rb.components})")
        assert rb.total >= 0.99, f"{env_id} s{seed} d{d}: oracle total {rb.total}"


@pytest.mark.parametrize("env_id", ENVS)
def test_untagged_oracle_is_semantically_successful_but_loses_tag_reward(env_id):
    from atlas_rl.core.protocol import extract_answer

    env = REGISTRY[env_id]
    inst = env.generate(17, 3)
    _, content = extract_answer(env.oracle(inst))
    rb = env.verify(inst, content)
    assert rb.correctness >= 0.99, (env_id, rb)
    assert rb.total < 1.0 and rb.success, (env_id, rb)


def test_regex_oracle_achieves_full_reward_across_generated_seeds():
    """Generated negatives must never accidentally satisfy the reference regex."""
    env = REGISTRY["regex_extract"]
    for seed in range(500):
        for difficulty in range(1, 6):
            inst = env.generate(seed, difficulty)
            rb = env.verify(inst, env.oracle(inst))
            assert rb.success, (
                f"regex_extract s{seed} d{difficulty}: family="
                f"{inst.ground_truth['family']} oracle failed ({rb.components})")


@pytest.mark.parametrize("env_id", ENVS)
def test_empty_response_scores_zero_ish(env_id):
    env = REGISTRY[env_id]
    inst = env.generate(0, 3)
    rb = env.verify(inst, "")
    assert rb.total <= 0.10 and not rb.success


@pytest.mark.parametrize("env_id", ENVS)
def test_reward_bounded(env_id):
    env = REGISTRY[env_id]
    inst = env.generate(5, 3)
    for resp in ["", "garbage", "<answer>{}</answer>", env.oracle(inst),
                 "<answer>[]</answer>", "<answer>* * * * *</answer>"]:
        rb = env.verify(inst, resp)
        assert 0.0 <= rb.total <= 1.0
        assert 0.0 <= rb.correctness <= 1.0


@pytest.mark.parametrize("env_id", ENVS)
def test_verify_does_not_mutate_instance(env_id):
    env = REGISTRY[env_id]
    inst = env.generate(9, 3)
    import copy
    gt_before = copy.deepcopy(inst.ground_truth)
    env.verify(inst, env.oracle(inst))
    env.verify(inst, "<answer>junk</answer>")
    assert inst.ground_truth == gt_before, f"{env_id}: verify mutated ground truth"
