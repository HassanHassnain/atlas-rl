"""Contract 3: every seeded exploit (canary) is caught by the verifier."""

import pytest

from atlas_rl import REGISTRY
from atlas_rl.core.seeding import AUDIT_SEED_BASE
from atlas_rl.core.types import CAUGHT_THRESHOLD

ENVS = sorted(REGISTRY)


@pytest.mark.parametrize("env_id", ENVS)
def test_all_canaries_caught(env_id):
    env = REGISTRY[env_id]
    failures = []
    for i in range(8):
        for d in (1, 3, 5):
            inst = env.generate(AUDIT_SEED_BASE + i, d)
            for c in env.canaries(inst):
                rb = env.verify(inst, c.response)
                if rb.total > CAUGHT_THRESHOLD or rb.success:
                    failures.append(
                        f"s{inst.seed} d{d} canary={c.name} total={rb.total:.3f} "
                        f"success={rb.success}")
    assert not failures, f"{env_id}: uncaught canaries:\n" + "\n".join(failures[:12])


def test_audit_module_end_to_end():
    from atlas_rl.rewards.hacking_audit import run_audit

    audit = run_audit(n_seeds=3, difficulties=(1, 4))
    s = audit["summary"]
    assert s["n_cases"] > 0
    assert s["catch_rate"] >= 0.95, f"catch rate {s['catch_rate']} below FYP target"


def test_format_only_capped_globally():
    for env_id in ENVS:
        env = REGISTRY[env_id]
        inst = env.generate(AUDIT_SEED_BASE, 3)
        rb = env.verify(inst, "<answer>the fix is straightforward</answer>")
        assert rb.total <= 0.15, f"{env_id}: format-only reward {rb.total}"


def test_cron_canaries_are_true_exploits_across_generated_seeds():
    env = REGISTRY["cron_author"]
    for i in range(100):
        for difficulty in range(1, 6):
            inst = env.generate(AUDIT_SEED_BASE + i, difficulty)
            for canary in env.extra_canaries(inst):
                rb = env.verify(inst, canary.response)
                assert rb.total <= CAUGHT_THRESHOLD and not rb.success, (
                    f"cron_author s{inst.seed} d{difficulty}: "
                    f"{canary.name} earned {rb.total}")
