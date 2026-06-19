"""Bridges Atlas-RL verifiers to TRL reward functions.

TRL's GRPOTrainer calls reward functions as
    fn(prompts=..., completions=..., **dataset_columns_as_lists)
so our dataset rows carry (env_id, seed, difficulty) and the reward function
regenerates the exact instance deterministically — no serialized state to
trust, no train/verify skew.
"""

from __future__ import annotations

from collections import Counter

from atlas_rl import REGISTRY

_CACHE: dict = {}
_CACHE_MAX = 50_000


def completion_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return "".join(m.get("content", "") for m in c if m.get("role") == "assistant") \
            or c[0].get("content", "")
    return str(c)


def get_instance(env_id: str, seed: int, difficulty: int):
    key = (env_id, seed, difficulty)
    inst = _CACHE.get(key)
    if inst is None:
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        inst = REGISTRY[env_id].generate(int(seed), int(difficulty))
        _CACHE[key] = inst
    return inst


class RewardStats:
    """Rolling success/hack-flag telemetry printed during training."""

    def __init__(self, log_every: int = 20):
        self.calls = 0
        self.log_every = log_every
        self.totals: list[float] = []
        self.succ: list[float] = []
        self.flags: Counter = Counter()
        self.per_env: Counter = Counter()
        self.per_env_succ: Counter = Counter()

    def update(self, env_id, rb):
        self.totals.append(rb.total)
        self.succ.append(float(rb.success))
        self.per_env[env_id] += 1
        self.per_env_succ[env_id] += int(rb.success)
        for fl in rb.hack_flags:
            self.flags[fl.split(":")[0]] += 1

    def maybe_log(self):
        self.calls += 1
        if self.calls % self.log_every:
            return
        n = len(self.totals)
        if not n:
            return
        last = self.totals[-512:]
        lsucc = self.succ[-512:]
        envs = " ".join(
            f"{e}:{self.per_env_succ[e] / max(1, self.per_env[e]):.2f}"
            for e in sorted(self.per_env))
        print(f"[reward] n={n} mean_reward(512)={sum(last) / len(last):.3f} "
              f"success(512)={sum(lsucc) / len(lsucc):.3f} flags={dict(self.flags)}")
        print(f"[reward] per-env success: {envs}")


def make_reward_fn(stats: RewardStats | None = None):
    stats = stats or RewardStats()

    def atlas_reward(prompts=None, completions=None, env_id=None, seed=None,
                     difficulty=None, **kwargs):
        rewards = []
        for c, e, s, d in zip(completions, env_id, seed, difficulty):
            inst = get_instance(e, s, d)
            rb = REGISTRY[e].verify(inst, completion_text(c))
            stats.update(e, rb)
            rewards.append(rb.total)
        stats.maybe_log()
        return rewards

    atlas_reward.__name__ = "atlas_verifier_reward"
    return atlas_reward
