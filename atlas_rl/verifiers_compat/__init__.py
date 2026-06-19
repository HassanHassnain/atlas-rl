"""Adapter exposing Atlas-RL environments through the community `verifiers`
spec (https://github.com/willccbb/verifiers), enabling Environments-Hub
submission ("hub submission = instant external users" from the project plan).

Import-guarded: the core project does NOT depend on `verifiers`. Install it on
the training box (`pip install verifiers`) to use this module. The `verifiers`
API moves quickly — this targets the stable SingleTurnEnv + Rubric surface; if
your installed version differs, adapt the two marked call sites.

Usage:
    import atlas_rl.verifiers_compat as vc
    env = vc.load_environment(env_id="config_repair", n_train=2000, n_eval=200)
    # -> a verifiers.SingleTurnEnv usable with verifiers' GRPO trainer or
    #    submittable to the Environments Hub.
"""

from __future__ import annotations

from atlas_rl import REGISTRY
from atlas_rl.core.seeding import EVAL_SEED_BASE, TRAIN_SEED_BASE


def _build_split(env, base: int, n: int, difficulties=(1, 2, 3, 4, 5)):
    rows = []
    for i in range(n):
        d = difficulties[i % len(difficulties)]
        inst = env.generate(base + i, d)
        rows.append({
            "question": inst.prompt,
            "answer": "",  # verified programmatically; no static answer
            "info": {"env_id": inst.env_id, "seed": inst.seed,
                     "difficulty": inst.difficulty},
            "task": inst.env_id,
        })
    return rows


def load_environment(env_id: str = "config_repair", n_train: int = 2000,
                     n_eval: int = 200, difficulties=(1, 2, 3, 4, 5)):
    try:
        import verifiers as vf
        from datasets import Dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "verifiers_compat requires `pip install verifiers datasets`") from e

    env = REGISTRY[env_id]
    train = Dataset.from_list(_build_split(env, TRAIN_SEED_BASE, n_train, difficulties))
    evald = Dataset.from_list(_build_split(env, EVAL_SEED_BASE, n_eval, difficulties))

    def atlas_reward(completion, info=None, **kwargs):
        # verifiers passes the completion (str or messages) plus dataset `info`.
        text = completion if isinstance(completion, str) else \
            "".join(m.get("content", "") for m in completion
                    if m.get("role") == "assistant")
        inst = env.generate(info["seed"], info["difficulty"])
        return env.verify(inst, text).total

    rubric = vf.Rubric(funcs=[atlas_reward], weights=[1.0])  # API call site 1
    return vf.SingleTurnEnv(                                  # API call site 2
        dataset=train,
        eval_dataset=evald,
        system_prompt=env.generate(0, 1).system,
        rubric=rubric,
    )
