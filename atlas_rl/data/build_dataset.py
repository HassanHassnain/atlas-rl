"""Build JSONL prompt datasets for GRPO / SFT training and evaluation.

Train seeds start at TRAIN_SEED_BASE (0); eval seeds at EVAL_SEED_BASE
(1,000,000). The ranges are disjoint, so eval instances are fresh by
construction — the contamination-proofing argument of the whole project.
Rows carry (env_id, seed, difficulty) so reward functions can regenerate the
exact instance instead of trusting serialized state.

Usage:
    python -m atlas_rl.data.build_dataset --split train --n-per-env 600 \
        --difficulty-mix 1:0.15 2:0.25 3:0.3 4:0.2 5:0.1 --out data/generated/train.jsonl
    python -m atlas_rl.data.build_dataset --split eval --n-per-env 100 \
        --difficulties 2 3 4 --out data/generated/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random

from atlas_rl import REGISTRY
from atlas_rl.core.seeding import EVAL_SEED_BASE, TRAIN_SEED_BASE


def build_rows(env_ids, split: str, n_per_env: int,
               difficulty_mix: dict[int, float] | None = None,
               difficulties: list[int] | None = None,
               include_instance: bool = False) -> list[dict]:
    base = TRAIN_SEED_BASE if split == "train" else EVAL_SEED_BASE
    rows = []
    for env_id in env_ids:
        env = REGISTRY[env_id]
        # difficulty assignment is deterministic given the split/env
        rng = random.Random(f"{split}:{env_id}")
        for i in range(n_per_env):
            if difficulties:
                d = difficulties[i % len(difficulties)]
            else:
                mix = difficulty_mix or {1: .15, 2: .25, 3: .3, 4: .2, 5: .1}
                d = rng.choices(sorted(mix), weights=[mix[k] for k in sorted(mix)])[0]
            inst = env.generate(base + i, d)
            row = {
                "prompt": [
                    {"role": "system", "content": inst.system},
                    {"role": "user", "content": inst.prompt},
                ],
                "env_id": env_id,
                "seed": inst.seed,
                "difficulty": d,
                "instance_id": inst.instance_id,
            }
            if include_instance:
                row["ground_truth"] = inst.ground_truth
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--envs", nargs="+", default=sorted(REGISTRY),
                    help="environment ids (default: all)")
    ap.add_argument("--n-per-env", type=int, default=600)
    ap.add_argument("--difficulties", type=int, nargs="+",
                    help="fixed difficulty cycle (eval style)")
    ap.add_argument("--difficulty-mix", nargs="+",
                    help="train-style weighted mix, e.g. 1:0.15 2:0.25 3:0.3 4:0.2 5:0.1")
    ap.add_argument("--include-instance", action="store_true",
                    help="embed ground truth in rows (debugging only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mix = None
    if args.difficulty_mix:
        mix = {int(kv.split(":")[0]): float(kv.split(":")[1]) for kv in args.difficulty_mix}
    for e in args.envs:
        if e not in REGISTRY:
            raise SystemExit(f"unknown env {e!r}; known: {sorted(REGISTRY)}")
    rows = build_rows(args.envs, args.split, args.n_per_env, mix,
                      args.difficulties, args.include_instance)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({len(args.envs)} envs x {args.n_per_env}) -> {args.out}")


if __name__ == "__main__":
    main()
