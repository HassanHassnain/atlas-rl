"""Deterministic, collision-resistant seeding.

All environment randomness flows through `rng_for(env_id, seed, difficulty)`.
Same (env_id, seed, difficulty) => byte-identical instance, on any machine,
any Python >= 3.10 (we only rely on `random.Random`, which is stable).
"""

from __future__ import annotations

import hashlib
import random

# Seed ranges. Train and eval draws come from disjoint integer ranges so that
# evaluation instances can never appear in training data ("contamination-proof
# by construction").
TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 1_000_000
AUDIT_SEED_BASE = 2_000_000
FINAL_TEST_SEED_BASE = 4_000_000


def child_seed(*parts: object) -> int:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**63)


def rng_for(env_id: str, seed: int, difficulty: int) -> random.Random:
    return random.Random(child_seed("atlas-rl", env_id, seed, difficulty))
