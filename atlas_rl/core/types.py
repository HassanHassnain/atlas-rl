"""Core datatypes shared by all environments."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Instance:
    """One procedurally generated task instance.

    Everything needed to (a) prompt a model and (b) verify its response.
    `ground_truth` and `metadata` must stay JSON-serializable so instances can
    be stored in JSONL datasets and regenerated deterministically.
    """

    env_id: str
    seed: int
    difficulty: int          # 1 (easiest) .. 5 (hardest)
    system: str              # system prompt
    prompt: str              # user prompt
    ground_truth: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        return f"{self.env_id}:s{self.seed}:d{self.difficulty}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(**d)


@dataclass
class RewardBreakdown:
    """Verifier output. `total` is always in [0, 1].

    total = 0.05 * tag_found + 0.05 * parsed_ok + 0.9 * correctness
    `success` is the strict semantic pass criterion used for pass@1.
    """

    total: float
    correctness: float
    format_score: float
    success: bool
    components: dict[str, float] = field(default_factory=dict)
    hack_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Canary:
    """A seeded exploit attempt against the reward function."""

    name: str
    response: str
    # A canary is "caught" if reward total <= CAUGHT_THRESHOLD and success is False.
    description: str = ""


CAUGHT_THRESHOLD = 0.25
MAX_RESPONSE_CHARS = 40_000
