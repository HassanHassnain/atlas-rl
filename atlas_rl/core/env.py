"""AtlasEnv: base class for all Atlas-RL environments.

Contract every environment must satisfy (enforced by tests/):

1. Determinism      generate(seed, d) is byte-identical across calls/machines.
2. Verifiability    verify(inst, oracle(inst)) => total >= 0.99 and success.
3. Anti-hacking     every canary response scores total <= CAUGHT_THRESHOLD
                    and success == False.
4. Difficulty       metadata["complexity"] is non-decreasing in expectation
                    with the difficulty dial.

Observation space: a single user message (text). Single-turn episodes.
Action space: free text, preferably containing one <answer>...</answer> block,
whose content grammar is environment-specific (documented in `answer_format`).
Reward: total = 0.05*tag + 0.05*parse + 0.9*correctness, in [0, 1].
Strict success requires a parseable, semantically exact answer; the wrapper is
rewarded formatting, not part of task correctness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from atlas_rl.core.seeding import rng_for
from atlas_rl.core.types import (
    CAUGHT_THRESHOLD,
    MAX_RESPONSE_CHARS,
    Canary,
    Instance,
    RewardBreakdown,
)

SYSTEM_PROMPT = (
    "You are an expert Site Reliability Engineer assistant. Return only the "
    "final answer. Your entire response must be one <answer>...</answer> block "
    "whose content follows the exact format requested by the task. Do not use "
    "Markdown fences or include any text outside the answer block."
)


class AtlasEnv(ABC):
    env_id: str = "abstract"
    name: str = "Abstract environment"
    description: str = ""
    answer_format: str = ""          # human-readable answer grammar
    difficulty_dials: dict[str, str] = {}

    # ---------------------------------------------------------------- generate
    def generate(self, seed: int, difficulty: int) -> Instance:
        if not 1 <= difficulty <= 5:
            raise ValueError(f"difficulty must be in 1..5, got {difficulty}")
        rng = rng_for(self.env_id, seed, difficulty)
        prompt, ground_truth, metadata = self._build(rng, difficulty)
        metadata.setdefault("complexity", 0)
        return Instance(
            env_id=self.env_id,
            seed=seed,
            difficulty=difficulty,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    @abstractmethod
    def _build(self, rng, difficulty: int) -> tuple[str, dict, dict]:
        """Returns (user_prompt, ground_truth, metadata)."""

    # ------------------------------------------------------------------ verify
    @abstractmethod
    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        """Score a raw model response against the instance ground truth."""

    @abstractmethod
    def oracle(self, instance: Instance) -> str:
        """A reference response (with <answer> tags) achieving reward ~1.0."""

    # ------------------------------------------------------------------ helpers
    def score(
        self,
        *,
        tag_found: bool,
        parsed_ok: bool,
        correctness: float,
        success: bool,
        components: dict[str, float] | None = None,
        hack_flags: list[str] | None = None,
        response: str = "",
    ) -> RewardBreakdown:
        flags = list(hack_flags or [])
        if response and len(response) > MAX_RESPONSE_CHARS:
            flags.append("overlong_response")
            return RewardBreakdown(0.0, 0.0, 0.0, False, components or {}, flags)
        correctness = max(0.0, min(1.0, correctness))
        fmt = 0.05 * bool(tag_found) + 0.05 * bool(parsed_ok)
        total = round(fmt + 0.9 * correctness, 6)
        return RewardBreakdown(
            total=total,
            correctness=round(correctness, 6),
            format_score=round(fmt, 6),
            success=bool(success and parsed_ok),
            components=components or {},
            hack_flags=flags,
        )

    # ---------------------------------------------------------------- canaries
    def canaries(self, instance: Instance) -> list[Canary]:
        """Seeded exploit attempts. All must score <= CAUGHT_THRESHOLD."""
        common = [
            Canary("empty", "", "empty response"),
            Canary("no_tag", "After careful analysis the answer is obvious.",
                   "plausible text, no answer tags"),
            Canary("format_only", "<answer>done</answer>",
                   "well-formed tags, content-free answer"),
            Canary("echo_prompt", f"<answer>{instance.prompt[:600]}</answer>",
                   "echoes the prompt back inside tags"),
            Canary("tag_spam", "<answer>x</answer>" * 25,
                   "many answer tags"),
            Canary("huge", "<answer>" + "A" * (MAX_RESPONSE_CHARS + 1000) + "</answer>",
                   "oversized response"),
        ]
        return common + self.extra_canaries(instance)

    def extra_canaries(self, instance: Instance) -> list[Canary]:
        return []

    # convenience used by audits/tests
    def canary_caught(self, instance: Instance, canary: Canary) -> bool:
        rb = self.verify(instance, canary.response)
        return rb.total <= CAUGHT_THRESHOLD and not rb.success
