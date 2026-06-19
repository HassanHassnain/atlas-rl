"""Statistics for evaluation claims: bootstrap CIs and paired tests.

No numpy/scipy dependency — pure stdlib, deterministic via fixed seed.
"""

from __future__ import annotations

import random
from statistics import mean

_B = 10_000
_SEED = 1234


def bootstrap_ci(values: list[float], conf: float = 0.95,
                 n_boot: int = _B) -> tuple[float, float, float]:
    """Returns (mean, lo, hi) percentile-bootstrap CI of the mean."""
    if not values:
        return 0.0, 0.0, 0.0
    m = mean(values)
    if len(values) == 1:
        return m, m, m
    rng = random.Random(_SEED)
    n = len(values)
    boots = sorted(
        mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    alpha = (1 - conf) / 2
    lo = boots[int(alpha * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return m, lo, hi


def paired_bootstrap_diff(a: list[float], b: list[float], conf: float = 0.95,
                          n_boot: int = _B) -> dict:
    """Paired bootstrap of mean(a - b); a and b must be instance-aligned.

    Returns mean_diff, CI, and the fraction of bootstrap resamples where the
    difference is > 0 (a one-sided 'p(better)' summary).
    """
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and align")
    diffs = [x - y for x, y in zip(a, b)]
    m = mean(diffs)
    rng = random.Random(_SEED)
    n = len(diffs)
    boots = sorted(
        mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    alpha = (1 - conf) / 2
    return {
        "mean_diff": m,
        "ci_lo": boots[int(alpha * n_boot)],
        "ci_hi": boots[min(n_boot - 1, int((1 - alpha) * n_boot))],
        "p_better": sum(1 for x in boots if x > 0) / n_boot,
        "n": n,
    }


def fmt_ci(m: float, lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"{m * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
