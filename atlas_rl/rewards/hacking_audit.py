"""Reward-hacking audit: run every seeded exploit (canary) against every
environment's verifier and report the catch rate.

A canary counts as CAUGHT when its reward total <= CAUGHT_THRESHOLD (0.25)
and strict success is False. The FYP success metric is >= 95% caught; the
test suite enforces 100% on the built-in canaries.

Usage:
    python -m atlas_rl.rewards.hacking_audit --seeds 25 --out results/audit
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from atlas_rl import REGISTRY
from atlas_rl.core.seeding import AUDIT_SEED_BASE
from atlas_rl.core.types import CAUGHT_THRESHOLD


def run_audit(n_seeds: int = 25, difficulties=(1, 3, 5)) -> dict:
    rows = []
    for env_id in sorted(REGISTRY):
        env = REGISTRY[env_id]
        for i in range(n_seeds):
            for d in difficulties:
                inst = env.generate(AUDIT_SEED_BASE + i, d)
                for canary in env.canaries(inst):
                    rb = env.verify(inst, canary.response)
                    rows.append({
                        "env": env_id, "seed": inst.seed, "difficulty": d,
                        "canary": canary.name, "total": rb.total,
                        "success": rb.success,
                        "caught": rb.total <= CAUGHT_THRESHOLD and not rb.success,
                        "hack_flags": rb.hack_flags,
                    })
    per_canary: dict[tuple, list] = defaultdict(list)
    for r in rows:
        per_canary[(r["env"], r["canary"])].append(r)
    summary = {
        "n_cases": len(rows),
        "caught": sum(r["caught"] for r in rows),
        "catch_rate": round(sum(r["caught"] for r in rows) / max(1, len(rows)), 6),
        "threshold": CAUGHT_THRESHOLD,
        "by_canary": {
            f"{e}/{c}": {
                "n": len(v),
                "catch_rate": round(sum(x["caught"] for x in v) / len(v), 4),
                "max_total": round(max(x["total"] for x in v), 4),
            }
            for (e, c), v in sorted(per_canary.items())
        },
    }
    return {"summary": summary, "rows": rows}


def render_markdown(audit: dict) -> str:
    s = audit["summary"]
    lines = [
        "# Reward-Hacking Audit",
        "",
        f"- Cases: **{s['n_cases']}**, caught: **{s['caught']}** "
        f"(catch rate **{s['catch_rate'] * 100:.2f}%**, threshold total <= {s['threshold']})",
        "",
        "| env/canary | n | catch rate | worst (max total) |",
        "|---|---|---|---|",
    ]
    for k, v in s["by_canary"].items():
        flag = "" if v["catch_rate"] == 1.0 else " ⚠️"
        lines.append(f"| {k}{flag} | {v['n']} | {v['catch_rate'] * 100:.1f}% | {v['max_total']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--difficulties", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--out", default="results/audit")
    args = ap.parse_args()

    audit = run_audit(args.seeds, tuple(args.difficulties))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "audit_rows.jsonl"), "w") as f:
        for r in audit["rows"]:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "audit_summary.json"), "w") as f:
        json.dump(audit["summary"], f, indent=2)
    md = render_markdown(audit)
    with open(os.path.join(args.out, "AUDIT.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    rate = audit["summary"]["catch_rate"]
    print(f"\n{'PASS' if rate >= 0.95 else 'FAIL'}: catch rate {rate * 100:.2f}% "
          f"(target >= 95%)")
    if rate < 0.95:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
