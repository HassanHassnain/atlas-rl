"""Interactive CLI demo for presentations.

Shows: procedural generation (difficulty dials), the verifier scoring an
oracle answer / your pasted answer / canary exploits — live, no GPU needed.

    python -m atlas_rl.demo                      # guided tour, all envs
    python -m atlas_rl.demo --env shell_golf --difficulty 4 --seed 7
    python -m atlas_rl.demo --env config_repair --interactive
"""

from __future__ import annotations

import argparse
import json

from atlas_rl import REGISTRY


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def show_instance(env, seed: int, difficulty: int, interactive: bool) -> None:
    inst = env.generate(seed, difficulty)
    hr(f"[{env.env_id}] {env.name}   (seed={seed}, difficulty={difficulty})")
    print(f"\n{env.description}")
    print(f"Difficulty dials: {env.difficulty_dials}")
    hr("PROMPT")
    print(inst.prompt[:3000] + ("\n... [truncated]" if len(inst.prompt) > 3000 else ""))

    hr("ORACLE ANSWER -> VERIFIER")
    oracle = env.oracle(inst)
    print(oracle[:800])
    rb = env.verify(inst, oracle)
    print(f"\nreward={rb.total:.3f}  success={rb.success}  components={rb.components}")

    hr("SEEDED EXPLOITS (CANARIES) -> VERIFIER")
    for c in env.canaries(inst)[:6]:
        rb = env.verify(inst, c.response)
        status = "CAUGHT" if (rb.total <= 0.25 and not rb.success) else "!! NOT CAUGHT"
        print(f"  {c.name:<22} reward={rb.total:.3f}  {status:<14} ({c.description})")

    if interactive:
        hr("YOUR TURN — paste a full response (end with a line containing only EOF)")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "EOF":
                break
            lines.append(line)
        if lines:
            rb = env.verify(inst, "\n".join(lines))
            print(f"\nreward={rb.total:.3f}  success={rb.success}  "
                  f"components={rb.components}  flags={rb.hack_flags}")
        same = env.generate(seed, difficulty)
        print(f"\n(determinism check: regenerated instance identical = "
              f"{same.prompt == inst.prompt and same.ground_truth == inst.ground_truth})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=sorted(REGISTRY), help="default: tour all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.env:
        show_instance(REGISTRY[args.env], args.seed, args.difficulty, args.interactive)
        return
    hr("ATLAS-RL ENVIRONMENT SUITE — GUIDED TOUR")
    print(json.dumps({e: REGISTRY[e].name for e in sorted(REGISTRY)}, indent=2))
    for env_id in sorted(REGISTRY):
        show_instance(REGISTRY[env_id], args.seed, args.difficulty, False)
        input("\n[enter] for next environment...")


if __name__ == "__main__":
    main()
