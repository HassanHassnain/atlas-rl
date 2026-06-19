"""Evaluate any backend on held-out, freshly generated instances.

Eval seeds live in a disjoint range from training seeds (EVAL_SEED_BASE), so
every evaluation instance is contamination-free by construction.

Examples:
    # CPU smoke test of the whole pipeline (no GPU needed):
    python -m atlas_rl.evaluation.run_eval --model mock:noisy_oracle:0.6 \
        --n-per-env 20 --difficulties 2 3 4 --out results/mock_noisy

    # RUNTIME-ONLY (3090 box): base model
    python -m atlas_rl.evaluation.run_eval --model hf:Qwen/Qwen2.5-3B-Instruct \
        --n-per-env 100 --difficulties 2 3 4 --out results/base_3b

    # RUNTIME-ONLY: GRPO checkpoint (LoRA adapter)
    python -m atlas_rl.evaluation.run_eval \
        --model "hf:Qwen/Qwen2.5-3B-Instruct:adapter=checkpoints/grpo_3b/final" \
        --n-per-env 100 --difficulties 2 3 4 --out results/grpo_3b

    # 10x-larger local open-model baseline:
    python -m atlas_rl.evaluation.run_eval \
        --model "hf:Qwen/Qwen2.5-32B-Instruct" \
        --n-per-env 100 --difficulties 2 3 4 --out results/qwen_32b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from statistics import mean

from tqdm import tqdm

from atlas_rl import REGISTRY
from atlas_rl.core.seeding import EVAL_SEED_BASE, child_seed
from atlas_rl.evaluation.stats import bootstrap_ci
from atlas_rl.inference.backends import GenConfig, make_backend


def evaluate(model_spec: str, env_ids: list[str], n_per_env: int,
             difficulties: list[int], k: int = 1, temperature: float = 0.2,
             max_new_tokens: int = 768, seed_offset: int = 0,
             progress: bool = True, top_p: float = 0.95) -> dict:
    if not env_ids:
        raise ValueError("env_ids must not be empty")
    if n_per_env <= 0:
        raise ValueError("n_per_env must be positive")
    if not difficulties or any(d not in range(1, 6) for d in difficulties):
        raise ValueError("difficulties must contain values in 1..5")
    if k <= 0:
        raise ValueError("k must be positive")

    backend = make_backend(model_spec)
    rows = []
    iterator = [(e, i) for e in env_ids for i in range(n_per_env)]
    if progress:
        iterator = tqdm(iterator, desc=f"eval {model_spec}", ncols=100)
    try:
        for env_id, i in iterator:
            env = REGISTRY[env_id]
            d = difficulties[i % len(difficulties)]
            inst = env.generate(EVAL_SEED_BASE + seed_offset + i, d)
            samples = []
            for j in range(k):
                sample_seed = child_seed("eval", env_id, inst.seed, d, j)
                cfg = GenConfig(
                    temperature=temperature if k > 1 or temperature > 0 else 0.0,
                    top_p=top_p, max_new_tokens=max_new_tokens, seed=sample_seed)
                t0 = time.time()
                err = None
                try:
                    text = backend.complete(inst, cfg)
                except Exception as exc:  # record the failure, then finish the run
                    text, err = "", f"{type(exc).__name__}: {exc}"[:300]
                dt = time.time() - t0
                rb = env.verify(inst, text)
                sample = {"sample": j, "total": rb.total, "success": rb.success,
                          "correctness": rb.correctness,
                          "components": rb.components, "hack_flags": rb.hack_flags,
                          "latency_s": round(dt, 3), "response_chars": len(text),
                          "response": text}
                if err:
                    sample["error"] = err
                samples.append(sample)
            rows.append({
                "env_id": env_id, "seed": inst.seed, "difficulty": d,
                "instance_id": inst.instance_id,
                "pass1": float(samples[0]["success"]),
                "reward1": samples[0]["total"],
                "pass_any": float(any(s["success"] for s in samples)),
                "had_error": any("error" in s for s in samples),
                "samples": samples,
            })
    finally:
        backend.close()
    return {"model": model_spec, "k": k, "temperature": temperature,
            "top_p": top_p, "difficulties": difficulties,
            "n_per_env": n_per_env, "seed_offset": seed_offset,
            "seed_base": EVAL_SEED_BASE + seed_offset, "rows": rows}


def summarize(result: dict) -> dict:
    rows = result["rows"]
    by_env: dict[str, list] = defaultdict(list)
    for r in rows:
        by_env[r["env_id"]].append(r)
    env_stats = {}
    for env_id, rs in sorted(by_env.items()):
        p = [r["pass1"] for r in rs]
        m, lo, hi = bootstrap_ci(p)
        env_stats[env_id] = {
            "n": len(rs),
            "pass1": m, "pass1_ci": [lo, hi],
            "mean_reward": mean(r["reward1"] for r in rs),
            "pass_any": mean(r["pass_any"] for r in rs),
            "by_difficulty": {
                str(d): mean(r["pass1"] for r in rs if r["difficulty"] == d)
                for d in sorted({r["difficulty"] for r in rs})
            },
        }
    allp = [r["pass1"] for r in rows]
    m, lo, hi = bootstrap_ci(allp)
    error_samples = sum("error" in sample for r in rows for sample in r["samples"])
    return {
        "model": result["model"],
        "overall": {"n": len(rows), "pass1": m, "pass1_ci": [lo, hi],
                    "pass_any": mean(r["pass_any"] for r in rows) if rows else 0.0,
                    "mean_reward": mean(r["reward1"] for r in rows) if rows else 0.0,
                    "instances_with_errors": sum(bool(r.get("had_error")) for r in rows),
                    "error_samples": error_samples},
        "by_env": env_stats,
        "config": {key: result[key] for key in
                   ("k", "temperature", "top_p", "difficulties", "n_per_env",
                    "seed_offset", "seed_base")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="backend spec (see module docstring)")
    ap.add_argument("--envs", nargs="+", default=sorted(REGISTRY))
    ap.add_argument("--n-per-env", type=int, default=100)
    ap.add_argument("--difficulties", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--k", type=int, default=1, help="samples per instance (pass@k)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--allow-errors", action="store_true",
                    help="exit zero even if backend generation calls fail")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    for e in args.envs:
        if e not in REGISTRY:
            raise SystemExit(f"unknown env {e!r}; known: {sorted(REGISTRY)}")
    result = evaluate(args.model, args.envs, args.n_per_env, args.difficulties,
                      args.k, args.temperature, args.max_new_tokens,
                      args.seed_offset, top_p=args.top_p)
    summary = summarize(result)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for r in result["rows"]:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    o = summary["overall"]
    pass_any = f"pass@{args.k}={o['pass_any'] * 100:.1f}%  " if args.k > 1 else ""
    print(f"\n{summary['model']}  pass@1={o['pass1'] * 100:.1f}% "
          f"[{o['pass1_ci'][0] * 100:.1f}, {o['pass1_ci'][1] * 100:.1f}]  "
          f"{pass_any}mean_reward={o['mean_reward']:.3f}  n={o['n']}  "
          f"errors={o['error_samples']}")
    for env_id, s in summary["by_env"].items():
        print(f"  {env_id:<18} pass@1={s['pass1'] * 100:5.1f}%  reward={s['mean_reward']:.3f}")
    print(f"\nwrote {args.out}/rows.jsonl and summary.json")
    if o["error_samples"] and not args.allow_errors:
        raise SystemExit(
            f"evaluation completed with {o['error_samples']} backend errors; "
            "inspect rows.jsonl or rerun with --allow-errors")


if __name__ == "__main__":
    main()
