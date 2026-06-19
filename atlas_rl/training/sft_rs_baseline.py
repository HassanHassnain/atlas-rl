"""SFT-from-rejection-sampling baseline (the de-risking ablation from the
project plan: "SFT-from-rejection-sampling baseline first — already a result").

Stage `oracle` (CPU):
    build deterministic supervised trajectories from each environment's
    programmatic oracle. This is useful as a curriculum before GRPO.

Stage `sample` (RUNTIME-ONLY: needs the GPU box or an API backend):
    sample k completions per train prompt from the BASE model, keep only those
    the verifier marks as strict successes, and write an SFT dataset. The
    per-env acceptance rate is itself a reported result (base-model pass@k).

Stage `train` (RUNTIME-ONLY): LoRA SFT on the accepted completions via TRL.

Usage:
    python -m atlas_rl.training.sft_rs_baseline --config configs/sft_oracle_qwen3b.yaml --stage oracle
    python -m atlas_rl.training.sft_rs_baseline --config configs/sft_rs_qwen3b.yaml --stage sample
    python -m atlas_rl.training.sft_rs_baseline --config configs/sft_rs_qwen3b.yaml --stage train
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import yaml


def stratified_limit(rows: list[dict], limit: int | None) -> list[dict]:
    """Cap rows while retaining deterministic, balanced env coverage."""
    if not limit or limit >= len(rows):
        return rows
    by_env: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_env[row["env_id"]].append(row)
    envs = sorted(by_env)
    base, remainder = divmod(limit, len(envs))
    selected = []
    for index, env_id in enumerate(envs):
        selected.extend(by_env[env_id][:base + (index < remainder)])
    return selected


def build_oracle_rows(rows: list[dict], limit: int | None = None) -> list[dict]:
    """Build deterministic SFT trajectories from verifier oracles."""
    from atlas_rl import REGISTRY
    from atlas_rl.training.reward_adapter import get_instance

    selected = stratified_limit(rows, limit)
    out = []
    for row in selected:
        env_id = row["env_id"]
        inst = get_instance(env_id, row["seed"], row["difficulty"])
        answer = REGISTRY[env_id].oracle(inst)
        rb = REGISTRY[env_id].verify(inst, answer)
        if not rb.success or rb.total < 0.99:
            raise RuntimeError(
                f"oracle contract failed for {env_id} seed={row['seed']} "
                f"difficulty={row['difficulty']}")
        out.append({
            "messages": row["prompt"] + [{"role": "assistant", "content": answer}],
            "env_id": env_id,
            "seed": row["seed"],
            "difficulty": row["difficulty"],
        })
    return out


def stage_oracle(cfg: dict) -> None:
    with open(cfg["dataset"]["path"]) as f:
        rows = [json.loads(line) for line in f]
    if cfg["dataset"].get("envs"):
        keep = set(cfg["dataset"]["envs"])
        rows = [row for row in rows if row["env_id"] in keep]
    out_rows = build_oracle_rows(rows, cfg.get("oracle", {}).get("max_prompts"))
    os.makedirs(os.path.dirname(cfg["sft_dataset"]) or ".", exist_ok=True)
    with open(cfg["sft_dataset"], "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[done] {len(out_rows)} oracle trajectories -> {cfg['sft_dataset']}")


def stage_sample(cfg: dict) -> None:
    from tqdm import tqdm

    from atlas_rl import REGISTRY
    from atlas_rl.core.seeding import child_seed
    from atlas_rl.inference.backends import GenConfig, make_backend
    from atlas_rl.training.reward_adapter import get_instance

    s = cfg["sample"]
    backend = make_backend(s["backend"])
    with open(cfg["dataset"]["path"]) as f:
        rows = [json.loads(line) for line in f]
    if cfg["dataset"].get("envs"):
        keep = set(cfg["dataset"]["envs"])
        rows = [r for r in rows if r["env_id"] in keep]
    rows = stratified_limit(rows, s.get("max_prompts"))

    out_rows, prompts, attempts, accepts, errors = [], Counter(), Counter(), Counter(), Counter()
    try:
        for r in tqdm(rows, desc="rejection sampling", ncols=100):
            env_id = r["env_id"]
            prompts[env_id] += 1
            inst = get_instance(env_id, r["seed"], r["difficulty"])
            env = REGISTRY[env_id]
            for j in range(s.get("k", 8)):
                attempts[env_id] += 1
                cfg_gen = GenConfig(temperature=s.get("temperature", 0.9),
                                    max_new_tokens=s.get("max_new_tokens", 768),
                                    seed=child_seed("sft-rs", env_id, r["seed"],
                                                    r["difficulty"], j))
                try:
                    text = backend.complete(inst, cfg_gen)
                except Exception:
                    errors[env_id] += 1
                    continue
                rb = env.verify(inst, text)
                if rb.success:
                    accepts[env_id] += 1
                    out_rows.append({
                        "messages": r["prompt"] + [{"role": "assistant", "content": text}],
                        "env_id": env_id, "seed": r["seed"],
                        "difficulty": r["difficulty"],
                    })
                    break  # one accepted completion per prompt
    finally:
        backend.close()

    os.makedirs(os.path.dirname(cfg["sft_dataset"]) or ".", exist_ok=True)
    with open(cfg["sft_dataset"], "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    rates = {
        e: {
            "prompts": prompts[e],
            "accepted_prompts": accepts[e],
            "pass_at_k": round(accepts[e] / max(1, prompts[e]), 4),
            "completions_attempted": attempts[e],
            "completion_acceptance": round(accepts[e] / max(1, attempts[e]), 4),
            "backend_errors": errors[e],
        }
        for e in sorted(prompts)
    }
    with open(cfg["sft_dataset"] + ".stats.json", "w") as f:
        json.dump(rates, f, indent=2)
    print(json.dumps(rates, indent=2))
    print(f"[done] {len(out_rows)} accepted trajectories -> {cfg['sft_dataset']}")
    n_errors = sum(errors.values())
    if n_errors and not s.get("allow_errors", False):
        raise RuntimeError(
            f"{n_errors} rejection-sampling backend calls failed; "
            "stats were written, but the run is not valid")
    if not out_rows:
        raise RuntimeError(
            "rejection sampling produced no strict-success trajectories; "
            "increase k or inspect the answer protocol")


def stage_train(cfg: dict) -> None:
    import dataclasses

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = cfg["model"]["id"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("json", data_files=cfg["sft_dataset"], split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    print(f"[data] {len(ds)} SFT rows")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    lo = cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=lo.get("r", 32), lora_alpha=lo.get("alpha", 64),
        lora_dropout=lo.get("dropout", 0.05), bias="none", task_type="CAUSAL_LM",
        target_modules=lo.get("target_modules",
                              ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]))
    t = cfg["train"]
    sft_dict = dict(
        output_dir=out_dir,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=float(t.get("learning_rate", 1e-5)),
        num_train_epochs=t.get("num_train_epochs", 2),
        logging_steps=t.get("logging_steps", 5),
        save_steps=t.get("save_steps", 200),
        bf16=True,
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        max_length=t.get("max_length", 2304),
        report_to=t.get("report_to", ["tensorboard"]),
        seed=t.get("seed", 17),
    )
    known = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(sft_dict) - known)
    if dropped:
        print(f"[warn] SFTConfig: dropping unsupported keys {dropped}")
    args = SFTConfig(**{k: v for k, v in sft_dict.items() if k in known})
    trainer = SFTTrainer(model=model, processing_class=tok, args=args,
                         train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"[done] SFT adapter saved to {final_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["oracle", "sample", "train", "all"], default="all")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.stage == "oracle":
        stage_oracle(cfg)
    if args.stage in ("sample", "all"):
        stage_sample(cfg)
    if args.stage in ("train", "all"):
        stage_train(cfg)


if __name__ == "__main__":
    main()
