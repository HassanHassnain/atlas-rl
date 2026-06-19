"""GRPO training on Atlas-RL environments. RUNTIME-ONLY (needs a CUDA GPU).

Tested target: one RTX 3090 (24 GB) with Qwen2.5-1.5B/3B-Instruct + QLoRA.
Two-GPU option: serve generations with TRL's vLLM server on GPU 0 and train on
GPU 1 using the `vllm` section of the YAML config.

Usage (on the GPU box):
    python -m atlas_rl.training.grpo_train --config configs/grpo_qwen1p5b_pilot.yaml
    python -m atlas_rl.training.grpo_train --config configs/grpo_qwen3b_3090.yaml

Outputs: checkpoints under cfg.output_dir, final adapter at
<output_dir>/final, TensorBoard logs at <output_dir>/runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil

import yaml


def write_reward_stats(out_dir: str, stats) -> None:
    """Persist rollout telemetry, including when training exits early."""
    last_totals = stats.totals[-512:]
    last_success = stats.succ[-512:]
    payload = {
        "mean_reward_last512": sum(last_totals) / max(1, len(last_totals)),
        "success_last512": sum(last_success) / max(1, len(last_success)),
        "hack_flags": dict(stats.flags),
        "n_reward_calls": len(stats.totals),
    }
    with open(os.path.join(out_dir, "reward_stats.json"), "w") as f:
        json.dump(payload, f, indent=2)


def _filter_kwargs(cls, d: dict, label: str) -> dict:
    """Drop keys the installed TRL version doesn't know, with a warning."""
    known = {f.name for f in dataclasses.fields(cls)}
    out = {k: v for k, v in d.items() if k in known}
    dropped = sorted(set(d) - known)
    if dropped:
        print(f"[warn] {label}: dropping unsupported keys {dropped} "
              f"for installed trl/transformers version")
    return out


def load_train_dataset(path: str, envs: list[str] | None):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    if envs:
        keep = set(envs)
        ds = ds.filter(lambda r: r["env_id"] in keep)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from atlas_rl.training.reward_adapter import RewardStats, make_reward_fn

    model_id = cfg["model"]["id"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config_used.yaml"))

    # ---------------------------------------------------------------- model
    model_kwargs: dict = {"torch_dtype": torch.bfloat16}
    if cfg["model"].get("attn_implementation"):
        model_kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]
    if cfg["model"].get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    peft_cfg = None
    init_adapter = cfg["model"].get("init_adapter")
    if init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        print(f"[model] continuing trainable adapter from {init_adapter}")
    elif cfg.get("lora", {}).get("enabled", True):
        lo = cfg.get("lora", {})
        peft_cfg = LoraConfig(
            r=lo.get("r", 32),
            lora_alpha=lo.get("alpha", 64),
            lora_dropout=lo.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lo.get("target_modules",
                                  ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]),
        )

    # ------------------------------------------------------------------ data
    ds = load_train_dataset(cfg["dataset"]["path"], cfg["dataset"].get("envs"))
    print(f"[data] {len(ds)} prompts from {cfg['dataset']['path']}")

    # ------------------------------------------------------------------ args
    t = cfg["train"]
    grpo_dict = dict(
        output_dir=out_dir,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=float(t.get("learning_rate", 1e-5)),
        lr_scheduler_type=t.get("lr_scheduler_type", "constant_with_warmup"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        max_steps=t.get("max_steps", 600),
        logging_steps=t.get("logging_steps", 5),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        bf16=True,
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        num_generations=t.get("num_generations", 8),
        max_prompt_length=t.get("max_prompt_length", 1536),
        max_completion_length=t.get("max_completion_length", 640),
        temperature=t.get("temperature", 0.9),
        top_p=t.get("top_p", 0.95),
        beta=t.get("beta", 0.04),
        report_to=t.get("report_to", ["tensorboard"]),
        seed=t.get("seed", 17),
    )
    # vLLM generation server (two-GPU setup) and any version-specific extras
    grpo_dict.update(cfg.get("vllm", {}))
    grpo_dict.update(cfg.get("grpo_extra", {}))
    grpo_args = GRPOConfig(**_filter_kwargs(GRPOConfig, grpo_dict, "GRPOConfig"))

    stats = RewardStats(log_every=t.get("reward_log_every", 20))
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(stats)],
        args=grpo_args,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    finally:
        write_reward_stats(out_dir, stats)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] adapter saved to {final_dir}")
    print(f"[next] evaluate with: python -m atlas_rl.evaluation.run_eval "
          f"--model 'hf:{model_id}:adapter={final_dir}' --n-per-env 100 "
          f"--difficulties 2 3 4 --out results/grpo_eval")


if __name__ == "__main__":
    main()
