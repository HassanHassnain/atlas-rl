"""Inference backends used by evaluation and rejection sampling.

Backend spec strings (the --model argument everywhere):

  mock:oracle            perfect policy (uses env.oracle)        [CPU, no deps]
  mock:noisy_oracle:0.6  oracle with prob 0.6, junk otherwise    [CPU, no deps]
  mock:format_only       well-formed but content-free answers    [CPU, no deps]
  mock:empty             empty responses                         [CPU, no deps]
  hf:Qwen/Qwen2.5-3B-Instruct[:adapter=PATH][:thinking=false]
                                               local transformers generate
                                               (RUNTIME-ONLY: needs GPU box)
  openai:MODEL@BASE_URL  any OpenAI-compatible server (vLLM serve, OpenRouter,
                         Together, ...). API key from ATLAS_API_KEY or
                         OPENAI_API_KEY env var.

The mock backends exist so the entire evaluation pipeline (sampling, scoring,
stats, reports) can be exercised and tested on a CPU-only machine.
"""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass

from atlas_rl.core.registry import get_env
from atlas_rl.core.types import Instance


@dataclass
class GenConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 768
    seed: int | None = None


class Backend:
    name = "base"

    def complete(self, instance: Instance, cfg: GenConfig) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------- mock
class MockBackend(Backend):
    def __init__(self, mode: str, p: float = 0.5):
        self.mode, self.p = mode, p
        self.name = f"mock:{mode}"

    def complete(self, instance: Instance, cfg: GenConfig) -> str:
        env = get_env(instance.env_id)
        if self.mode == "oracle":
            return ("Let me work through this carefully.\n"
                    + env.oracle(instance))
        if self.mode == "noisy_oracle":
            from atlas_rl.core.seeding import child_seed

            rng = random.Random(
                child_seed("mock", instance.seed, instance.difficulty, cfg.seed or 0))
            if rng.random() < self.p:
                return env.oracle(instance)
            return "<answer>not sure</answer>"
        if self.mode == "format_only":
            return "I think the answer is clear.\n<answer>see above</answer>"
        if self.mode == "empty":
            return ""
        raise ValueError(f"unknown mock mode {self.mode!r}")


# ----------------------------------------------------------------- hf local
class HFBackend(Backend):
    """Local transformers backend. RUNTIME-ONLY (needs torch + a GPU)."""

    def __init__(self, model_id: str, adapter: str | None = None,
                 enable_thinking: bool | None = None):
        import torch  # noqa: F401  (deferred import)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}" + (f"+{adapter}" if adapter else "")
        self.enable_thinking = enable_thinking
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def complete(self, instance: Instance, cfg: GenConfig) -> str:
        import torch
        from transformers import set_seed

        messages = [{"role": "system", "content": instance.system},
                    {"role": "user", "content": instance.prompt}]
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            **template_kwargs,
        ).to(self.model.device)
        do_sample = cfg.temperature > 0
        if cfg.seed is not None:
            set_seed(cfg.seed % (2**32))
        with torch.no_grad():
            out = self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def close(self) -> None:
        # Long-running evaluators may create several backends in one process.
        # Release our own allocations explicitly; never touch other processes.
        self.model = None
        self.tok = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ------------------------------------------------------------------- openai
class OpenAIBackend(Backend):
    """Any OpenAI-compatible endpoint (vLLM serve, OpenRouter, Together...)."""

    def __init__(self, model: str, base_url: str | None = None):
        from openai import OpenAI

        key = os.environ.get("ATLAS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.send_seed = os.environ.get("ATLAS_SEND_SEED", "").lower() in ("1", "true", "yes")
        self.name = f"openai:{model}" + (f"@{base_url}" if base_url else "")

    def complete(self, instance: Instance, cfg: GenConfig) -> str:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": instance.system},
                      {"role": "user", "content": instance.prompt}],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
        )
        if cfg.seed is not None and self.send_seed:
            kwargs["seed"] = cfg.seed % (2**31)
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def close(self) -> None:
        self.client.close()


# ------------------------------------------------------------------ factory
def make_backend(spec: str) -> Backend:
    if spec.startswith("mock:"):
        parts = spec.split(":")
        mode = parts[1]
        p = float(parts[2]) if len(parts) > 2 else 0.5
        return MockBackend(mode, p)
    if spec.startswith("hf:"):
        parts = spec[3:].split(":")
        model_id, options = parts[0], {}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError(f"invalid HF backend option {part!r}")
            key, value = part.split("=", 1)
            options[key] = value
        unknown = sorted(set(options) - {"adapter", "thinking"})
        if unknown:
            raise ValueError(f"unknown HF backend options: {unknown}")
        enable_thinking = None
        if "thinking" in options:
            value = options["thinking"].lower()
            if value not in {"true", "false"}:
                raise ValueError("HF backend thinking option must be true or false")
            enable_thinking = value == "true"
        return HFBackend(model_id, options.get("adapter"), enable_thinking)
    if spec.startswith("openai:"):
        rest = spec[len("openai:"):]
        if "@" in rest:
            model, base = rest.split("@", 1)
            return OpenAIBackend(model, base)
        return OpenAIBackend(rest)
    raise ValueError(
        f"unknown backend spec {spec!r} (expected mock:|hf:|openai: prefix)")
