"""Cross-environment generalization (transfer) study.

Evaluates checkpoints trained on a SUBSET of environments against ALL
environments, producing the transfer matrix that is one of the FYP's research
deliverables. Works with any backend spec, so the pipeline can be smoke-tested
with mock backends before GPU checkpoints exist.

Config (YAML):
    models:
      - name: grpo_heldout_shell      # trained on all envs EXCEPT shell_golf
        spec: "hf:Qwen/Qwen2.5-3B-Instruct:adapter=checkpoints/grpo_no_shell/final"
        trained_on: [log_triage, config_repair, ...]
    n_per_env: 50
    difficulties: [2, 3, 4]

Usage:
    python -m atlas_rl.evaluation.transfer_matrix --config configs/transfer.yaml \
        --out results/transfer
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import yaml

from atlas_rl import REGISTRY
from atlas_rl.evaluation.run_eval import evaluate, summarize


def run_transfer(cfg: dict) -> dict:
    env_ids = cfg.get("envs", sorted(REGISTRY))
    matrix = {}
    summaries = {}
    for m in cfg["models"]:
        res = evaluate(m["spec"], env_ids, cfg.get("n_per_env", 50),
                       cfg.get("difficulties", [2, 3, 4]),
                       temperature=cfg.get("temperature", 0.2),
                       top_p=cfg.get("top_p", 0.95),
                       max_new_tokens=cfg.get("max_new_tokens", 768))
        s = summarize(res)
        if s["overall"]["error_samples"]:
            raise RuntimeError(
                f"{m['name']} had {s['overall']['error_samples']} backend errors; "
                "transfer results would be invalid")
        summaries[m["name"]] = s
        matrix[m["name"]] = {
            "trained_on": m.get("trained_on", []),
            "pass1_by_env": {e: s["by_env"][e]["pass1"] for e in s["by_env"]},
            "reward_by_env": {e: s["by_env"][e]["mean_reward"] for e in s["by_env"]},
        }
    return {"matrix": matrix, "summaries": summaries, "envs": env_ids}


def _render_metric(matrix: dict, env_ids: list[str], key: str, scale: float) -> list[str]:
    lines = ["| model \\ env | " + " | ".join(env_ids) + " |",
             "|---|" + "---|" * len(env_ids)]
    for name, row in matrix.items():
        cells = []
        for e in env_ids:
            v = row[key].get(e)
            star = "*" if e in row["trained_on"] else ""
            cells.append(f"{v * scale:.0f}%{star}" if v is not None else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def render(matrix: dict, env_ids: list[str]) -> str:
    lines = ["# Transfer Matrix (rows = models, cols = eval envs)", "",
             "`*` marks in-training-set cells; unmarked cells are held-out transfer.",
             "", "## Strict pass@1", ""]
    lines.extend(_render_metric(matrix, env_ids, "pass1_by_env", 100))
    lines.extend(["", "## Mean verifier reward", ""])
    lines.extend(_render_metric(matrix, env_ids, "reward_by_env", 100))
    return "\n".join(lines)


def plot(matrix: dict, env_ids: list[str], out_png: str) -> bool:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "atlas-rl-matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = list(matrix)
    fig, axes = plt.subplots(
        2, 1, figsize=(1.1 * len(env_ids) + 2, 1.2 * len(names) + 4))
    for ax, key, title in zip(
            axes, ("pass1_by_env", "reward_by_env"),
            ("Strict pass@1", "Mean verifier reward")):
        data = [[matrix[n][key].get(e, 0.0) for e in env_ids] for n in names]
        im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(env_ids)), env_ids, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i, n in enumerate(names):
            for j, e in enumerate(env_ids):
                mark = "*" if e in matrix[n]["trained_on"] else ""
                ax.text(j, i, f"{data[i][j] * 100:.0f}{mark}", ha="center", va="center",
                        color="white" if data[i][j] < 0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, label=title)
        ax.set_title(f"{title} (* = trained on)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = run_transfer(cfg)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "transfer.json"), "w") as f:
        json.dump(out["matrix"], f, indent=2)
    md = render(out["matrix"], out["envs"])
    with open(os.path.join(args.out, "TRANSFER.md"), "w") as f:
        f.write(md + "\n")
    plotted = plot(out["matrix"], out["envs"], os.path.join(args.out, "transfer.png"))
    print(md)
    print(f"\nwrote {args.out}/TRANSFER.md" + (" and transfer.png" if plotted else ""))


if __name__ == "__main__":
    main()
