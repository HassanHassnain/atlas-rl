"""Build the comparison report (the headline table) from eval summaries.

Takes multiple run directories produced by run_eval, aligns them, computes
paired significance vs a chosen baseline, and emits Markdown + a PNG chart.

Usage:
    python -m atlas_rl.evaluation.report \
        --runs results/base_3b results/grpo_3b results/sft_3b results/qwen_32b \
        --baseline results/base_3b --out results/report
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict

from atlas_rl.evaluation.stats import bootstrap_ci, fmt_ci, paired_bootstrap_diff


def load_run(run_dir: str) -> tuple[dict, dict[str, dict]]:
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    rows = {}
    with open(os.path.join(run_dir, "rows.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rows[r["instance_id"]] = r
    return summary, rows


def build_report(run_dirs: list[str], baseline_dir: str | None) -> str:
    runs = [(d, *load_run(d)) for d in run_dirs]
    base = None
    if baseline_dir:
        baseline_norm = os.path.abspath(os.path.normpath(baseline_dir))
        base = next((r for r in runs
                     if os.path.abspath(os.path.normpath(r[0])) == baseline_norm), None)
        if base is None:
            raise ValueError(f"baseline directory is not one of the runs: {baseline_dir}")

    lines = ["# Atlas-RL Evaluation Report", ""]
    lines.append("## Overall Held-Out Results")
    lines.append("")
    lines.append("| model | pass@1 [95% CI] | mean reward [95% CI] | n | backend errors |"
                 + (" Δ pass@1 [95% CI] | p(better) | Δ reward [95% CI] | "
                    "p(better) |" if base else ""))
    lines.append("|---|---|---|---|---|" + ("---|---|---|---|" if base else ""))
    for d, s, rows in runs:
        o = s["overall"]
        reward_ci = bootstrap_ci([r["reward1"] for r in rows.values()])
        cell = f"| `{s['model']}` | {fmt_ci(o['pass1'], *o['pass1_ci'])} | " \
               f"{reward_ci[0]:.3f} [{reward_ci[1]:.3f}, {reward_ci[2]:.3f}] | " \
               f"{o['n']} | {o.get('error_samples', 0)} |"
        if base and os.path.abspath(os.path.normpath(d)) != \
                os.path.abspath(os.path.normpath(base[0])):
            common = sorted(set(rows) & set(base[2]))
            if not common:
                raise ValueError(f"run {d} has no instances in common with baseline {base[0]}")
            a = [rows[i]["pass1"] for i in common]
            b = [base[2][i]["pass1"] for i in common]
            pass_diff = paired_bootstrap_diff(a, b)
            reward_diff = paired_bootstrap_diff(
                [rows[i]["reward1"] for i in common],
                [base[2][i]["reward1"] for i in common],
            )
            cell += (f" {pass_diff['mean_diff'] * 100:+.1f} pts "
                     f"[{pass_diff['ci_lo'] * 100:+.1f}, "
                     f"{pass_diff['ci_hi'] * 100:+.1f}] | "
                     f"{pass_diff['p_better']:.3f} | "
                     f"{reward_diff['mean_diff']:+.3f} "
                     f"[{reward_diff['ci_lo']:+.3f}, {reward_diff['ci_hi']:+.3f}] | "
                     f"{reward_diff['p_better']:.3f} |")
        elif base:
            cell += " — | — | — | — |"
        lines.append(cell)

    lines += ["", "## Per-environment pass@1", ""]
    env_ids = sorted({e for _, s, _ in runs for e in s["by_env"]})
    header = "| env | " + " | ".join(f"`{s['model']}`" for _, s, _ in runs) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(runs))
    for e in env_ids:
        cells = []
        for _, s, _ in runs:
            v = s["by_env"].get(e)
            cells.append(f"{v['pass1'] * 100:.1f}%" if v else "—")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-environment mean reward", ""]
    lines.append(header)
    lines.append("|---|" + "---|" * len(runs))
    for e in env_ids:
        cells = []
        for _, s, _ in runs:
            v = s["by_env"].get(e)
            cells.append(f"{v['mean_reward']:.3f}" if v else "—")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-difficulty pass@1 (averaged over envs)", ""]
    lines.append("| model | " + " | ".join(f"d{d}" for d in (1, 2, 3, 4, 5)) + " |")
    lines.append("|---|" + "---|" * 5)
    for _, s, rows in runs:
        byd = defaultdict(list)
        for r in rows.values():
            byd[r["difficulty"]].append(r["pass1"])
        cells = [f"{(sum(byd[d]) / len(byd[d])) * 100:.1f}%" if byd.get(d) else "—"
                 for d in (1, 2, 3, 4, 5)]
        lines.append(f"| `{s['model']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_chart(run_dirs: list[str], out_png: str) -> bool:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "atlas-rl-matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    summaries = [load_run(d)[0] for d in run_dirs]
    env_ids = sorted({e for s in summaries for e in s["by_env"]})
    x = range(len(env_ids))
    width = 0.8 / max(1, len(summaries))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.3 * len(env_ids)), 8.5))
    ax_pass, ax_reward = axes
    for i, s in enumerate(summaries):
        pass_vals = [100 * s["by_env"].get(e, {}).get("pass1", 0) for e in env_ids]
        reward_vals = [s["by_env"].get(e, {}).get("mean_reward", 0) for e in env_ids]
        positions = [xi + i * width for xi in x]
        label = s["model"][:42]
        ax_pass.bar(positions, pass_vals, width=width, label=label)
        ax_reward.bar(positions, reward_vals, width=width, label=label)
    for ax in axes:
        ax.set_xticks([xi + 0.4 - width / 2 for xi in x])
        ax.set_xticklabels(env_ids, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7)
    ax_pass.set_ylabel("pass@1 (%)")
    ax_pass.set_title("Strict pass@1 by environment")
    ax_reward.set_ylabel("mean verifier reward")
    ax_reward.set_ylim(0, 1)
    ax_reward.set_title("Partial-credit reward by environment")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories")
    ap.add_argument("--baseline", help="run dir used for paired comparisons")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    md = build_report(args.runs, args.baseline)
    path = os.path.join(args.out, "REPORT.md")
    with open(path, "w") as f:
        f.write(md + "\n")
    png = os.path.join(args.out, "metrics_by_env.png")
    plotted = plot_chart(args.runs, png)
    print(md)
    print(f"\nwrote {path}" + (f" and {png}" if plotted else " (matplotlib missing: no chart)"))


if __name__ == "__main__":
    main()
