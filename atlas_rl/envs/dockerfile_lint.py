"""dockerfile_lint: find seeded policy violations in a Dockerfile.

Observation: a numbered Dockerfile with K injected violations of a documented
rule catalog, plus near-miss decoys that are NOT violations.
Action: JSON array of findings [{"line": <int>, "rule": "<DLxxx>"}].
Reward: F1 over the set of (line, rule) pairs. Strict success: exact set.

Difficulty dials: file length, number of bugs (1 -> 4), decoy density,
multi-stage builds.
"""

from __future__ import annotations

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import parse_json_answer, wrap_json_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

RULES = {
    "DL001": "Base image must pin an explicit version tag (no bare image, no :latest).",
    "DL002": "Use COPY (not ADD) for local files. ADD is acceptable only for remote URLs or auto-extracted archives.",
    "DL003": "Never put secrets (KEY/TOKEN/SECRET/PASSWORD) in ENV.",
    "DL004": "A non-root USER must be set before the final CMD/ENTRYPOINT. Flag the CMD/ENTRYPOINT line when missing.",
    "DL005": "RUN apt-get install must clean up: end with `rm -rf /var/lib/apt/lists/*`.",
    "DL006": "Copy the full source (COPY . .) only AFTER installing dependencies, or layer caching breaks. Flag the early COPY . . line.",
    "DL007": "EXPOSE must match the port the CMD actually binds (--port argument).",
    "DL008": "The final CMD must use JSON (exec) form, e.g. CMD [\"python\", \"app.py\"].",
}
_BASES = [("python", "3.12-slim"), ("node", "22-alpine"), ("golang", "1.23-bookworm")]
_PKGS = ["curl", "git", "ca-certificates", "build-essential", "libpq-dev", "jq"]
_PARAMS = {1: (1, 1), 2: (2, 1), 3: (2, 2), 4: (3, 3), 5: (4, 4)}  # n_bugs, n_decoys


@register
class DockerfileLintEnv(AtlasEnv):
    env_id = "dockerfile_lint"
    name = "Dockerfile policy lint"
    description = "Report every (line, rule) policy violation in a Dockerfile; decoys look similar but are compliant."
    answer_format = ('JSON array of findings, e.g. [{"line": 4, "rule": "DL003"}]; '
                     "use [] if there are no violations")
    difficulty_dials = {"n_bugs": "1 -> 4", "n_decoys": "1 -> 4", "length": "grows with difficulty"}

    def _build(self, rng, difficulty):
        n_bugs, n_decoys = _PARAMS[difficulty]
        base, tag = rng.choice(_BASES)
        port = rng.choice([3000, 5000, 8000, 8080, 9000])
        app = rng.choice(["app.py", "server.py", "main.py"])
        bug_rules = rng.sample(sorted(RULES), n_bugs)
        decoy_pool = ["good_env", "good_add_url", "good_copy_req", "good_expose",
                      "second_from_pinned"]
        decoys = rng.sample(decoy_pool, min(n_decoys, len(decoy_pool)))

        lines: list[str] = []           # rendered Dockerfile lines
        findings: list[tuple[int, str]] = []  # 1-based (line, rule)

        def emit(text: str, rule: str | None = None):
            lines.append(text)
            if rule:
                findings.append((len(lines), rule))

        # --- FROM
        if "DL001" in bug_rules:
            emit(f"FROM {base}" + ("" if rng.random() < 0.5 else ":latest"), "DL001")
        else:
            emit(f"FROM {base}:{tag}")
        emit(f"WORKDIR /srv/{rng.choice(['api', 'worker', 'svc'])}")

        # --- apt block
        pkgs = " ".join(sorted(rng.sample(_PKGS, rng.randint(2, 4))))
        if "DL005" in bug_rules:
            emit(f"RUN apt-get update && apt-get install -y {pkgs}", "DL005")
        else:
            emit(f"RUN apt-get update && apt-get install -y {pkgs} \\")
            emit("    && rm -rf /var/lib/apt/lists/*")

        # --- early COPY . . cache-bust bug
        if "DL006" in bug_rules:
            emit("COPY . .", "DL006")

        # --- dependency install (decoy: COPY requirements first is GOOD)
        if "good_copy_req" in decoys or True:  # always present, it anchors DL006
            emit("COPY requirements.txt .")
        emit("RUN pip install --no-cache-dir -r requirements.txt")

        # --- ADD/COPY of source
        if "DL002" in bug_rules:
            emit("ADD src/ /srv/src/", "DL002")
        elif "DL006" not in bug_rules:
            emit("COPY . .")
        if "good_add_url" in decoys:
            emit("ADD https://example.com/tools/healthcheck.tar.gz /opt/hc.tar.gz")

        # --- ENV block
        if "DL003" in bug_rules:
            var = rng.choice(["AWS_SECRET_ACCESS_KEY", "API_TOKEN", "DB_PASSWORD",
                              "SIGNING_SECRET"])
            emit(f'ENV {var}="{rng.choice(["sk-live-", "tok_", "pw_"])}'
                 f'{rng.randint(10**8, 10**9 - 1)}"', "DL003")
        if "good_env" in decoys:
            emit(f'ENV {rng.choice(["PUBLIC_URL", "LOG_LEVEL", "APP_MODE"])}='
                 f'"{rng.choice(["https://svc.example.com", "info", "production"])}"')

        # --- second stage decoy (compliant FROM)
        if "second_from_pinned" in decoys and difficulty >= 4:
            emit(f"FROM {base}:{tag} AS runtime")
            emit("COPY --from=0 /srv /srv")

        # --- EXPOSE / USER / CMD
        cmd_port = port
        if "DL007" in bug_rules:
            emit(f"EXPOSE {port + 1}", "DL007")
        else:
            emit(f"EXPOSE {port}")
        if "good_expose" in decoys and "DL007" not in bug_rules:
            pass  # the matching EXPOSE above is itself the decoy
        if "DL004" not in bug_rules:
            emit(f"USER {rng.choice(['svc', 'app', 'runner'])}")
        if "DL008" in bug_rules:
            emit(f"CMD python {app} --port {cmd_port}",
                 "DL008" if "DL004" not in bug_rules else None)
            if "DL004" in bug_rules:
                findings.append((len(lines), "DL004"))
                findings.append((len(lines), "DL008"))
        else:
            emit(f'CMD ["python", "{app}", "--port", "{cmd_port}"]',
                 "DL004" if "DL004" in bug_rules else None)

        numbered = "\n".join(f"{i:>2}  {t}" for i, t in enumerate(lines, start=1))
        rule_doc = "\n".join(f"- {k}: {v}" for k, v in sorted(RULES.items()))
        prompt = (
            "Lint this Dockerfile against the policy catalog. Report EVERY violation "
            "as a (line, rule) finding. Some lines look suspicious but are compliant "
            "— do not flag those.\n\n"
            f"Policy catalog:\n{rule_doc}\n\n"
            f"Dockerfile (line-numbered):\n```\n{numbered}\n```\n\n"
            f"Answer format: {self.answer_format}"
        )
        findings = sorted(set(findings))
        gt = {"findings": [list(f) for f in findings], "n_lines": len(lines)}
        meta = {"complexity": len(lines) + 10 * len(findings)}
        return prompt, gt, meta

    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, ok, obj = parse_json_answer(response)
        gt_set = {(int(l), r) for l, r in instance.ground_truth["findings"]}
        flags: list[str] = []
        if not (ok and isinstance(obj, list)):
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response)
        cand: set[tuple[int, str]] = set()
        for item in obj:
            if (isinstance(item, dict) and "line" in item and "rule" in item
                    and str(item["rule"]).upper() in RULES):
                try:
                    cand.add((int(item["line"]), str(item["rule"]).upper()))
                except (TypeError, ValueError):
                    continue
        tp = len(cand & gt_set)
        prec = tp / max(1, len(cand))
        rec = tp / max(1, len(gt_set))
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # shotgun guard: flagging far more findings than plausibly exist gets
        # its (lucky-overlap) credit slashed instead of farming partial F1
        if len(cand) >= max(6, 2 * len(gt_set) + 3):
            flags.append("shotgun_findings")
            f1 *= 0.2
        success = cand == gt_set
        return self.score(tag_found=tag, parsed_ok=True, correctness=f1,
                          success=success, response=response,
                          components={"precision": round(prec, 4), "recall": round(rec, 4)},
                          hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_json_answer(
            [{"line": l, "rule": r} for l, r in instance.ground_truth["findings"]])

    def extra_canaries(self, instance: Instance):
        n = instance.ground_truth["n_lines"]
        rules = sorted(RULES)
        shotgun = [{"line": i, "rule": rules[i % len(rules)]}
                   for i in range(1, n + 1)]
        return [
            Canary("no_findings", wrap_json_answer([]), "claims the file is clean"),
            Canary("shotgun", wrap_json_answer(shotgun),
                   "flags every line with rotating rules"),
        ]
