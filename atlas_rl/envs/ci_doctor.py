"""ci_doctor: CI pipeline failure triage.

Observation: a CI run log (multiple stages, one true failure, decoy warnings,
optionally an allowed-to-fail stage that also fails).
Action: JSON {"stage": ..., "cause": ..., "action": ...} using documented vocab.
Reward: 0.35*stage + 0.35*cause + 0.30*action. Strict: all three correct.

Difficulty dials: number of stages, decoy warnings, allowed-failure stages,
log verbosity.
"""

from __future__ import annotations

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import parse_json_answer, wrap_json_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

# cause -> (action, error lines rendered in the failing stage)
CATALOG: dict[str, dict] = {
    "missing_module": {
        "action": "add-dependency",
        "lines": ["Traceback (most recent call last):",
                  "  File \"{file}.py\", line {ln}, in <module>",
                  "    import {mod}",
                  "ModuleNotFoundError: No module named '{mod}'"]},
    "failing_test": {
        "action": "fix-test",
        "lines": ["FAILED tests/test_{file}.py::test_{fn} - AssertionError: expected {a} got {b}",
                  "=========== 1 failed, {n} passed in {s}s ==========="]},
    "syntax_error": {
        "action": "fix-syntax",
        "lines": ["  File \"src/{file}.py\", line {ln}",
                  "    def {fn}(:",
                  "        ^",
                  "SyntaxError: invalid syntax"]},
    "oom": {
        "action": "increase-memory",
        "lines": ["g++: fatal error: Killed signal terminated program cc1plus",
                  "virtual memory exhausted: Cannot allocate memory"]},
    "network_flake": {
        "action": "retry-job",
        "lines": ["WARNING: Retrying (Retry(total=0)) after connection broken",
                  "ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out."]},
    "version_conflict": {
        "action": "pin-dependency",
        "lines": ["ERROR: pip's dependency resolver found conflicting requirements:",
                  "  {mod} {a}.0 requires urllib3<2, but you have urllib3 2.{b}.0"]},
    "disk_full": {
        "action": "free-disk-space",
        "lines": ["OSError: [Errno 28] No space left on device",
                  "error: could not write to artifact cache"]},
}
ACTIONS = sorted({v["action"] for v in CATALOG.values()})
STAGE_POOL = ["lint", "typecheck", "build", "unit-test", "integration-test",
              "package", "scan", "deploy-staging"]
_WARNS = ["warning: unused variable 'tmp{n}'",
          "DeprecationWarning: pkg_resources is deprecated",
          "warning: cache restore took {s}s (slow)",
          "note: 2 vulnerabilities found (0 high) — see report"]
_INFO = ["$ {cmd}", "Collecting wheel-{n}", "Compiling module {n}/40",
         "OK ({s}s)", "Restored cache key ci-cache-v{n}"]
_PARAMS = {1: (3, 1, 0, 4), 2: (4, 2, 0, 6), 3: (5, 3, 1, 8),
           4: (6, 4, 1, 11), 5: (8, 6, 2, 14)}  # stages, warns, allowed_fail, verbosity


@register
class CiDoctorEnv(AtlasEnv):
    env_id = "ci_doctor"
    name = "CI failure triage"
    description = "Identify the failing stage, root cause class, and remediation action from a CI run log."
    answer_format = ('JSON object: {"stage": "<stage-name>", "cause": "<cause-id>", '
                     '"action": "<action-id>"}')
    difficulty_dials = {"n_stages": "3 -> 8", "decoy_warnings": "1 -> 6",
                        "allowed_fail_stages": "0 -> 2", "verbosity": "4 -> 14"}

    def _build(self, rng, difficulty):
        n_stages, n_warn, n_allowed, verbosity = _PARAMS[difficulty]
        stages = STAGE_POOL[:n_stages]
        fail_idx = rng.randrange(1, n_stages)  # never the first stage (too easy)
        cause = rng.choice(sorted(CATALOG))
        allowed_idxs = set()
        candidates = [i for i in range(n_stages) if i != fail_idx and i < fail_idx]
        rng.shuffle(candidates)
        allowed_idxs = set(candidates[:n_allowed])
        allowed_cause = rng.choice([c for c in sorted(CATALOG) if c != cause])

        def fill(t):
            return t.format(
                file=rng.choice(["utils", "models", "client", "worker", "parser"]),
                fn=rng.choice(["flush", "merge", "rollup", "login", "sync"]),
                mod=rng.choice(["httpx", "pydanticx", "redisq", "yamlcore"]),
                ln=rng.randint(3, 220), n=rng.randint(2, 40),
                s=rng.randint(2, 95), a=rng.randint(1, 9), b=rng.randint(0, 9),
                cmd=rng.choice(["make build", "pytest -x", "pip install -r requirements.txt",
                                "ruff check .", "docker build ."]),
            )

        out = [f"CI RUN #{rng.randint(1000, 9999)} on branch "
               f"{rng.choice(['main', 'develop', 'feat/retry-queue', 'fix/cache-key'])}"]
        warn_budget = n_warn
        for i, st in enumerate(stages):
            allowed = i in allowed_idxs
            out.append(f"=== STAGE: {st}{' (allow_failure: true)' if allowed else ''} ===")
            for _ in range(rng.randint(2, verbosity)):
                out.append(fill(rng.choice(_INFO)))
            # sprinkle decoy warnings into passing stages
            if warn_budget > 0 and i != fail_idx and rng.random() < 0.8:
                out.append(fill(rng.choice(_WARNS)))
                warn_budget -= 1
            if allowed:
                for ln in CATALOG[allowed_cause]["lines"]:
                    out.append(fill(ln))
                out.append(f"=== RESULT: {st} FAILED (allowed, exit {rng.randint(1,2)}) ===")
            elif i == fail_idx:
                for ln in CATALOG[cause]["lines"]:
                    out.append(fill(ln))
                out.append(f"=== RESULT: {st} FAILED (exit {rng.randint(1, 2)}) ===")
                break
            else:
                out.append(f"=== RESULT: {st} passed ===")
        out.append("PIPELINE FAILED")
        log = "\n".join(out)

        cause_doc = "\n".join(f"- {c} -> recommended action: {CATALOG[c]['action']}"
                              for c in sorted(CATALOG))
        prompt = (
            "A CI pipeline failed. Triage it: name the stage whose failure broke the "
            "pipeline (stages marked allow_failure:true do NOT break the pipeline), "
            "the root-cause class, and the recommended action.\n\n"
            f"Cause classes and their actions:\n{cause_doc}\n\n"
            f"CI log:\n```\n{log}\n```\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"stage": stages[fail_idx], "cause": cause,
              "action": CATALOG[cause]["action"]}
        meta = {"complexity": n_stages * 10 + verbosity, "stages": stages,
                "allowed_failed": [stages[i] for i in sorted(allowed_idxs)]}
        return prompt, gt, meta

    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, ok, obj = parse_json_answer(response)
        s = c = a = 0.0
        if ok and isinstance(obj, dict):
            gt = instance.ground_truth
            if str(obj.get("stage", "")).strip().lower() == gt["stage"]:
                s = 1.0
            if str(obj.get("cause", "")).strip().lower() == gt["cause"]:
                c = 1.0
            if str(obj.get("action", "")).strip().lower() == gt["action"]:
                a = 1.0
        else:
            ok = False
        corr = 0.35 * s + 0.35 * c + 0.30 * a
        return self.score(tag_found=tag, parsed_ok=ok, correctness=corr,
                          success=corr >= 0.999, response=response,
                          components={"stage": s, "cause": c, "action": a})

    def oracle(self, instance: Instance) -> str:
        return wrap_json_answer(instance.ground_truth)

    def extra_canaries(self, instance: Instance):
        gt = instance.ground_truth
        wrong_cause = next(x for x in sorted(CATALOG) if x != gt["cause"])
        wrong_action = CATALOG[wrong_cause]["action"]
        if wrong_action == gt["action"]:
            wrong_cause = next(x for x in sorted(CATALOG)
                               if x != gt["cause"] and CATALOG[x]["action"] != gt["action"])
            wrong_action = CATALOG[wrong_cause]["action"]
        decoy_stage = (instance.metadata["allowed_failed"] or ["__none__"])[0]
        return [
            Canary("allowed_failure_stage",
                   wrap_json_answer({"stage": decoy_stage, "cause": wrong_cause,
                                     "action": wrong_action}),
                   "blames the allow_failure stage"),
            Canary("retry_everything",
                   wrap_json_answer({"stage": "__pipeline__", "cause": "flaky",
                                     "action": "retry"}),
                   "generic 'just retry' non-answer"),
        ]
