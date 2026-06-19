"""log_triage: root-cause service identification from multi-service logs.

Observation: an interleaved log window from N microservices with one injected
root-cause failure that cascades to dependent services, plus decoy transient
errors that recover.
Action: JSON {"service": <name>, "category": <one of CATEGORIES>}.
Reward: 0.6 * correct root service + 0.4 * correct failure category.
Strict success: both correct.

Difficulty dials: number of services, log volume, cascade depth, number of
recovered decoy errors, noise level.
"""

from __future__ import annotations

import json

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import parse_json_answer, wrap_json_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

CATEGORIES = {
    "oom_kill": "process killed by the OOM killer / memory exhaustion",
    "disk_full": "no space left on device",
    "conn_refused": "TCP connection refused to a dependency",
    "dns_failure": "DNS resolution failure",
    "tls_expired": "TLS certificate expired",
    "deadlock": "database / lock wait deadlock",
}

_ROOT_LINES = {
    "oom_kill": ["FATAL worker pid={pid} killed: Out of memory (oom-killer invoked)",
                 "ERROR memory cgroup limit exceeded, rss={rss}MB"],
    "disk_full": ["ERROR write failed on /var/lib/{svc}/data: No space left on device",
                  "FATAL cannot append to WAL: disk usage 100%"],
    "conn_refused": ["ERROR dial tcp 10.0.{oct}.7:{port}: connect: connection refused",
                     "ERROR upstream {dep} unreachable: connection refused"],
    "dns_failure": ["ERROR lookup {dep}.svc.cluster.local: no such host",
                    "ERROR getaddrinfo EAI_AGAIN {dep}"],
    "tls_expired": ["ERROR tls handshake with {dep}: x509: certificate has expired",
                    "FATAL cert /etc/certs/{svc}.pem expired 14h ago"],
    "deadlock": ["ERROR Deadlock found when trying to get lock; try restarting transaction",
                 "FATAL lock wait timeout exceeded on table jobs"],
}

_SYMPTOM_LINES = [
    "ERROR request to {dep} failed: 503 Service Unavailable",
    "ERROR call {dep}.GetStatus timed out after 5000ms",
    "WARN circuit breaker OPEN for {dep}",
    "ERROR 5xx ratio from {dep} above threshold (0.42)",
]

_NOISE = [
    "INFO request handled path=/api/v1/{path} status=200 dur={ms}ms",
    "INFO gc pause {ms}ms",
    "DEBUG cache hit ratio 0.{r2}",
    "INFO healthcheck ok",
    "WARN slow query {ms}ms on index users_by_email",
    "INFO rotated log file",
]

_DECOY = [
    ("ERROR transient: read tcp 10.0.{oct}.9:{port}: i/o timeout (attempt 1/3)",
     "INFO retry succeeded for previous operation (attempt 2/3)"),
    ("WARN dropped 3 metrics datapoints (buffer full)",
     "INFO metrics buffer drained, backlog 0"),
    ("ERROR temporary failure publishing event, will retry",
     "INFO event published after retry"),
]

_SVC_POOL = ["gateway", "auth", "billing", "orders", "payments", "inventory",
             "search", "notify", "users", "ledger", "shipping", "catalog"]

_PARAMS = {1: (3, 40, 1, 0), 2: (5, 70, 1, 1), 3: (7, 110, 2, 1),
           4: (9, 160, 2, 2), 5: (12, 230, 3, 3)}


@register
class LogTriageEnv(AtlasEnv):
    env_id = "log_triage"
    name = "Incident log triage"
    description = "Identify the root-cause service and failure category from interleaved microservice logs."
    answer_format = 'JSON object: {"service": "<service-name>", "category": "<category-id>"}'
    difficulty_dials = {
        "n_services": "3 -> 12", "n_lines": "40 -> 230",
        "cascade_depth": "1 -> 3", "n_decoys": "0 -> 3",
    }

    def _build(self, rng, difficulty):
        n_svc, n_lines, cascade, n_decoys = _PARAMS[difficulty]
        services = rng.sample(_SVC_POOL, n_svc)
        root = rng.choice(services)
        # dependency chain: dependents[0] depends on root, etc.
        others = [s for s in services if s != root]
        dependents = rng.sample(others, min(cascade, len(others)))
        category = rng.choice(sorted(CATEGORIES))

        t0 = rng.randint(20, max(21, n_lines - 15))  # root error position
        lines: list[tuple[int, str, str]] = []  # (tick, svc, text)

        def fmt(template, svc, dep=""):
            return template.format(
                svc=svc, dep=dep, pid=rng.randint(100, 9999),
                rss=rng.randint(900, 4000), oct=rng.randint(0, 254),
                port=rng.choice([5432, 6379, 9092, 8080, 443]),
                path=rng.choice(["users", "orders", "items", "login"]),
                ms=rng.randint(1, 900), r2=rng.randint(10, 99),
            )

        # noise everywhere
        for tick in range(n_lines):
            svc = rng.choice(services)
            lines.append((tick, svc, fmt(rng.choice(_NOISE), svc)))
        # decoys (error followed by recovery), strictly before t0
        for _ in range(n_decoys):
            dsvc = rng.choice([s for s in services if s != root])
            dt = rng.randint(2, max(3, t0 - 6))
            err, rec = rng.choice(_DECOY)
            lines.append((dt, dsvc, fmt(err, dsvc)))
            lines.append((dt + 2, dsvc, rec))
        # root cause
        dep_of_root = rng.choice([s for s in services if s != root])
        lines.append((t0, root, fmt(rng.choice(_ROOT_LINES[category]), root, dep_of_root)))
        # cascade symptoms after t0, each dependent blames the chain upstream
        chain = [root] + dependents
        for i, dsvc in enumerate(dependents):
            upstream = chain[i]
            for k in range(rng.randint(1, 2)):
                lines.append((t0 + 2 + 3 * i + k, dsvc,
                              fmt(rng.choice(_SYMPTOM_LINES), dsvc, upstream)))

        lines.sort(key=lambda x: (x[0], x[1]))
        rendered = []
        base_min = rng.randint(0, 30)
        for tick, svc, text in lines:
            mm = (base_min + tick) // 60
            ss = (base_min + tick) % 60
            rendered.append(f"2026-03-14T09:{mm:02d}:{ss:02d}Z [{svc:<9}] {text}")
        log_block = "\n".join(rendered)

        cat_doc = "\n".join(f"- {k}: {v}" for k, v in sorted(CATEGORIES.items()))
        prompt = (
            "An incident is in progress. Below is the merged log window from all "
            f"services ({', '.join(sorted(services))}).\n\n"
            "Find the ROOT-CAUSE service (the service whose own failure started the "
            "incident, not services failing because a dependency failed, and not "
            "transient errors that recovered) and classify the failure.\n\n"
            f"Failure categories:\n{cat_doc}\n\n"
            f"Logs:\n```\n{log_block}\n```\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"service": root, "category": category}
        meta = {"complexity": n_svc * 10 + n_lines,
                "services": sorted(services), "cascade": dependents}
        return prompt, gt, meta

    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, ok, obj = parse_json_answer(response)
        flags: list[str] = []
        svc_pts = cat_pts = 0.0
        if ok and isinstance(obj, dict):
            svc = str(obj.get("service", "")).strip().lower()
            cat = str(obj.get("category", "")).strip().lower()
            gt = instance.ground_truth
            if svc == gt["service"]:
                svc_pts = 1.0
            if cat == gt["category"]:
                cat_pts = 1.0
            if svc not in [s.lower() for s in instance.metadata["services"]] and svc:
                flags.append("hallucinated_service")
        else:
            ok = False
        corr = 0.6 * svc_pts + 0.4 * cat_pts
        return self.score(tag_found=tag, parsed_ok=ok, correctness=corr,
                          success=corr >= 0.999, response=response,
                          components={"service": svc_pts, "category": cat_pts},
                          hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_json_answer(instance.ground_truth)

    def extra_canaries(self, instance: Instance):
        svcs = instance.metadata["services"]
        wrong_cat = next(c for c in sorted(CATEGORIES)
                         if c != instance.ground_truth["category"])
        victim = (instance.metadata["cascade"] or ["___none___"])[-1]
        return [
            Canary("all_services",
                   f"<answer>{json.dumps({'service': ' or '.join(svcs), 'category': wrong_cat})}</answer>",
                   "hedges by naming every service"),
            Canary("loudest_symptom",
                   f"<answer>{json.dumps({'service': victim, 'category': wrong_cat})}</answer>",
                   "names the last cascading victim instead of the root cause"),
        ]
