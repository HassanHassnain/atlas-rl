"""k8s_doctor: repair Kubernetes manifests via a restricted patch language.

Observation: Deployment + Service + ConfigMap YAML with K violations of a
documented production-policy checklist.
Action: JSON array of patch ops:
    {"op": "set",    "path": "deployment.spec.replicas", "value": 2}
    {"op": "remove", "path": "service.spec.selector.tier"}
Paths are dot-separated; list elements use [i]. Verification applies the patch
and re-runs ALL policy checks (semantic: fixing either side of a mismatch
counts), checks that frozen identity fields are untouched, and penalizes
regressions on checks that passed before.
Reward: fixed_frac * (0.8 + 0.1*frozen_ok + 0.1*ops_valid), x0.3 if any
previously-passing check was broken. Strict: everything passes, nothing broken.

Difficulty dials: number of seeded violations (1 -> 4), manifest size.
"""

from __future__ import annotations

import copy
import re

import yaml

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import parse_json_answer, wrap_json_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

CHECKS_DOC = {
    "replicas_min": "Deployment must run at least 2 replicas in production.",
    "selector_match": "Service spec.selector must equal the Deployment's pod template labels, and the Deployment's spec.selector.matchLabels must too.",
    "targetport_match": "Service ports[0].targetPort must equal the container's ports[0].containerPort.",
    "probe_port_match": "The livenessProbe httpGet port must equal the containerPort.",
    "cm_keys_exist": "Every env var configMapKeyRef key must exist in the ConfigMap's data.",
    "pull_policy": "imagePullPolicy must be IfNotPresent or Always.",
    "limits_present": "The container must set resources.limits.cpu and resources.limits.memory.",
}
_PARAMS = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}
_PATH_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)((?:\[\d+\])*)$")


# ----------------------------------------------------------------- patch ops
def _parse_path(path: str) -> list:
    out: list = []
    for raw in path.split("."):
        m = _PATH_TOKEN.match(raw.strip())
        if not m:
            raise ValueError(f"bad path token {raw!r}")
        out.append(m.group(1))
        for idx in re.findall(r"\[(\d+)\]", m.group(2)):
            out.append(int(idx))
    return out


def apply_op(root: dict, op: dict) -> bool:
    try:
        kind = op["op"]
        toks = _parse_path(op["path"])
        if toks[0] not in ("deployment", "service", "configmap") or len(toks) < 2:
            return False
        node = root
        for i, t in enumerate(toks[:-1]):
            nxt = toks[i + 1]
            if isinstance(t, int):
                if not isinstance(node, list) or t >= len(node):
                    return False
                node = node[t]
            else:
                if not isinstance(node, dict):
                    return False
                if t not in node:
                    if kind == "set" and not isinstance(nxt, int):
                        node[t] = {}
                    else:
                        return False
                node = node[t]
        last = toks[-1]
        if kind == "set":
            if "value" not in op:
                return False
            if isinstance(last, int):
                if not isinstance(node, list) or last >= len(node):
                    return False
                node[last] = op["value"]
            else:
                if not isinstance(node, dict):
                    return False
                node[last] = op["value"]
            return True
        if kind == "remove":
            if isinstance(last, int):
                if isinstance(node, list) and last < len(node):
                    del node[last]
                    return True
                return False
            if isinstance(node, dict) and last in node:
                del node[last]
                return True
            return False
        return False
    except (KeyError, TypeError, ValueError):
        return False


def _dig(d, *keys, default=None):
    for k in keys:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return default
    return d


# ------------------------------------------------------------------- checks
def run_checks(m: dict) -> dict[str, bool]:
    dep, svc, cm = m.get("deployment", {}), m.get("service", {}), m.get("configmap", {})
    c0 = _dig(dep, "spec", "template", "spec", "containers", 0, default={}) or {}
    tmpl_labels = _dig(dep, "spec", "template", "metadata", "labels")
    cport = _dig(c0, "ports", 0, "containerPort")
    out = {}
    r = _dig(dep, "spec", "replicas")
    out["replicas_min"] = isinstance(r, int) and r >= 2
    sel = _dig(svc, "spec", "selector")
    mls = _dig(dep, "spec", "selector", "matchLabels")
    out["selector_match"] = (isinstance(tmpl_labels, dict) and bool(tmpl_labels)
                             and sel == tmpl_labels and mls == tmpl_labels)
    tport = _dig(svc, "spec", "ports", 0, "targetPort")
    out["targetport_match"] = cport is not None and tport == cport
    pport = _dig(c0, "livenessProbe", "httpGet", "port")
    out["probe_port_match"] = cport is not None and pport == cport
    data = _dig(cm, "data", default={}) or {}
    refs = []
    for env in _dig(c0, "env", default=[]) or []:
        ref = _dig(env, "valueFrom", "configMapKeyRef")
        if ref:
            refs.append(ref)
    # non-vacuous: deleting the env block does NOT satisfy this check
    out["cm_keys_exist"] = bool(refs) and all(
        isinstance(r, dict) and r.get("key") in data for r in refs)
    out["pull_policy"] = c0.get("imagePullPolicy") in ("IfNotPresent", "Always")
    lim = _dig(c0, "resources", "limits", default={}) or {}
    out["limits_present"] = bool(lim.get("cpu")) and bool(lim.get("memory"))
    return out


@register
class K8sDoctorEnv(AtlasEnv):
    env_id = "k8s_doctor"
    name = "Kubernetes manifest repair"
    description = "Patch Deployment/Service/ConfigMap manifests to satisfy a production policy checklist."
    answer_format = ('JSON array of patch ops, e.g. '
                     '[{"op": "set", "path": "deployment.spec.replicas", "value": 3}]')
    difficulty_dials = {"n_violations": "1 -> 4", "manifest_size": "grows with difficulty"}

    def _healthy(self, rng, difficulty) -> dict:
        app = rng.choice(["checkout", "ingest", "ratings", "webhooks", "search-api"])
        port = rng.choice([8080, 8000, 9090, 3000])
        labels = {"app": app, "tier": rng.choice(["backend", "api"])}
        cm_keys = rng.sample(["DATABASE_URL", "REDIS_URL", "FEATURE_FLAGS",
                              "LOG_LEVEL", "QUEUE_NAME"], 3 if difficulty >= 3 else 2)
        envs = [{"name": k, "valueFrom": {"configMapKeyRef": {"name": f"{app}-config",
                                                              "key": k}}}
                for k in cm_keys[:2]]
        container = {
            "name": app,
            "image": f"registry.internal/{app}:{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}",
            "imagePullPolicy": rng.choice(["IfNotPresent", "Always"]),
            "ports": [{"containerPort": port}],
            "env": envs,
            "livenessProbe": {"httpGet": {"path": "/healthz", "port": port},
                              "initialDelaySeconds": rng.choice([5, 10, 15])},
            "resources": {"requests": {"cpu": "100m", "memory": "128Mi"},
                          "limits": {"cpu": rng.choice(["500m", "1"]),
                                     "memory": rng.choice(["512Mi", "1Gi"])}},
        }
        dep = {"apiVersion": "apps/v1", "kind": "Deployment",
               "metadata": {"name": app, "namespace": "prod"},
               "spec": {"replicas": rng.randint(2, 5),
                        "selector": {"matchLabels": dict(labels)},
                        "template": {"metadata": {"labels": dict(labels)},
                                     "spec": {"containers": [container]}}}}
        svc = {"apiVersion": "v1", "kind": "Service",
               "metadata": {"name": f"{app}-svc", "namespace": "prod"},
               "spec": {"selector": dict(labels),
                        "ports": [{"port": 80, "targetPort": port}]}}
        cm = {"apiVersion": "v1", "kind": "ConfigMap",
              "metadata": {"name": f"{app}-config", "namespace": "prod"},
              "data": {k: f"value-{rng.randint(100, 999)}" for k in cm_keys}}
        return {"deployment": dep, "service": svc, "configmap": cm}

    def _break_check(self, rng, m: dict, check: str) -> list[dict]:
        """Mutates m to violate `check`; returns oracle patch ops that fix it."""
        dep, svc = m["deployment"], m["service"]
        c0 = dep["spec"]["template"]["spec"]["containers"][0]
        port = c0["ports"][0]["containerPort"]
        if check == "replicas_min":
            old = dep["spec"]["replicas"]
            dep["spec"]["replicas"] = 0
            return [{"op": "set", "path": "deployment.spec.replicas", "value": old}]
        if check == "selector_match":
            good = dict(svc["spec"]["selector"])
            svc["spec"]["selector"] = {**good, "app": good["app"] + "-svc"}
            return [{"op": "set", "path": "service.spec.selector.app",
                     "value": good["app"]}]
        if check == "targetport_match":
            svc["spec"]["ports"][0]["targetPort"] = port + 1
            return [{"op": "set", "path": "service.spec.ports[0].targetPort",
                     "value": port}]
        if check == "probe_port_match":
            c0["livenessProbe"]["httpGet"]["port"] = port + 100
            return [{"op": "set",
                     "path": "deployment.spec.template.spec.containers[0].livenessProbe.httpGet.port",
                     "value": port}]
        if check == "cm_keys_exist":
            good_key = c0["env"][0]["valueFrom"]["configMapKeyRef"]["key"]
            c0["env"][0]["valueFrom"]["configMapKeyRef"]["key"] = good_key + "_PROD"
            return [{"op": "set",
                     "path": "deployment.spec.template.spec.containers[0].env[0].valueFrom.configMapKeyRef.key",
                     "value": good_key}]
        if check == "pull_policy":
            old = c0["imagePullPolicy"]
            c0["imagePullPolicy"] = "Never"
            return [{"op": "set",
                     "path": "deployment.spec.template.spec.containers[0].imagePullPolicy",
                     "value": old}]
        if check == "limits_present":
            lim = dict(c0["resources"]["limits"])
            del c0["resources"]["limits"]
            return [{"op": "set",
                     "path": "deployment.spec.template.spec.containers[0].resources.limits",
                     "value": lim}]
        raise ValueError(check)

    def _build(self, rng, difficulty):
        n_bugs = _PARAMS[difficulty]
        m = self._healthy(rng, difficulty)
        broken_checks = rng.sample(sorted(CHECKS_DOC), n_bugs)
        oracle_ops: list[dict] = []
        for ch in broken_checks:
            oracle_ops += self._break_check(rng, m, ch)
        frozen = {
            "deployment.metadata.name": m["deployment"]["metadata"]["name"],
            "service.metadata.name": m["service"]["metadata"]["name"],
            "configmap.metadata.name": m["configmap"]["metadata"]["name"],
            "deployment...containers[0].name":
                m["deployment"]["spec"]["template"]["spec"]["containers"][0]["name"],
            "deployment...containers[0].image":
                m["deployment"]["spec"]["template"]["spec"]["containers"][0]["image"],
            "service.spec.ports[0].port": m["service"]["spec"]["ports"][0]["port"],
        }
        checks_doc = "\n".join(f"- {k}: {v}" for k, v in sorted(CHECKS_DOC.items()))
        blocks = "\n---\n".join(
            yaml.safe_dump(m[k], sort_keys=False) for k in ("deployment", "service", "configmap"))
        prompt = (
            "These manifests fail our production policy. Produce a minimal patch "
            "that makes ALL policy checks pass. Do not rename resources, containers "
            "or images, and do not change the Service's external port.\n\n"
            f"Production policy checklist:\n{checks_doc}\n\n"
            f"Manifests:\n```yaml\n{blocks}```\n\n"
            "Patch language: a JSON array of ops over the roots `deployment`, "
            "`service`, `configmap`. Each op is {\"op\": \"set\"|\"remove\", "
            "\"path\": \"dot.path[with][indexes]\", \"value\": <json, for set>}. "
            "Values must use correct JSON types (numbers as numbers).\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"manifests": m, "broken_checks": broken_checks,
              "oracle_ops": oracle_ops, "frozen": frozen}
        meta = {"complexity": difficulty * 10 + n_bugs * 6}
        return prompt, gt, meta

    def _frozen_ok(self, patched: dict, frozen_gt: dict) -> float:
        m = patched
        vals = {
            "deployment.metadata.name": _dig(m, "deployment", "metadata", "name"),
            "service.metadata.name": _dig(m, "service", "metadata", "name"),
            "configmap.metadata.name": _dig(m, "configmap", "metadata", "name"),
            "deployment...containers[0].name":
                _dig(m, "deployment", "spec", "template", "spec", "containers", 0, "name"),
            "deployment...containers[0].image":
                _dig(m, "deployment", "spec", "template", "spec", "containers", 0, "image"),
            "service.spec.ports[0].port": _dig(m, "service", "spec", "ports", 0, "port"),
        }
        hits = sum(1 for k, v in frozen_gt.items() if vals.get(k) == v)
        return hits / len(frozen_gt)

    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, ok, obj = parse_json_answer(response)
        gt = instance.ground_truth
        if not (ok and isinstance(obj, list) and all(isinstance(x, dict) for x in obj)):
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response)
        flags: list[str] = []
        if len(obj) > 25:
            flags.append("patch_spam")
            obj = obj[:25]
        patched = copy.deepcopy(gt["manifests"])
        before = run_checks(gt["manifests"])
        valid_ops = sum(1 for op in obj if apply_op(patched, op))
        ops_valid = valid_ops / len(obj) if obj else 0.0
        after = run_checks(patched)
        broken = gt["broken_checks"]
        fixed = sum(1 for c in broken if after[c]) / len(broken)
        regressions = [c for c, was in before.items() if was and not after[c]]
        frozen = self._frozen_ok(patched, gt["frozen"])
        corr = fixed * (0.8 + 0.1 * (frozen >= 0.999) + 0.1 * (ops_valid >= 0.999))
        if regressions:
            corr *= 0.3
            flags.append("policy_regression:" + ",".join(regressions[:3]))
        success = (all(after.values()) and not regressions
                   and frozen >= 0.999 and ops_valid >= 0.999 and bool(obj))
        return self.score(tag_found=tag, parsed_ok=True, correctness=corr,
                          success=success, response=response,
                          components={"fixed": round(fixed, 4), "frozen": round(frozen, 4),
                                      "ops_valid": round(ops_valid, 4)},
                          hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_json_answer(instance.ground_truth["oracle_ops"])

    def extra_canaries(self, instance: Instance):
        gt = instance.ground_truth
        app = gt["manifests"]["deployment"]["metadata"]["name"]
        return [
            Canary("empty_patch", wrap_json_answer([]), "submits no ops"),
            Canary("rename_everything",
                   wrap_json_answer([
                       {"op": "set", "path": "deployment.metadata.name", "value": app + "-v2"},
                       {"op": "set", "path": "service.metadata.name", "value": app + "-v2"}]),
                   "renames resources instead of fixing policy"),
            Canary("delete_spec",
                   wrap_json_answer([{"op": "remove", "path": "deployment.spec"}]),
                   "deletes the deployment spec"),
        ]
