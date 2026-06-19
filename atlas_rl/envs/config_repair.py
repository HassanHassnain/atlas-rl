"""config_repair: fix a broken YAML service config against a documented schema.

Observation: schema documentation + a YAML config with K injected bugs.
Action: the FULL corrected YAML config inside <answer> tags.
Reward: 0.55 * bugs fixed + 0.35 * untouched fields preserved + 0.10 * schema-valid.
Strict success: all bugs fixed, all other fields byte-preserved, schema valid.

Anti-hacking: dumping the schema defaults scores ~0 because original values are
sampled to differ from defaults; preservation is checked field-by-field.

Difficulty dials: number of bugs (1->4), bug subtlety, config size.
"""

from __future__ import annotations

import yaml

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import extract_answer, wrap_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

# schema: path -> (type, default, doc, validator-params)
ENUMS = {
    "log_level": ["debug", "info", "warn", "error"],
    "region": ["us-east", "us-west", "eu-central", "ap-south"],
    "compression": ["none", "gzip", "zstd"],
}
SCHEMA: dict[str, dict] = {
    "port":            {"type": "int", "default": 8080, "min": 1024, "max": 65535},
    "log_level":       {"type": "enum", "default": "info", "values": ENUMS["log_level"]},
    "region":          {"type": "enum", "default": "us-east", "values": ENUMS["region"]},
    "compression":     {"type": "enum", "default": "none", "values": ENUMS["compression"]},
    "timeout":         {"type": "duration", "default": "30s"},
    "flush_interval":  {"type": "duration", "default": "10s"},
    "retries":         {"type": "int", "default": 3, "min": 0, "max": 10},
    "buffer_size_mb":  {"type": "int", "default": 64, "min": 16, "max": 1024},
    "tls_enabled":     {"type": "bool", "default": False},
    "cert_path":       {"type": "path", "default": "/etc/logship/cert.pem"},
    "upstreams":       {"type": "hostport_list", "default": ["collector-1:9000"]},
}
_DUR_UNITS = ["ms", "s", "m"]
_HOSTS = ["collector", "ingest", "relay", "sink"]
_PARAMS = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}  # n_bugs
_SUBTLE = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}  # n subtle bugs among them


def _is_duration(v) -> bool:
    import re
    return isinstance(v, str) and re.fullmatch(r"\d+(ms|s|m)", v) is not None


def _is_hostport(v) -> bool:
    if not isinstance(v, str) or ":" not in v:
        return False
    host, _, port = v.rpartition(":")
    return bool(host) and port.isdigit() and 1 <= int(port) <= 65535


def _field_valid(key: str, v) -> bool:
    s = SCHEMA[key]
    t = s["type"]
    if t == "int":
        return isinstance(v, int) and not isinstance(v, bool) and s["min"] <= v <= s["max"]
    if t == "enum":
        return v in s["values"]
    if t == "duration":
        return _is_duration(v)
    if t == "bool":
        return isinstance(v, bool)
    if t == "path":
        return isinstance(v, str) and v.startswith("/")
    if t == "hostport_list":
        return (isinstance(v, list) and 1 <= len(v) <= 4
                and all(_is_hostport(x) for x in v))
    return False


def _schema_doc() -> str:
    lines = ["Field reference (all fields required, no other fields allowed):"]
    for k, s in SCHEMA.items():
        if s["type"] == "int":
            lines.append(f"- {k}: integer in [{s['min']}, {s['max']}] (default {s['default']})")
        elif s["type"] == "enum":
            lines.append(f"- {k}: one of {s['values']} (default {s['default']!r})")
        elif s["type"] == "duration":
            lines.append(f"- {k}: duration string matching <int><ms|s|m>, e.g. \"45s\" (default {s['default']!r})")
        elif s["type"] == "bool":
            lines.append(f"- {k}: boolean true/false (default {str(s['default']).lower()})")
        elif s["type"] == "path":
            lines.append(f"- {k}: absolute filesystem path (default {s['default']!r})")
        else:
            lines.append(f"- {k}: list of 1-4 \"host:port\" strings (default {s['default']})")
    return "\n".join(lines)


@register
class ConfigRepairEnv(AtlasEnv):
    env_id = "config_repair"
    name = "Service config repair"
    description = "Repair a broken logship-agent YAML config against its schema, preserving valid fields."
    answer_format = "the complete corrected YAML config (all fields, valid YAML)"
    difficulty_dials = {"n_bugs": "1 -> 4", "subtle_bugs": "0 -> 2"}

    # ------------------------------------------------------------ generation
    def _sample_value(self, rng, key):
        s = SCHEMA[key]
        t = s["type"]
        while True:
            if t == "int":
                v = rng.randint(s["min"], s["max"])
            elif t == "enum":
                v = rng.choice(s["values"])
            elif t == "duration":
                v = f"{rng.randint(1, 120)}{rng.choice(_DUR_UNITS)}"
            elif t == "bool":
                v = rng.random() < 0.5
            elif t == "path":
                v = f"/etc/{rng.choice(['logship', 'svc', 'agent'])}/{rng.choice(['cert', 'key', 'ca'])}-{rng.randint(1,99)}.pem"
            else:
                v = [f"{rng.choice(_HOSTS)}-{rng.randint(1,9)}:{rng.randint(7000, 9999)}"
                     for _ in range(rng.randint(1, 3))]
            if v != s["default"]:
                return v

    def _inject(self, rng, cfg: dict, key: str, subtle: bool) -> tuple[str, dict]:
        """Mutate cfg[key]; returns (bug_type, info)."""
        s = SCHEMA[key]
        t = s["type"]
        choices = []
        if t == "int":
            choices = [("type_error", lambda: str(cfg[key]) + " units"),
                       ("range_error", lambda: s["max"] * 10)]
        elif t == "enum":
            subtle_map = {"info": "information", "warn": "warning", "none": "off",
                          "us-east": "us-east-1", "gzip": "gz"}
            choices = [("enum_invalid", lambda: subtle_map.get(cfg[key], cfg[key] + "x")
                        if subtle else "verbose")]
        elif t == "duration":
            import re as _re
            choices = [("unit_error", lambda: cfg[key].rstrip("ms")
                        + rng.choice([" seconds", "sec", "minutes"])),
                       ("type_error", lambda: int(_re.match(r"\d+", cfg[key]).group()))]
        elif t == "bool":
            choices = [("bool_string", lambda: rng.choice(["yes", "enabled", "True'"]))]
        elif t == "path":
            choices = [("path_relative", lambda: cfg[key].lstrip("/"))]
        else:
            def bad_list():
                v = list(cfg[key])
                i = rng.randrange(len(v))
                host, _, port = v[i].rpartition(":")
                v[i] = host if rng.random() < 0.5 else f"{host}:{int(port) * 17}"
                return v
            choices = [("bad_upstream", bad_list)]
        bug_type, make = rng.choice(choices)
        original = cfg[key]
        cfg[key] = make()
        return bug_type, {"key": key, "original": original, "broken": cfg[key]}

    def _build(self, rng, difficulty):
        n_bugs = _PARAMS[difficulty]
        cfg = {k: self._sample_value(rng, k) for k in SCHEMA}
        original = {k: cfg[k] for k in cfg}

        keys = rng.sample(sorted(SCHEMA), n_bugs + 1)
        bugs = []
        # one optional structural bug at higher difficulty: missing or unknown key
        structural = difficulty >= 3 and rng.random() < 0.5
        if structural:
            mode = rng.choice(["missing_key", "unknown_key"])
            if mode == "missing_key":
                k = keys.pop()
                del cfg[k]
                bugs.append({"type": "missing_key", "key": k, "original": original[k]})
            else:
                keys.pop()
                bad_key = rng.choice(["loglevel", "time_out", "buffer_mb", "regions"])
                cfg[bad_key] = "true"
                bugs.append({"type": "unknown_key", "key": bad_key, "original": None})
            n_bugs -= 1 if n_bugs > 1 else 0
        n_subtle = _SUBTLE[difficulty]
        for i, k in enumerate(keys[:n_bugs]):
            bt, info = self._inject(rng, cfg, k, subtle=i < n_subtle)
            bugs.append({"type": bt, **info})

        broken_yaml = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        prompt = (
            "The logship-agent on a production host fails to start because its "
            "config is invalid. Repair it.\n\n"
            f"{_schema_doc()}\n\n"
            "Rules:\n"
            "- Fix ONLY what is broken; preserve every valid field exactly as-is.\n"
            "- If a required field is missing, restore it with its documented default.\n"
            "- Remove fields that are not in the schema.\n\n"
            f"Broken config (config.yaml):\n```yaml\n{broken_yaml}```\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"original": original, "bugs": bugs}
        meta = {"complexity": difficulty * 10 + len(bugs) * 5, "n_bugs": len(bugs)}
        return prompt, gt, meta

    # ----------------------------------------------------------- verification
    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, content = extract_answer(response)
        flags: list[str] = []
        original = instance.ground_truth["original"]
        bugs = instance.ground_truth["bugs"]
        parsed_ok = False
        fixed = preserved = valid = 0.0
        try:
            cand = yaml.safe_load(content) if content else None
            parsed_ok = isinstance(cand, dict)
        except yaml.YAMLError:
            cand = None
        if parsed_ok:
            bug_keys = {b["key"] for b in bugs}
            # bugs fixed
            pts = []
            for b in bugs:
                if b["type"] == "unknown_key":
                    pts.append(1.0 if b["key"] not in cand else 0.0)
                elif b["type"] == "missing_key":
                    ok_vals = (original[b["key"]], SCHEMA[b["key"]]["default"])
                    pts.append(1.0 if cand.get(b["key"]) in ok_vals else 0.0)
                else:
                    pts.append(1.0 if cand.get(b["key"]) == original[b["key"]] else 0.0)
            fixed = sum(pts) / len(pts)
            # preservation of untouched fields
            keep_keys = [k for k in original if k not in bug_keys]
            kept = [1.0 if cand.get(k) == original[k] else 0.0 for k in keep_keys]
            preserved = sum(kept) / len(kept) if kept else 1.0
            # schema validity
            valid = 1.0 if (
                set(cand) == set(SCHEMA)
                and all(_field_valid(k, v) for k, v in cand.items())
            ) else 0.0
            if valid and fixed < 0.01 and preserved < 0.3:
                flags.append("schema_valid_but_unrelated_config")
        # Multiplicative gating: preservation/validity credit only flows through
        # actual bug-fixing, so no-op or defaults-dump answers earn ~0.
        corr = fixed * (0.85 * preserved + 0.15 * valid)
        success = fixed >= 0.999 and preserved >= 0.999 and valid >= 0.999
        return self.score(tag_found=tag, parsed_ok=parsed_ok, correctness=corr,
                          success=success, response=response,
                          components={"fixed": fixed, "preserved": preserved, "valid": valid},
                          hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        fixed = dict(instance.ground_truth["original"])
        return wrap_answer(yaml.safe_dump(fixed, sort_keys=False, default_flow_style=False))

    def extra_canaries(self, instance: Instance):
        defaults = {k: s["default"] for k, s in SCHEMA.items()}
        return [
            Canary("schema_defaults",
                   wrap_answer(yaml.safe_dump(defaults, sort_keys=False)),
                   "ignores the broken file and submits the documented defaults"),
            Canary("resubmit_broken",
                   wrap_answer(instance.prompt.split("```yaml\n")[1].split("```")[0]),
                   "submits the broken config unchanged"),
        ]
