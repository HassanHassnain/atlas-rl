"""regex_extract: author one regex that extracts a target field from log lines.

Observation: an extraction spec + sample log lines (the verifier holds
positives with gold spans and decoy negatives).
Action: a single Python-flavored regular expression.
Verification: the regex is run (with a timeout guard) on every line. On
positive lines the extracted span (group 1 if present, else group 0) must equal
the gold span; on negative (decoy) lines it must not match at all.
Reward: F1^2 over lines (squared to deny credit to trivial catch-all patterns).
Strict success: F1 == 1.

Anti-hacking: `.*` scores 0 (span mismatch + decoy matches); catastrophic
backtracking is cut off by a regex timeout and flagged (`redos`).

Difficulty dials: number of lines, decoy families, context variety.
"""

from __future__ import annotations

try:  # `regex` supports match timeouts; declared in pyproject.toml
    import regex as _rx
    _HAS_TIMEOUT = True
except ImportError:  # pragma: no cover
    import re as _rx
    _HAS_TIMEOUT = False

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import extract_answer, wrap_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

_HEX = "0123456789ABCDEF"
_PARAMS = {1: (8, 6, 1), 2: (10, 8, 2), 3: (12, 10, 2), 4: (14, 12, 3), 5: (16, 14, 3)}


def _safe_search(pattern: str, line: str):
    if _HAS_TIMEOUT:
        return _rx.search(pattern, line, timeout=0.05)
    return _rx.search(pattern, line)


class _Family:
    fid = ""
    spec = ""
    reference = ""  # an oracle regex

    def positive(self, rng) -> tuple[str, str]:
        raise NotImplementedError

    def negative(self, rng, variant: int) -> str:
        raise NotImplementedError


class _ReqId(_Family):
    fid = "request_id"
    spec = ("Extract request IDs. A request ID is the literal prefix 'req-' "
            "followed by exactly 8 uppercase hex characters [0-9A-F]. "
            "Report the full ID including the 'req-' prefix.")
    reference = r"req-[0-9A-F]{8}\b"

    def positive(self, rng):
        rid = "req-" + "".join(rng.choice(_HEX) for _ in range(8))
        ctx = rng.choice([f"trace {rid} started by scheduler",
                          f"[{rid}] retrying upstream call",
                          f"completed id={rid} in 84ms",
                          f"GET /api/orders {rid} 200"])
        return ctx, rid

    def negative(self, rng, variant):
        if variant == 0:  # lowercase hex
            chars = [rng.choice("0123456789abcdef") for _ in range(8)]
            # An all-digit token is also valid uppercase hex. Force at least one
            # lowercase letter so this generated example is always a true decoy.
            chars[rng.randrange(len(chars))] = rng.choice("abcdef")
            rid = "req-" + "".join(chars)
        elif variant == 1:  # 7 chars then delimiter
            rid = "req-" + "".join(rng.choice(_HEX) for _ in range(7))
        else:  # wrong prefix
            rid = "reqq-" + "".join(rng.choice(_HEX) for _ in range(8))
        return rng.choice([f"saw token {rid} in legacy header",
                           f"[{rid}] ignored (malformed)"])


class _BracketIp(_Family):
    fid = "bracket_ip"
    spec = ("Extract client IPs. A client IP is an IPv4 dotted quad that appears "
            "inside square brackets, e.g. [10.2.3.4]. Report ONLY the IP itself, "
            "without the brackets.")
    reference = r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]"

    def positive(self, rng):
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        ctx = rng.choice([f"conn from [{ip}] accepted",
                          f"client [{ip}] rate-limited",
                          f"audit [{ip}] path=/login ok"])
        return ctx, ip

    def negative(self, rng, variant):
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        if variant == 0:  # unbracketed IP
            return f"resolved upstream {ip} via dns"
        if variant == 1:  # bracketed non-IP
            return f"queue [{rng.choice(['alpha', 'beta-7', 'x9'])}] drained"
        return f"version v{ip} deployed"  # version string


class _IsoDate(_Family):
    fid = "iso_date"
    spec = ("Extract event dates. Lines contain ISO-8601 UTC timestamps like "
            "2026-03-14T09:22:11Z. Report ONLY the date part (YYYY-MM-DD) of "
            "such timestamps.")
    reference = r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}Z"

    def positive(self, rng):
        d = f"2026-{rng.randint(1, 12):02d}-{rng.randint(10, 28):02d}"
        ts = f"{d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        ctx = rng.choice([f"{ts} job finished", f"snapshot at {ts}",
                          f"{ts} WARN slow flush"])
        return ctx, d

    def negative(self, rng, variant):
        if variant == 0:  # space-separated, no Z
            return (f"2026-{rng.randint(1, 12):02d}-{rng.randint(10, 28):02d} "
                    f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:11 local backup")
        if variant == 1:  # slashes
            return (f"2026/{rng.randint(1, 12):02d}/{rng.randint(10, 28):02d}"
                    f"T08:00:00Z legacy export")
        return f"build 2026{rng.randint(10, 28)} tagged"  # compact build id


class _LatencyMs(_Family):
    fid = "latency_ms"
    spec = ("Extract request latencies. Latency appears as the field "
            "latency=<integer>ms. Report ONLY the integer (no units, no field name).")
    reference = r"latency=(\d+)ms"

    def positive(self, rng):
        n = str(rng.randint(2, 4999))
        ctx = rng.choice([f"GET /api ok latency={n}ms user=alice",
                          f"upstream call latency={n}ms (cache miss)",
                          f"latency={n}ms status=200"])
        return ctx, n

    def negative(self, rng, variant):
        n = rng.randint(2, 4999)
        if variant == 0:
            return f"timeout={n}ms exceeded for shard 3"
        if variant == 1:
            return f"latency={n}s on cold start (seconds!)"
        return f"budget {n}ms allotted to handler"


FAMILIES: list[_Family] = [_ReqId(), _BracketIp(), _IsoDate(), _LatencyMs()]


@register
class RegexExtractEnv(AtlasEnv):
    env_id = "regex_extract"
    name = "Log-field regex authoring"
    description = "Write one regex that extracts a specified field and rejects decoys; scored by F1^2."
    answer_format = "a single regular expression (Python flavor), nothing else"
    difficulty_dials = {"positives": "8 -> 16", "negatives": "6 -> 14",
                        "decoy_families": "1 -> 3"}

    def _build(self, rng, difficulty):
        n_pos, n_neg, n_variants = _PARAMS[difficulty]
        fam = FAMILIES[rng.randrange(len(FAMILIES))]
        positives = [fam.positive(rng) for _ in range(n_pos)]
        negatives = [fam.negative(rng, rng.randrange(n_variants)) for _ in range(n_neg)]
        shown = [l for l, _ in positives] + negatives
        rng.shuffle(shown)
        sample = "\n".join(shown)
        prompt = (
            f"Spec: {fam.spec}\n\n"
            "Write ONE regular expression (Python `re` flavor) for a log parser. "
            "It will be applied to each line with re.search. On lines that contain "
            "the target field it must extract exactly the specified span (use a "
            "capture group if you need to trim surrounding context; group 1 is used "
            "when present, otherwise the whole match). On all other lines — including "
            "near-miss decoys — it must not match at all.\n\n"
            f"Sample lines (mixed):\n```\n{sample}\n```\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"positives": positives, "negatives": negatives, "family": fam.fid,
              "reference": fam.reference}
        meta = {"complexity": n_pos + n_neg + 5 * n_variants}
        return prompt, gt, meta

    def _score_pattern(self, pattern: str, gt: dict) -> tuple[float, list[str]]:
        flags: list[str] = []
        if len(pattern) > 300:
            return 0.0, ["pattern_too_long"]
        try:
            _rx.compile(pattern)
        except Exception:
            return 0.0, ["regex_compile_error"]
        tp = fn = fp = 0
        try:
            for line, gold in gt["positives"]:
                m = _safe_search(pattern, line)
                got = None
                if m:
                    got = m.group(1) if (m.lastindex or 0) >= 1 else m.group(0)
                if got == gold:
                    tp += 1
                else:
                    fn += 1
            for line in gt["negatives"]:
                if _safe_search(pattern, line):
                    fp += 1
        except TimeoutError:
            return 0.0, ["redos_timeout"]
        except Exception:
            return 0.0, ["regex_runtime_error"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return f1, flags

    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, content = extract_answer(response)
        if not content:
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response)
        lines = [l for l in content.splitlines() if l.strip()]
        pattern = lines[-1].strip() if lines else ""
        if (pattern.startswith("/") and pattern.endswith("/")
                and len(pattern) > 2):  # tolerate /slash/ delimiters
            pattern = pattern[1:-1]
        f1, flags = self._score_pattern(pattern, instance.ground_truth)
        corr = f1 * f1
        return self.score(tag_found=tag, parsed_ok=bool(pattern), correctness=corr,
                          success=f1 >= 0.999, response=response,
                          components={"f1": round(f1, 4)}, hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_answer(instance.ground_truth["reference"])

    def extra_canaries(self, instance: Instance):
        gold0 = instance.ground_truth["positives"][0][1]
        import re as _stdre
        return [
            Canary("match_anything", wrap_answer(".*"),
                   "catch-all pattern"),
            Canary("literal_first_gold", wrap_answer(_stdre.escape(gold0)),
                   "hardcodes one observed value"),
            Canary("redos_attempt", wrap_answer(r"(([a-zA-Z0-9]+)+)+$"),
                   "catastrophic-backtracking pattern"),
        ]
