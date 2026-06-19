"""cron_author: translate a natural-language schedule into a cron expression.

Observation: an operations request describing a schedule in prose.
Action: a standard 5-field cron expression.
Verification is SEMANTIC: the candidate expression's fire times over a fixed
4-week simulation window are compared to the ground-truth schedule's fire
times, so any equivalent expression (e.g. `0,30 * ...` vs `*/30 * ...`) passes.
Reward: exact fire-set match -> 1.0, else 0.7 * squared Jaccard overlap.
Strict success: exact fire-set match.

Difficulty dials: number of constrained fields, steps/ranges/lists, 12h vs 24h
phrasing.
"""

from __future__ import annotations

import datetime as dt

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import extract_answer, wrap_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

WINDOW_START = dt.datetime(2026, 1, 5, 0, 0)  # a Monday
WINDOW_MINUTES = 28 * 24 * 60
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


class CronError(Exception):
    pass


def _parse_field(text: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for term in text.split(","):
        term = term.strip()
        if not term:
            raise CronError("empty term")
        step = 1
        if "/" in term:
            term, s = term.split("/", 1)
            if not s.isdigit() or int(s) < 1:
                raise CronError(f"bad step {s!r}")
            step = int(s)
        if term == "*":
            a, b = lo, hi
        elif "-" in term:
            a_s, b_s = term.split("-", 1)
            if not (a_s.isdigit() and b_s.isdigit()):
                raise CronError(f"bad range {term!r}")
            a, b = int(a_s), int(b_s)
        elif term.isdigit():
            a = b = int(term)
        else:
            raise CronError(f"bad term {term!r}")
        if not (lo <= a <= hi and lo <= b <= hi and a <= b):
            raise CronError(f"out of bounds: {term!r}")
        out.update(range(a, b + 1, step))
    return out


def parse_cron(expr: str) -> dict:
    parts = expr.split()
    if len(parts) != 5:
        raise CronError(f"expected 5 fields, got {len(parts)}")
    m, h, dom, mon, dow = parts
    dow_set = {d % 7 for d in _parse_field(dow, 0, 7)}  # 7 == Sunday == 0
    return {
        "minute": _parse_field(m, 0, 59),
        "hour": _parse_field(h, 0, 23),
        "dom": _parse_field(dom, 1, 31),
        "month": _parse_field(mon, 1, 12),
        "dow": dow_set,
        "dom_star": dom.strip() == "*",
        "dow_star": dow.strip() == "*",
    }


def _calendar() -> list[tuple[int, int, int, int, int]]:
    """(minute, hour, dom, month, cron_dow) for every minute of the window."""
    out = []
    t = WINDOW_START
    for _ in range(WINDOW_MINUTES):
        out.append((t.minute, t.hour, t.day, t.month, (t.weekday() + 1) % 7))
        t += dt.timedelta(minutes=1)
    return out


_CAL = _calendar()


def fire_set(expr: str) -> set[int]:
    """Minutes (offset from WINDOW_START) at which the expression fires."""
    spec = parse_cron(expr)
    mset, hset, domset = spec["minute"], spec["hour"], spec["dom"]
    monset, dowset = spec["month"], spec["dow"]
    dom_star, dow_star = spec["dom_star"], spec["dow_star"]
    fires: set[int] = set()
    for off, (mi, hr, dy, mo, dw) in enumerate(_CAL):
        if mi not in mset or hr not in hset or mo not in monset:
            continue
        if dom_star and dow_star:
            ok = True
        elif dom_star:
            ok = dw in dowset
        elif dow_star:
            ok = dy in domset
        else:
            ok = dy in domset or dw in dowset  # standard cron OR rule
        if ok:
            fires.add(off)
    return fires


def _say_time(rng, h: int, m: int) -> str:
    if rng.random() < 0.5:
        return f"{h:02d}:{m:02d}"
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


@register
class CronAuthorEnv(AtlasEnv):
    env_id = "cron_author"
    name = "Cron schedule authoring"
    description = "Write a 5-field cron expression matching a natural-language schedule; checked by fire-time simulation."
    answer_format = "a single standard 5-field cron expression, e.g. `*/10 8-17 * * 1-5`"
    difficulty_dials = {"constrained_fields": "1 -> 4", "steps_ranges_lists": "none -> combined",
                        "phrasing": "24h -> mixed 12h/12h+lists"}

    def _make_spec(self, rng, difficulty) -> tuple[str, str]:
        """Returns (nl_description, cron_expr)."""
        m = rng.randint(0, 59)
        h = rng.randint(0, 23)
        if difficulty == 1:
            kind = rng.choice(["every_n", "hourly_at", "daily_at"])
            if kind == "every_n":
                n = rng.choice([5, 10, 15, 20, 30])
                return f"Run the job every {n} minutes.", f"*/{n} * * * *"
            if kind == "hourly_at":
                return (f"Run the job once an hour, at minute {m} past the hour.",
                        f"{m} * * * *")
            return (f"Run the job once a day at {_say_time(rng, h, m)}.",
                    f"{m} {h} * * *")
        if difficulty == 2:
            kind = rng.choice(["weekdays_at", "monthly", "daily_at"])
            if kind == "weekdays_at":
                return (f"Run the backup at {_say_time(rng, h, m)} on weekdays "
                        f"(Monday through Friday).", f"{m} {h} * * 1-5")
            if kind == "monthly":
                d = rng.randint(5, 28)
                return (f"Run the report on day {d} of every month at "
                        f"{_say_time(rng, h, m)}.", f"{m} {h} {d} * *")
            return (f"Run the cleanup daily at {_say_time(rng, h, m)}.",
                    f"{m} {h} * * *")
        if difficulty == 3:
            kind = rng.choice(["specific_days", "twice_daily", "between_hours"])
            if kind == "specific_days":
                days = sorted(rng.sample(range(1, 7), rng.choice([2, 3])))
                names = ", ".join(DAY_NAMES[d] for d in days)
                return (f"Run the sync at {_say_time(rng, h, m)} on {names}.",
                        f"{m} {h} * * {','.join(map(str, days))}")
            if kind == "twice_daily":
                h2 = (h + rng.randint(6, 12)) % 24
                lo, hi = sorted((h, h2))
                if lo == hi:
                    hi = (hi + 7) % 24
                    lo, hi = sorted((lo, hi))
                return (f"Run the healthcheck twice a day, at {_say_time(rng, lo, m)} "
                        f"and at {_say_time(rng, hi, m)} (same minute).",
                        f"{m} {lo},{hi} * * *")
            n = rng.choice([10, 15, 20, 30])
            h1 = rng.randint(6, 11)
            h2 = rng.randint(13, 21)
            return (f"Run the poller every {n} minutes from {h1:02d}:00 through "
                    f"{h2:02d}:59, every day.", f"*/{n} {h1}-{h2} * * *")
        if difficulty == 4:
            n = rng.choice([10, 15, 20, 30])
            h1 = rng.randint(6, 11)
            h2 = rng.randint(13, 21)
            return (f"Run the exporter every {n} minutes from {h1:02d}:00 through "
                    f"{h2:02d}:59 on weekdays (Monday to Friday) only.",
                    f"*/{n} {h1}-{h2} * * 1-5")
        # difficulty 5
        kind = rng.choice(["quarter_list_days", "step_on_days"])
        if kind == "quarter_list_days":
            mins = sorted(rng.sample([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                                     rng.choice([2, 3])))
            days = sorted(rng.sample(range(1, 7), 2))
            h1 = rng.randint(7, 10)
            h2 = rng.randint(16, 20)
            names = " and ".join(DAY_NAMES[d] for d in days)
            mins_txt = ", ".join(f"minute {x}" for x in mins)
            return (f"During business hours ({h1:02d}:00 through {h2:02d}:59), run the "
                    f"job at {mins_txt} past each hour, but only on {names}.",
                    f"{','.join(map(str, mins))} {h1}-{h2} * * {','.join(map(str, days))}")
        n = rng.choice([15, 20, 30])
        days = sorted(rng.sample(range(1, 7), 3))
        names = ", ".join(DAY_NAMES[d] for d in days)
        return (f"Run the audit every {n} minutes, all day, on {names}.",
                f"*/{n} * * * {','.join(map(str, days))}")

    def _build(self, rng, difficulty):
        nl, expr = self._make_spec(rng, difficulty)
        prompt = (
            "Translate this operations request into a standard 5-field cron "
            "expression (minute hour day-of-month month day-of-week; "
            "day-of-week 0=Sunday..6=Saturday):\n\n"
            f"Request: {nl}\n\n"
            "Notes: any expression that fires at exactly the requested times is "
            "accepted (equivalent forms are fine).\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"cron": expr, "nl": nl}
        meta = {"complexity": difficulty * 10 + len(expr)}
        return prompt, gt, meta

    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, content = extract_answer(response)
        flags: list[str] = []
        content = content.strip().strip("`")
        if not content:
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response)
        cand_line = [l for l in content.splitlines() if l.strip()]
        cand = cand_line[-1].strip() if cand_line else ""
        try:
            cand_fires = fire_set(cand)
            parsed_ok = True
        except CronError as e:
            flags.append(f"cron_parse:{str(e)[:30]}")
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response, hack_flags=flags)
        gt_fires = fire_set(instance.ground_truth["cron"])
        if cand_fires == gt_fires:
            corr, success = 1.0, True
        else:
            union = len(cand_fires | gt_fires)
            jac = len(cand_fires & gt_fires) / union if union else 0.0
            # Squaring keeps useful partial credit while denying excessive
            # reward to broad or field-swapped schedules.
            corr, success = 0.7 * jac * jac, False
        return self.score(tag_found=tag, parsed_ok=parsed_ok, correctness=corr,
                          success=success, response=response,
                          components={"jaccard": round(jac, 4) if not success else 1.0},
                          hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_answer(instance.ground_truth["cron"])

    def extra_canaries(self, instance: Instance):
        gt = instance.ground_truth["cron"].split()
        swapped = " ".join([gt[1], gt[0]] + gt[2:])  # classic hour/minute swap
        gt_fires = fire_set(instance.ground_truth["cron"])
        candidates = [
            ("fire_always", "* * * * *",
             "fires every minute hoping to cover the schedule"),
            ("swapped_fields", swapped, "minute and hour fields swapped"),
            ("midnight_default", "0 0 * * *", "generic daily-at-midnight guess"),
        ]
        canaries = []
        for name, expression, description in candidates:
            try:
                is_exploit = fire_set(expression) != gt_fires
            except CronError:
                is_exploit = True
            if is_exploit:
                canaries.append(Canary(name, wrap_answer(expression), description))
        return canaries
