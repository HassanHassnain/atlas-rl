"""semver_resolve: resolve a dependency graph to exact versions.

Observation: a package universe (available versions per package), the root
project's requirements (semver constraints), and each package's own
dependencies. Resolution rule: include exactly the transitively required
packages, and pick for each the HIGHEST version satisfying ALL constraints
that apply to it.
Action: JSON object {package: "x.y.z", ...} for exactly the required set.
Reward: (constraints_satisfied_frac)^2 * (0.6 + 0.2*completeness(Jaccard)
+ 0.2*maximality). Squared gating denies credit to constraint-violating
shotgun answers (e.g. "pick every latest version").
Strict success: assignment equals the unique correct resolution.

Difficulty dials: package count, dependency-edge count, constraint-operator mix.
"""

from __future__ import annotations

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import parse_json_answer, wrap_json_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown

_NAMES = ["httpcore", "logfmt", "yamlx", "authkit", "dbdriver", "metricsd",
          "queuelib", "tlswrap", "cachev", "jsonfast"]
_PARAMS = {1: (2, 0), 2: (3, 1), 3: (3, 2), 4: (4, 2), 5: (4, 3)}  # n_root, n_dep_edges

V = tuple[int, int, int]


def _vs(v: V) -> str:
    return ".".join(map(str, v))


def _parse_v(s: str) -> V | None:
    parts = str(s).strip().lstrip("v").split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def satisfies(v: V, constraint: str) -> bool:
    """Supports ^x.y.z, ~x.y.z, =x.y.z, and space-separated comparator lists."""
    c = constraint.strip()
    if c.startswith("^"):
        base = _parse_v(c[1:])
        if base is None:
            return False
        if base[0] == 0:
            return base <= v < (0, base[1] + 1, 0)
        return base <= v < (base[0] + 1, 0, 0)
    if c.startswith("~"):
        base = _parse_v(c[1:])
        return base is not None and base <= v < (base[0], base[1] + 1, 0)
    ok = True
    for part in c.split():
        if part.startswith(">="):
            b = _parse_v(part[2:])
            ok &= b is not None and v >= b
        elif part.startswith("<="):
            b = _parse_v(part[2:])
            ok &= b is not None and v <= b
        elif part.startswith(">"):
            b = _parse_v(part[1:])
            ok &= b is not None and v > b
        elif part.startswith("<"):
            b = _parse_v(part[1:])
            ok &= b is not None and v < b
        elif part.startswith("="):
            b = _parse_v(part[1:])
            ok &= b is not None and v == b
        else:
            b = _parse_v(part)
            ok &= b is not None and v == b
    return ok


@register
class SemverResolveEnv(AtlasEnv):
    env_id = "semver_resolve"
    name = "Dependency version resolution"
    description = "Resolve transitive requirements to the highest versions satisfying all semver constraints."
    answer_format = ('JSON object mapping every required package to its resolved '
                     'version, e.g. {"httpcore": "2.3.1", "logfmt": "1.4.0"}')
    difficulty_dials = {"root_requirements": "2 -> 4", "dependency_edges": "0 -> 3",
                        "operators": "exact/caret -> mixed"}

    # ------------------------------------------------------------- generation
    def _gen_versions(self, rng) -> list[V]:
        """>=2 majors; the lower target-major has >=2 versions."""
        majors = sorted(rng.sample(range(0, 4), 2))
        vers: set[V] = set()
        lo, hi = majors
        for _ in range(rng.randint(2, 3)):
            vers.add((lo, rng.randint(0, 6), rng.randint(0, 9)))
        # the resolution target is max(lower major); it must never equal the
        # package's global min, or "pick every oldest version" can satisfy an
        # exact/lower-bound constraint (audit leak found by the canary suite)
        while sum(1 for v in vers if v[0] == lo) < 2:
            vers.add((lo, rng.randint(0, 6), rng.randint(0, 9)))
        for _ in range(rng.randint(1, 3)):
            vers.add((hi, rng.randint(0, 6), rng.randint(0, 9)))
        while sum(1 for v in vers if v[0] == hi) < 1:
            vers.add((hi, rng.randint(0, 6), rng.randint(0, 9)))
        return sorted(vers)

    def _constraint_for(self, rng, versions: list[V], target: V) -> str:
        """Build a constraint whose max-satisfying version (in `versions`) is `target`."""
        forms = ["caret", "tilde", "exact", "range"]
        if target[0] == 0:
            forms.remove("caret")
        form = rng.choice(forms)
        if form == "caret":
            same_major_max = max(v for v in versions if v[0] == target[0])
            if same_major_max != target:
                form = "tilde"
            else:
                return f"^{_vs(target)}"
        if form == "tilde":
            same_minor_max = max((v for v in versions
                                  if v[:2] == target[:2]), default=None)
            if same_minor_max != target:
                form = "range"
            else:
                return f"~{_vs(target)}"
        if form == "exact":
            return f"={_vs(target)}"
        higher = sorted(v for v in versions if v > target)
        upper = higher[0] if higher else (target[0] + 1, 0, 0)
        return f">={_vs(target)} <{_vs(upper)}"

    def _build(self, rng, difficulty):
        n_root, n_edges = _PARAMS[difficulty]
        n_pkgs = min(len(_NAMES), n_root + n_edges + rng.randint(1, 2))
        names = rng.sample(_NAMES, n_pkgs)
        universe = {p: self._gen_versions(rng) for p in names}

        def pick_target(p: str) -> V:
            vers = universe[p]
            lo_major = vers[0][0]
            lo_versions = [v for v in vers if v[0] == lo_major]
            # max of the lower major: never the global max, never the global min
            # (lower major has >=2 versions by construction)
            return max(lo_versions)

        root_pkgs = names[:n_root]
        resolution: dict[str, V] = {}
        constraints: list[tuple[str, str, str]] = []  # (source, pkg, constraint)
        for p in root_pkgs:
            resolution[p] = pick_target(p)
            constraints.append(("<root>", p, self._constraint_for(rng, universe[p], resolution[p])))
        # dependency edges: from an already-required pkg to a new pkg
        deps: dict[str, list[tuple[str, str]]] = {p: [] for p in names}
        pool = [p for p in names if p not in resolution]
        for _ in range(n_edges):
            if not pool:
                break
            src = rng.choice(sorted(resolution))
            dst = pool.pop(rng.randrange(len(pool)))
            resolution[dst] = pick_target(dst)
            cstr = self._constraint_for(rng, universe[dst], resolution[dst])
            deps[src].append((dst, cstr))
            constraints.append((src, dst, cstr))

        uni_doc = "\n".join(
            f"- {p}: available versions {', '.join(_vs(v) for v in universe[p])}"
            for p in sorted(universe))
        root_doc = "\n".join(f"- requires {p} {c}" for s, p, c in constraints
                             if s == "<root>")
        dep_lines = []
        for p in sorted(names):
            if deps[p]:
                dep_lines.append(f"- {p} (any version) depends on: "
                                 + ", ".join(f"{d} {c}" for d, c in deps[p]))
            elif p in resolution:
                dep_lines.append(f"- {p} has no dependencies")
        prompt = (
            "Resolve this project's dependencies.\n\n"
            f"Package universe:\n{uni_doc}\n\n"
            f"Root project requirements:\n{root_doc}\n\n"
            f"Package dependencies:\n" + "\n".join(dep_lines) + "\n\n"
            "Semver semantics: ^a.b.c allows >=a.b.c and <(a+1).0.0 (for a=0: "
            "<0.(b+1).0); ~a.b.c allows >=a.b.c and <a.(b+1).0; =a.b.c is exact; "
            "comparator lists like \">=1.2.0 <2.0.0\" are ANDed.\n"
            "Rules: include EXACTLY the packages that are transitively required "
            "(root requirements plus their dependencies, recursively) — no extras. "
            "For each, choose the HIGHEST available version satisfying every "
            "constraint that applies to it.\n\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {
            "universe": {p: [list(v) for v in vs] for p, vs in universe.items()},
            "resolution": {p: _vs(v) for p, v in resolution.items()},
            "constraints": [list(c) for c in constraints],
        }
        meta = {"complexity": n_pkgs * 5 + len(constraints) * 8}
        return prompt, gt, meta

    # ----------------------------------------------------------- verification
    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, ok, obj = parse_json_answer(response)
        gt = instance.ground_truth
        if not (ok and isinstance(obj, dict)):
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response)
        universe = {p: [tuple(v) for v in vs] for p, vs in gt["universe"].items()}
        closure = set(gt["resolution"])
        chosen: dict[str, V] = {}
        flags: list[str] = []
        for p, vstr in obj.items():
            v = _parse_v(vstr) if isinstance(vstr, str) else None
            if p in universe and v is not None and v in universe[p]:
                chosen[p] = v
            else:
                flags.append("nonexistent_version_or_pkg")
        sat = total = 0
        for _, pkg, cstr in gt["constraints"]:
            total += 1
            if pkg in chosen and satisfies(chosen[pkg], cstr):
                sat += 1
        cfrac = sat / max(1, total)
        keys = set(obj)
        completeness = len(keys & closure) / max(1, len(keys | closure))
        maximality = sum(1 for p in closure
                         if p in chosen and _vs(chosen[p]) == gt["resolution"][p]) / len(closure)
        corr = (cfrac ** 2) * (0.6 + 0.2 * completeness + 0.2 * maximality)
        success = ({p: _vs(v) for p, v in chosen.items()} == gt["resolution"]
                   and keys == closure)
        return self.score(tag_found=tag, parsed_ok=True, correctness=corr,
                          success=success, response=response,
                          components={"constraints": round(cfrac, 4),
                                      "completeness": round(completeness, 4),
                                      "maximality": round(maximality, 4)},
                          hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_json_answer(instance.ground_truth["resolution"])

    def extra_canaries(self, instance: Instance):
        gt = instance.ground_truth
        latest = {p: _vs(max(tuple(v) for v in gt["universe"][p]))
                  for p in gt["resolution"]}
        lowest = {p: _vs(min(tuple(v) for v in gt["universe"][p]))
                  for p in gt["resolution"]}
        return [
            Canary("all_latest", wrap_json_answer(latest),
                   "ignores constraints, picks every newest version"),
            Canary("all_lowest", wrap_json_answer(lowest),
                   "picks every oldest version"),
        ]
