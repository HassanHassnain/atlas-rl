"""shell_golf: write a one-line shell pipeline that solves a filesystem task.

Observation: a virtual filesystem listing (paths + sizes), a reference for the
supported command subset, and a concrete data question.
Action: ONE shell pipeline (single line) using only the supported commands.
Verification is SEMANTIC: the command is executed by a deterministic,
sandboxed mini-interpreter against the same VFS, and its stdout is compared to
the ground-truth output. Any correct pipeline passes, not just the oracle's.
Reward: exact output -> 1.0; else 0.7 * line-multiset F1. Strict: exact.

Anti-hacking: there is no `echo`/`printf` (you cannot print a guessed literal);
`;`, `&&`, redirection and command substitution are rejected; unknown commands
and paths error out (reward 0).

Difficulty dials: filesystem size, required pipeline depth (1 -> 4 stages).
"""

from __future__ import annotations

import re
import shlex
from collections import Counter

from atlas_rl.core.env import AtlasEnv
from atlas_rl.core.protocol import extract_answer, wrap_answer
from atlas_rl.core.registry import register
from atlas_rl.core.types import Canary, Instance, RewardBreakdown


# --------------------------------------------------------------------------- #
# Mini shell interpreter over a virtual filesystem
# --------------------------------------------------------------------------- #
class CommandError(Exception):
    pass


BANNED = set(";&<>`")


class VFS:
    def __init__(self, files: dict[str, list[str]], dirs: list[str]):
        self.files = files
        self.dirs = set(dirs)

    @staticmethod
    def norm(p: str) -> str:
        p = p.strip()
        if p.startswith("./"):
            p = p[2:]
        return p.rstrip("/") or "."

    def is_file(self, p: str) -> bool:
        return self.norm(p) in self.files

    def is_dir(self, p: str) -> bool:
        p = self.norm(p)
        return p == "." or p in self.dirs

    def read(self, p: str) -> list[str]:
        p = self.norm(p)
        if p not in self.files:
            raise CommandError(f"cat: {p}: No such file")
        return list(self.files[p])

    def size(self, p: str) -> int:
        return sum(len(l) + 1 for l in self.files[self.norm(p)])

    def entries(self, d: str) -> list[str]:
        d = self.norm(d)
        if not self.is_dir(d):
            raise CommandError(f"ls: {d}: No such directory")
        prefix = "" if d == "." else d + "/"
        out = set()
        for p in list(self.files) + sorted(self.dirs):
            if p == d:
                continue
            if p.startswith(prefix):
                rest = p[len(prefix):]
                out.add(rest.split("/")[0])
        return sorted(out)

    def walk_files(self, d: str) -> list[str]:
        d = self.norm(d)
        if not self.is_dir(d):
            raise CommandError(f"find: {d}: No such directory")
        prefix = "" if d == "." else d + "/"
        return sorted(p for p in self.files if p.startswith(prefix))

    def walk_dirs(self, d: str) -> list[str]:
        d = self.norm(d)
        prefix = "" if d == "." else d + "/"
        return sorted(p for p in self.dirs if p.startswith(prefix))


def split_pipeline(cmd: str) -> list[str]:
    stages, cur, quote = [], [], None
    for ch in cmd:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == "|":
            stages.append("".join(cur))
            cur = []
        elif ch in BANNED:
            raise CommandError(f"unsupported shell operator: {ch!r}")
        else:
            cur.append(ch)
    if quote:
        raise CommandError("unterminated quote")
    stages.append("".join(cur))
    stages = [s.strip() for s in stages]
    if any(not s for s in stages):
        raise CommandError("empty pipeline stage")
    return stages


def _numkey(line: str):
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", line)
    return float(m.group(1)) if m else 0.0


def _parse_n(args: list[str], default: int = 10) -> tuple[int, list[str]]:
    if args and args[0] == "-n":
        if len(args) < 2 or not args[1].lstrip("-").isdigit():
            raise CommandError("expected number after -n")
        return int(args[1]), args[2:]
    if args and re.fullmatch(r"-\d+", args[0]):
        return int(args[0][1:]), args[1:]
    return default, args


def run_stage(stage: str, stdin: list[str], vfs: VFS) -> list[str]:
    try:
        argv = shlex.split(stage, posix=True)
    except ValueError as e:
        raise CommandError(f"parse error: {e}") from None
    if not argv:
        raise CommandError("empty command")
    cmd, args = argv[0], argv[1:]

    if cmd == "cat":
        if not args:
            return stdin
        out = []
        for f in args:
            out.extend(vfs.read(f))
        return out

    if cmd == "grep":
        invert = ignore = count = False
        while args and args[0] in ("-v", "-i", "-c", "-vi", "-iv", "-ci", "-ic", "-cv", "-vc"):
            for ch in args[0][1:]:
                invert |= ch == "v"
                ignore |= ch == "i"
                count |= ch == "c"
            args = args[1:]
        if not args:
            raise CommandError("grep: missing pattern")
        pat, files = args[0], args[1:]
        try:
            rx = re.compile(pat, re.IGNORECASE if ignore else 0)
        except re.error as e:
            raise CommandError(f"grep: bad pattern: {e}") from None
        lines = stdin if not files else [l for f in files for l in vfs.read(f)]
        hits = [l for l in lines if bool(rx.search(l)) != invert]
        return [str(len(hits))] if count else hits

    if cmd == "wc":
        if args[:1] != ["-l"]:
            raise CommandError("wc: only -l supported")
        files = args[1:]
        lines = stdin if not files else [l for f in files for l in vfs.read(f)]
        return [str(len(lines))]

    if cmd == "sort":
        numeric = rev = uniq = False
        rest = []
        for a in args:
            if a.startswith("-") and set(a[1:]) <= set("nru") and len(a) > 1:
                numeric |= "n" in a
                rev |= "r" in a
                uniq |= "u" in a
            else:
                rest.append(a)
        lines = stdin if not rest else [l for f in rest for l in vfs.read(f)]
        out = sorted(lines, key=_numkey if numeric else None, reverse=rev)
        if numeric:  # stable secondary order for equal keys
            out = sorted(lines, key=lambda l: (_numkey(l), l), reverse=rev)
        if uniq:
            seen, dedup = set(), []
            for l in out:
                if l not in seen:
                    seen.add(l)
                    dedup.append(l)
            out = dedup
        return out

    if cmd == "uniq":
        count = args[:1] == ["-c"]
        files = args[1:] if count else args
        lines = stdin if not files else [l for f in files for l in vfs.read(f)]
        out, prev, n = [], None, 0
        for l in lines + [None]:
            if l == prev:
                n += 1
            else:
                if prev is not None:
                    out.append(f"{n} {prev}" if count else prev)
                prev, n = l, 1
        return out

    if cmd in ("head", "tail"):
        n, rest = _parse_n(args)
        lines = stdin if not rest else [l for f in rest for l in vfs.read(f)]
        return lines[:n] if cmd == "head" else lines[-n:] if n else []

    if cmd == "cut":
        delim, field = None, None
        while args:
            if args[0] == "-d":
                delim, args = args[1], args[2:]
            elif args[0].startswith("-d") and len(args[0]) > 2:
                delim, args = args[0][2:], args[1:]
            elif args[0] == "-f":
                field, args = args[1], args[2:]
            elif args[0].startswith("-f") and len(args[0]) > 2:
                field, args = args[0][2:], args[1:]
            else:
                break
        if delim is None or field is None or not field.isdigit():
            raise CommandError("cut: need -d <delim> -f <field-number>")
        i = int(field)
        lines = stdin if not args else [l for f in args for l in vfs.read(f)]
        out = []
        for l in lines:
            if delim not in l:
                out.append(l)  # matches coreutils cut behaviour
            else:
                parts = l.split(delim)
                out.append(parts[i - 1] if 1 <= i <= len(parts) else "")
        return out

    if cmd == "ls":
        d = args[0] if args else "."
        return vfs.entries(d)

    if cmd == "find":
        d = args[0] if args and not args[0].startswith("-") else "."
        rest = args[1:] if args and not args[0].startswith("-") else args
        ftype, name, size = None, None, None
        while rest:
            if rest[0] == "-type":
                ftype, rest = rest[1], rest[2:]
            elif rest[0] == "-name":
                name, rest = rest[1], rest[2:]
            elif rest[0] == "-size":
                size, rest = rest[1], rest[2:]
            else:
                raise CommandError(f"find: unsupported predicate {rest[0]}")
        cands: list[str] = []
        if ftype in (None, "f"):
            cands += vfs.walk_files(d)
        if ftype in (None, "d"):
            cands += vfs.walk_dirs(d)
        out = []
        for p in sorted(cands):
            base = p.split("/")[-1]
            if name:
                import fnmatch
                if not fnmatch.fnmatch(base, name):
                    continue
            if size:
                m = re.fullmatch(r"([+-])(\d+)c?", size)
                if not m or p not in vfs.files:
                    if not m:
                        raise CommandError("find: bad -size")
                    continue
                sz, thr = vfs.size(p), int(m.group(2))
                if m.group(1) == "+" and not sz > thr:
                    continue
                if m.group(1) == "-" and not sz < thr:
                    continue
            out.append(p)
        return out

    raise CommandError(f"{cmd}: command not found")


def run_pipeline(cmd: str, vfs: VFS) -> list[str]:
    if len(cmd) > 500:
        raise CommandError("command too long")
    stages = split_pipeline(cmd)
    if len(stages) > 8:
        raise CommandError("pipeline too deep")
    data: list[str] = []
    for st in stages:
        data = run_stage(st, data, vfs)
    return data


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
_USERS = ["alice", "bob", "carol", "dave", "erin", "frank"]
_REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]
_STATUS = [200, 201, 301, 403, 404, 500, 502]
_PARAMS = {1: (2, 2, 14), 2: (3, 3, 22), 3: (3, 4, 30), 4: (4, 4, 38), 5: (5, 5, 46)}
_TASKS_BY_D = {1: ["count_match", "count_entries"],
               2: ["count_match", "first_k_not", "field_unique", "count_entries"],
               3: ["first_k_not", "field_unique", "big_files"],
               4: ["top_k_field", "big_files", "multi_count"],
               5: ["top_k_field", "multi_count", "big_files"]}

COMMAND_DOC = """Supported commands (this exact subset, nothing else):
- cat FILE...                     concatenate files
- grep [-v|-i|-c] PATTERN [FILE]  regex filter (-v invert, -i ignore case, -c print match count)
- wc -l [FILE]                    line count (prints just the number)
- sort [-n|-r|-u]                 sort lines (-n numeric by leading number, -r reverse, -u unique)
- uniq [-c]                       collapse consecutive duplicates (-c prefixes "<count> <line>")
- head|tail -n N                  first/last N lines
- cut -d DELIM -f N               extract field N (1-indexed)
- ls [DIR]                        list directory entries (sorted)
- find DIR [-type f|d] [-name GLOB] [-size +Nc|-Nc]   find paths (sorted)
Pipes `|` are supported. Redirection, `;`, `&&`, `$()`, backticks and any other
command (echo, awk, sed, xargs, ...) are NOT available."""


@register
class ShellGolfEnv(AtlasEnv):
    env_id = "shell_golf"
    name = "Shell pipeline synthesis"
    description = "Write a one-line pipeline over a virtual filesystem; verified by sandboxed execution."
    answer_format = "a single shell pipeline on one line (no commentary inside the tags)"
    difficulty_dials = {"fs_dirs": "2 -> 5", "files_per_dir": "2 -> 5",
                        "lines_per_file": "14 -> 46", "pipeline_depth": "1 -> 4 stages"}

    # ------------------------------------------------------------- generation
    def _gen_fs(self, rng, difficulty) -> VFS:
        n_dirs, files_per, lines_per = _PARAMS[difficulty]
        dirnames = rng.sample(["logs", "data", "etc", "archive", "tmp"], n_dirs)
        dirs, files = [], {}
        for d in dirnames:
            dirs.append(d)
            for _ in range(rng.randint(max(1, files_per - 1), files_per)):
                kind = rng.choice(["log", "log", "csv", "txt"])
                fname = {
                    "log": f"{rng.choice(['app', 'access', 'worker', 'sync'])}{rng.randint(1, 9)}.log",
                    "csv": f"{rng.choice(['users', 'orders', 'metrics'])}{rng.randint(1, 9)}.csv",
                    "txt": f"{rng.choice(['notes', 'readme', 'todo'])}{rng.randint(1, 9)}.txt",
                }[kind]
                path = f"{d}/{fname}"
                if path in files:
                    continue
                n = rng.randint(max(6, lines_per - 8), lines_per)
                if kind == "log":
                    lines = [
                        f"2026-03-{rng.randint(10, 28)}T0{rng.randint(0, 9)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}Z "
                        f"{rng.choice(_LEVELS)} request path=/api/{rng.choice(['users', 'orders', 'items'])} "
                        f"user={rng.choice(_USERS)} status={rng.choice(_STATUS)} latency={rng.randint(2, 950)}ms"
                        for _ in range(n)]
                elif kind == "csv":
                    lines = [f"{rng.choice(_USERS)},{rng.choice(_REGIONS)},"
                             f"{rng.choice(_STATUS)},{rng.randint(2, 900)}" for _ in range(n)]
                else:
                    lines = [" ".join(rng.choice(["deploy", "rotate", "check", "backlog",
                                                  "ticket", "ack", "todo"])
                                      for _ in range(rng.randint(3, 7))) for _ in range(n)]
                files[path] = lines
        return VFS(files, dirs)

    def _make_task(self, rng, vfs: VFS, difficulty) -> tuple[str, str]:
        """Returns (task_text, oracle_cmd). Retries until the task is well-posed."""
        kind = rng.choice(_TASKS_BY_D[difficulty])
        logs = [p for p in vfs.files if p.endswith(".log")]
        csvs = [p for p in vfs.files if p.endswith(".csv")]
        for _ in range(60):
            if kind == "count_match" and logs:
                f = rng.choice(logs)
                tok = rng.choice([f"user={rng.choice(_USERS)}", "ERROR", "WARN",
                                  f"status={rng.choice(_STATUS)}"])
                if not any(tok in l for l in vfs.files[f]):
                    continue
                return (f"How many lines in `{f}` contain the exact text `{tok}`?",
                        f"grep -c {tok} {f}")
            if kind == "first_k_not" and logs:
                f = rng.choice(logs)
                tok = rng.choice(["INFO", "DEBUG", f"user={rng.choice(_USERS)}"])
                k = rng.randint(2, 5)
                kept = [l for l in vfs.files[f] if tok not in l]
                if len(kept) < k:
                    continue
                return (f"Print the first {k} lines of `{f}` that do NOT contain `{tok}`.",
                        f"grep -v {tok} {f} | head -n {k}")
            if kind == "field_unique" and csvs:
                f = rng.choice(csvs)
                col = rng.choice([1, 2])
                return (f"Print the sorted unique values of comma-separated field {col} "
                        f"of `{f}` (one per line).",
                        f"cut -d , -f {col} {f} | sort -u")
            if kind == "top_k_field" and csvs:
                f = rng.choice(csvs)
                # rebuild column 2 with distinct frequencies so top-k is unambiguous
                vals = rng.sample(_REGIONS, 3)
                freqs = rng.sample(range(3, 12), 3)
                col2 = [v for v, fq in zip(vals, freqs) for _ in range(fq)]
                rng.shuffle(col2)
                rows = vfs.files[f]
                new_rows = []
                for i, v in enumerate(col2):
                    src = rows[i % len(rows)].split(",")
                    new_rows.append(f"{src[0]},{v},{src[2]},{src[3]}")
                vfs.files[f] = new_rows
                k = rng.randint(1, 2)
                return (f"Print the {k} most frequent value(s) of comma-separated field 2 "
                        f"of `{f}`, one per line formatted as `<count> <value>`, most "
                        f"frequent first.",
                        f"cut -d , -f 2 {f} | sort | uniq -c | sort -n -r | head -n {k}")
            if kind == "count_entries":
                d = rng.choice(sorted(vfs.dirs))
                return (f"How many entries are directly inside the directory `{d}`?",
                        f"ls {d} | wc -l")
            if kind == "big_files":
                d = rng.choice(sorted(vfs.dirs))
                sizes = sorted(vfs.size(p) for p in vfs.walk_files(d))
                if len(sizes) < 2:
                    continue
                thr = (sizes[len(sizes) // 2 - 1] + sizes[len(sizes) // 2]) // 2
                if all(s == sizes[0] for s in sizes):
                    continue
                return (f"List the paths of all files under `{d}` (recursively) that are "
                        f"larger than {thr} bytes, sorted.",
                        f"find {d} -type f -size +{thr}c")
            if kind == "multi_count" and len(logs) >= 2:
                f1, f2 = rng.sample(logs, 2)
                tok = rng.choice(["ERROR", "status=500", "status=404"])
                if not any(tok in l for f in (f1, f2) for l in vfs.files[f]):
                    continue
                return (f"Across `{f1}` and `{f2}` combined, how many lines contain `{tok}`?",
                        f"cat {f1} {f2} | grep -c {tok}")
            kind = rng.choice(_TASKS_BY_D[difficulty])  # re-roll task type
        # safe fallback, always well-posed
        d = sorted(vfs.dirs)[0]
        return (f"How many entries are directly inside the directory `{d}`?",
                f"ls {d} | wc -l")

    def _build(self, rng, difficulty):
        vfs = self._gen_fs(rng, difficulty)
        task, oracle_cmd = self._make_task(rng, vfs, difficulty)
        expected = run_pipeline(oracle_cmd, vfs)
        listing = "\n".join(f"{vfs.size(p):>6}  {p}" for p in sorted(vfs.files))
        prompt = (
            "You are at the root of this directory tree (sizes in bytes):\n"
            f"```\n{listing}\n```\n\n"
            f"{COMMAND_DOC}\n\n"
            f"Task: {task}\n\n"
            "Reply with ONE pipeline that prints exactly the requested output.\n"
            f"Answer format: {self.answer_format}"
        )
        gt = {"files": vfs.files, "dirs": sorted(vfs.dirs),
              "expected_output": expected, "oracle_cmd": oracle_cmd, "task": task}
        meta = {"complexity": len(vfs.files) * 10 + sum(len(v) for v in vfs.files.values()) // 10}
        return prompt, gt, meta

    # ----------------------------------------------------------- verification
    def verify(self, instance: Instance, response: str) -> RewardBreakdown:
        tag, content = extract_answer(response)
        gt = instance.ground_truth
        flags: list[str] = []
        if not content:
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response)
        cmd = [l for l in content.splitlines() if l.strip()]
        cmd = cmd[-1].strip() if cmd else ""
        cmd = cmd.lstrip("$ ").strip()
        vfs = VFS(gt["files"], gt["dirs"])
        try:
            out = run_pipeline(cmd, vfs)
            parsed_ok = True
        except CommandError as e:
            flags.append(f"exec_error:{str(e)[:40]}")
            return self.score(tag_found=tag, parsed_ok=False, correctness=0.0,
                              success=False, response=response, hack_flags=flags)
        expected = gt["expected_output"]
        if out == expected:
            corr, success = 1.0, True
        else:
            ca, cb = Counter(out), Counter(expected)
            inter = sum((ca & cb).values())
            prec = inter / max(1, sum(ca.values()))
            rec = inter / max(1, sum(cb.values()))
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            corr, success = 0.7 * f1, False
        return self.score(tag_found=tag, parsed_ok=parsed_ok, correctness=corr,
                          success=success, response=response,
                          components={"exact": float(success)}, hack_flags=flags)

    def oracle(self, instance: Instance) -> str:
        return wrap_answer(instance.ground_truth["oracle_cmd"])

    def extra_canaries(self, instance: Instance):
        exp = instance.ground_truth["expected_output"]
        literal = exp[0] if exp else "42"
        return [
            Canary("echo_literal", wrap_answer(f"echo {literal}"),
                   "tries to print the answer literally (echo is not available)"),
            Canary("outside_vfs", wrap_answer("cat /etc/passwd"),
                   "reads outside the task filesystem"),
            Canary("shell_injection", wrap_answer("ls . ; cat /etc/shadow && rm -rf /"),
                   "uses banned shell operators"),
            Canary("dump_everything", wrap_answer("cat " + " ".join(sorted(instance.ground_truth["files"])[:3])),
                   "dumps file contents hoping for overlap"),
        ]
