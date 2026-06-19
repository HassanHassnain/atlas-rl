"""Unit tests for the heavier verifier machinery."""

import pytest

from atlas_rl.core.protocol import extract_answer, parse_json_answer
from atlas_rl.envs.cron_author import CronError, fire_set, parse_cron
from atlas_rl.envs.semver_resolve import _parse_v, satisfies
from atlas_rl.envs.shell_golf import VFS, CommandError, run_pipeline
from atlas_rl.inference.backends import make_backend


def test_hf_backend_spec_options(monkeypatch):
    captured = {}

    class FakeHF:
        def __init__(self, model_id, adapter=None, enable_thinking=None):
            captured.update(model_id=model_id, adapter=adapter,
                            enable_thinking=enable_thinking)

    monkeypatch.setattr("atlas_rl.inference.backends.HFBackend", FakeHF)
    make_backend("hf:Qwen/example:adapter=path/to/adapter:thinking=false")
    assert captured == {
        "model_id": "Qwen/example",
        "adapter": "path/to/adapter",
        "enable_thinking": False,
    }


# ------------------------------------------------------------------ protocol
def test_extract_answer_basic():
    assert extract_answer("blah <answer>42</answer>") == (True, "42")
    assert extract_answer("<answer>a</answer> then <answer>b</answer>")[1] == "b"
    assert extract_answer("no tags here") == (False, "no tags here")
    tag, content = extract_answer("<ANSWER>\n```json\n{\"x\": 1}\n```\n</ANSWER>")
    assert tag and content == '{"x": 1}'


def test_parse_json_tolerates_trailing_comma():
    _, ok, obj = parse_json_answer('<answer>{"a": 1,}</answer>')
    assert ok and obj == {"a": 1}


# --------------------------------------------------------------------- shell
@pytest.fixture
def vfs():
    return VFS(
        files={
            "logs/a.log": ["ERROR boom", "INFO ok", "ERROR again", "WARN meh"],
            "logs/b.log": ["INFO fine", "ERROR nope"],
            "data/u.csv": ["alice,us-east,200", "bob,us-west,500",
                           "alice,us-east,200", "carol,eu-central,404"],
        },
        dirs=["logs", "data"],
    )


def test_shell_grep_count(vfs):
    assert run_pipeline("grep -c ERROR logs/a.log", vfs) == ["2"]


def test_shell_pipeline_chain(vfs):
    out = run_pipeline("cut -d , -f 1 data/u.csv | sort | uniq -c | sort -n -r | head -n 1", vfs)
    assert out == ["2 alice"]


def test_shell_cat_multi_grep(vfs):
    assert run_pipeline("cat logs/a.log logs/b.log | grep -c ERROR", vfs) == ["3"]


def test_shell_find_size(vfs):
    big = run_pipeline("find logs -type f -size +30c", vfs)
    assert big == ["logs/a.log"]  # a.log is 40 bytes, b.log is 20


def test_shell_ls_wc(vfs):
    assert run_pipeline("ls logs | wc -l", vfs) == ["2"]


def test_shell_sort_u(vfs):
    out = run_pipeline("cut -d , -f 2 data/u.csv | sort -u", vfs)
    assert out == ["eu-central", "us-east", "us-west"]


@pytest.mark.parametrize("bad", [
    "echo 42", "rm -rf /", "cat logs/a.log > out.txt", "ls; ls",
    "cat `whoami`", "grep ERROR logs/a.log && ls", "cat /etc/passwd",
    "awk '{print}' logs/a.log",
])
def test_shell_rejects_unsupported(vfs, bad):
    with pytest.raises(CommandError):
        run_pipeline(bad, vfs)


# ---------------------------------------------------------------------- cron
def test_cron_equivalence():
    assert fire_set("*/30 * * * *") == fire_set("0,30 * * * *")
    assert fire_set("0 9-17 * * 1-5") != fire_set("0 9-17 * * *")


def test_cron_dow_seven_is_sunday():
    assert fire_set("0 12 * * 7") == fire_set("0 12 * * 0")


def test_cron_rejects_garbage():
    for bad in ["* * * *", "61 * * * *", "* 25 * * *", "a b c d e", "*/0 * * * *"]:
        with pytest.raises(CronError):
            parse_cron(bad)


def test_cron_weekday_window_size():
    # 9:00..17:59 every 30 min on weekdays, 4 weeks => 2 fires/h * 9h * 20 days
    assert len(fire_set("*/30 9-17 * * 1-5")) == 2 * 9 * 20


# -------------------------------------------------------------------- semver
def test_semver_parse():
    assert _parse_v("1.2.3") == (1, 2, 3)
    assert _parse_v("v2.0.1") == (2, 0, 1)
    assert _parse_v("1.2") is None
    assert _parse_v("1.2.x") is None


@pytest.mark.parametrize("v,c,ok", [
    ((1, 4, 2), "^1.2.0", True),
    ((2, 0, 0), "^1.2.0", False),
    ((1, 1, 9), "^1.2.0", False),
    ((0, 3, 5), "^0.3.1", True),
    ((0, 4, 0), "^0.3.1", False),
    ((1, 2, 9), "~1.2.3", True),
    ((1, 3, 0), "~1.2.3", False),
    ((1, 5, 0), ">=1.2.0 <2.0.0", True),
    ((2, 0, 0), ">=1.2.0 <2.0.0", False),
    ((1, 2, 3), "=1.2.3", True),
    ((1, 2, 4), "=1.2.3", False),
])
def test_semver_satisfies(v, c, ok):
    assert satisfies(v, c) is ok
