"""Reliability contracts for evaluation and reporting."""

from pathlib import Path

import pytest
import yaml

from atlas_rl.evaluation import run_eval
from atlas_rl.evaluation.report import build_report


class _FailingBackend:
    name = "failing"

    def __init__(self):
        self.closed = False

    def complete(self, instance, cfg):
        raise RuntimeError("provider unavailable")

    def close(self):
        self.closed = True


def test_eval_records_backend_errors_and_closes(monkeypatch):
    backend = _FailingBackend()
    monkeypatch.setattr(run_eval, "make_backend", lambda spec: backend)
    result = run_eval.evaluate(
        "test:failing", ["cron_author"], n_per_env=2,
        difficulties=[2], progress=False)
    summary = run_eval.summarize(result)

    assert backend.closed
    assert summary["overall"]["error_samples"] == 2
    assert summary["overall"]["instances_with_errors"] == 2
    assert summary["overall"]["pass1"] == 0.0


def test_eval_summary_records_seed_provenance():
    result = run_eval.evaluate(
        "mock:empty", ["semver_resolve"], 1, [2],
        seed_offset=123_456, progress=False,
    )
    summary = run_eval.summarize(result)

    assert result["rows"][0]["seed"] == 1_123_456
    assert summary["config"]["seed_offset"] == 123_456
    assert summary["config"]["seed_base"] == 1_123_456


def test_eval_rejects_invalid_workloads():
    with pytest.raises(ValueError):
        run_eval.evaluate("mock:oracle", [], 1, [1], progress=False)
    with pytest.raises(ValueError):
        run_eval.evaluate("mock:oracle", ["cron_author"], 0, [1], progress=False)
    with pytest.raises(ValueError):
        run_eval.evaluate("mock:oracle", ["cron_author"], 1, [0], progress=False)


def test_report_rejects_missing_baseline(tmp_path):
    with pytest.raises(ValueError, match="baseline directory"):
        build_report([], str(tmp_path / "missing"))


def test_all_yaml_configs_load():
    for path in Path("configs").glob("*.yaml"):
        with path.open() as f:
            assert isinstance(yaml.safe_load(f), dict), path
