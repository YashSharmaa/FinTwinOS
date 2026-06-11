"""Consolidated offline-RL pipeline: JSON-serialisable report across all envs."""

from __future__ import annotations

import json
from pathlib import Path

from fintwinos.rl.envs import ENV_NAMES
from fintwinos.rl.pipeline import VERDICTS, run_rl_pipeline


def test_run_rl_pipeline_covers_every_env():
    report = run_rl_pipeline(seed=7, epochs=8, n_episodes=15)
    assert set(report["results"]) == set(ENV_NAMES)
    assert report["seed"] == 7
    assert sum(report["verdicts"].values()) == len(ENV_NAMES)
    assert set(report["verdicts"]) == set(VERDICTS)
    for env_report in report["results"].values():
        assert env_report["verdict"] in VERDICTS
        assert "doubly_robust" in env_report["ope"]
        assert "estimate" in env_report["ope"]["doubly_robust"]


def test_run_rl_pipeline_is_deterministic_and_serialisable(tmp_path: Path):
    a = run_rl_pipeline(seed=11, report_dir=tmp_path / "a", epochs=6, n_episodes=10)
    b = run_rl_pipeline(seed=11, report_dir=tmp_path / "b", epochs=6, n_episodes=10)
    # Same seed -> identical verdicts.
    assert a["verdicts"] == b["verdicts"]
    # The written report round-trips through JSON unchanged.
    written = json.loads((tmp_path / "a" / "report.json").read_text())
    assert written["verdicts"] == a["verdicts"]
    assert a["report_path"].endswith("report.json")


def test_run_rl_pipeline_defaults_seed_from_settings(monkeypatch):
    monkeypatch.setenv("FINTWIN_SEED", "3")
    from fintwinos.core.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        report = run_rl_pipeline(epochs=5, n_episodes=8)
        assert report["seed"] == 3
    finally:
        get_settings.cache_clear()  # type: ignore[attr-defined]
