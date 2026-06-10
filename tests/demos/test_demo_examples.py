"""Smoke tests for the thin example scripts under examples/."""

from __future__ import annotations

import runpy
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_examples_directory_complete() -> None:
    names = {p.name for p in EXAMPLES_DIR.glob("*.py")}
    assert {
        "01_quickstart.py",
        "02_liquidity_stress.py",
        "03_aml_triage.py",
        "04_full_day.py",
    } <= names
    assert (EXAMPLES_DIR / "README.md").exists()


def test_quickstart_example_runs_offline(capsys) -> None:
    runpy.run_path(str(EXAMPLES_DIR / "01_quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "tool catalog" in out
    assert "audit chain verified: True" in out


def test_liquidity_example_runs_offline(capsys) -> None:
    runpy.run_path(str(EXAMPLES_DIR / "02_liquidity_stress.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "severe_squeeze" in out
    assert "audit verified: True" in out
