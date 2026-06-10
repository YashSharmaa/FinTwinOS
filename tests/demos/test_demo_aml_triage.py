"""End-to-end offline tests for the AML triage demo, including both execute branches."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from fintwinos.demos import aml_triage


def _run(seed: int = 7) -> dict[str, Any]:
    return aml_triage.run(seed=seed, console=Console(file=io.StringIO(), width=120))


def test_run_offline_end_to_end_returns_contracted_keys() -> None:
    result = _run()
    assert result["demo"] == "aml_triage"
    assert result["offline"] is True
    for key in ("surge", "queue_size", "scorer", "triage", "top_alert",
                "narrative", "close_case", "audit", "warnings"):
        assert key in result
    assert result["queue_size"] >= 20


def test_scorer_separates_ring_from_benign() -> None:
    result = _run()
    assert result["scorer"]["auc"] >= 0.7
    assert result["scorer"]["source"]


def test_triage_table_precision_recall_tradeoff() -> None:
    triage = _run()["triage"]
    assert len(triage) >= 3
    for row in triage:
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
    recalls = [row["recall"] for row in triage]
    assert recalls == sorted(recalls)  # recall is monotone in triage depth


def test_top_alert_and_narrative() -> None:
    result = _run()
    top = result["top_alert"]
    assert top["alert_id"]
    assert 0.0 <= top["score"] <= 1.0
    assert isinstance(result["narrative"], str) and len(result["narrative"]) > 60


def test_close_case_blocked_without_approval() -> None:
    blocked = _run()["close_case"]["blocked"]
    assert blocked["ok"] is False
    assert blocked["requires_approval"] is True
    assert blocked["error"]


def test_close_case_executes_with_granted_approval_and_writes_outbox() -> None:
    close_case = _run()["close_case"]
    approved = close_case["approved"]
    assert approved["ok"] is True
    assert approved["approval_token"].startswith("apr_")
    outbox_dir = Path(close_case["outbox_dir"])
    assert outbox_dir.exists()
    files = close_case["outbox_files"]
    assert len(files) >= 1
    payload = Path(files[0]).read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(payload[-1])
    assert record["case_id"] == aml_triage.CASE_ID


def test_execute_kill_switch_restored_after_run() -> None:
    # The demo flips execute_tools_enabled on for the approved branch; the
    # deployment default (off) must be restored afterwards.
    import fintwinos.demos.stack as stack_mod

    console = Console(file=io.StringIO(), width=120)
    stack = stack_mod.build_stack(seed=7, console=console)
    import asyncio

    asyncio.run(aml_triage.arun(stack=stack, console=console, seed=7))
    assert stack.registry.settings.execute_tools_enabled is False


def test_determinism_same_seed_same_triage() -> None:
    first = _run(seed=21)
    second = _run(seed=21)
    assert first["triage"] == second["triage"]
    assert first["scorer"]["auc"] == second["scorer"]["auc"]
    assert first["top_alert"]["alert_id"] == second["top_alert"]["alert_id"]
