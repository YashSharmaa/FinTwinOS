"""Integration smoke tests against the real twin_core runtime and tool catalog.

These are intentionally left in place while the sibling modules are built in
parallel: ``pytest.importorskip`` skips them until ``twin_core`` and
``tools.catalog`` exist, after which they run for real in the integration phase.
Everything stays offline (``offline=True`` settings), so they never touch the
network even with the full stack present.
"""

from __future__ import annotations

import pytest

from fintwinos.agents.runtime import handle_case
from fintwinos.models.llm_routing.client import LLMClient

from .conftest import make_settings

twin_core_runtime = pytest.importorskip(
    "fintwinos.twin_core.runtime", reason="twin_core is being built in parallel"
)
catalog = pytest.importorskip(
    "fintwinos.tools.catalog", reason="tools.catalog is being built in parallel"
)


def _build_stack():
    settings = make_settings()
    runtime = twin_core_runtime.build_runtime(seed=7, with_demo_data=True)
    try:  # simulators enrich the rehearsals but are not required for the smoke test
        import fintwinos.twin_sim as twin_sim

        twin_sim.register_all(runtime)
    except ImportError:
        pass
    registry = catalog.build_default_registry(runtime, settings=settings)
    return runtime, registry, LLMClient(settings=settings)


async def test_handle_case_on_real_runtime_smoke():
    runtime, registry, llm = _build_stack()
    result = await handle_case(
        "case-real-001",
        "Review the current liquidity and funding buffers",
        runtime,
        registry,
        llm=llm,
    )
    assert result["status"] in {"complete", "awaiting_human", "blocked"}
    for key in ("decision", "policy", "outputs", "critique", "audit_len"):
        assert key in result
    assert result["audit_len"] == len(runtime.audit)
    assert runtime.audit.verify()


async def test_real_runtime_high_risk_routes_to_human():
    runtime, registry, llm = _build_stack()
    result = await handle_case(
        "case-real-002",
        "Hedge the equity exposure against a severe market shock immediately",
        runtime,
        registry,
        llm=llm,
    )
    # the escalation keywords guarantee at least a high-tier proposal
    assert result["status"] in {"awaiting_human", "blocked"}
    assert result["policy"]["requires_human_review"] or not result["policy"]["allowed"]
