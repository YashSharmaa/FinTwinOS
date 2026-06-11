"""Integration tests against the real twin_core/twin_sim runtime.

These are skipped automatically while the sibling modules are still being built;
the integration phase re-runs them once `fintwinos.twin_core.runtime.build_runtime`
and `fintwinos.twin_sim.register_all` are importable.
"""

from __future__ import annotations

import pytest

from fintwinos.core.config import Settings
from fintwinos.tools.catalog import build_default_registry
from fintwinos.tools.registry import CallContext

twin_core_runtime = pytest.importorskip(
    "fintwinos.twin_core.runtime", reason="twin_core is being built in parallel"
)
twin_sim = pytest.importorskip(
    "fintwinos.twin_sim", reason="twin_sim is being built in parallel"
)

CTX = CallContext(caller="test-integration")


@pytest.fixture
def real_runtime():
    runtime = twin_core_runtime.build_runtime(seed=7, with_demo_data=True)
    twin_sim.register_all(runtime)
    return runtime


@pytest.fixture
def real_registry(real_runtime, tmp_path):
    settings = Settings(offline=True, seed=7, data_dir=tmp_path / "data")
    return build_default_registry(real_runtime, settings=settings)


def test_real_registry_builds_full_catalog(real_registry, real_runtime):
    assert len(real_registry) >= 18
    assert real_registry.audit is real_runtime.audit


async def test_observe_band_works_on_real_twin(real_registry):
    for tool, arguments in (
        ("observe_positions", {}),
        ("observe_exposures", {}),
        ("observe_cash_ladder", {}),
        ("observe_alerts", {}),
        ("observe_queue_state", {}),
        ("observe_market_regime", {}),
    ):
        result = await real_registry.call(tool, arguments, CTX)
        assert result.ok, f"{tool}: {result.error}"


async def test_observe_market_regime_detects_on_price_series(real_registry):
    """The regime tool runs the z-score detector over a real twin price series."""
    result = await real_registry.call(
        "observe_market_regime", {"on": "abs_returns", "window": 20}, CTX
    )
    assert result.ok, result.error
    data = result.data
    assert data["series_key"].startswith("price:")
    assert data["current_regime"] in {"calm", "stressed"}
    assert 0.0 <= data["stressed_fraction"] <= 1.0
    assert data["n_change_points"] == len(data["change_points"])


async def test_simulate_liquidity_stress_on_real_twin(real_registry, real_runtime):
    if "treasury" not in real_runtime.simulators:
        pytest.skip("real twin_sim does not expose a 'treasury' simulator yet")
    result = await real_registry.call(
        "simulate_liquidity_stress",
        {
            "horizon_days": 10,
            "stress_name": "integration_smoke",
            "currencies": ["USD"],
            "assumptions_version": "v1",
            "ticket_id": "TCK-INT-1",
        },
        CTX,
    )
    assert result.ok, result.error
    assert result.data["confidence"], "simulators must return confidence intervals"
