"""Tests for the contracted entry point ``fintwinos.twin_sim.register_all``."""

from __future__ import annotations

from fintwinos.core.interfaces import Simulator
from fintwinos.core.types import Scenario
from fintwinos.twin_sim import register_all

EXPECTED_NAMES = {"market", "treasury", "compliance", "customer_ops"}


def test_register_all_populates_exactly_four_simulators(runtime):
    register_all(runtime)
    assert set(runtime.simulators) == EXPECTED_NAMES
    assert len(runtime.simulators) == 4


def test_registered_simulators_satisfy_the_protocol(runtime):
    register_all(runtime)
    for name, simulator in runtime.simulators.items():
        assert isinstance(simulator, Simulator), name
        assert simulator.name == name
        report = simulator.calibration_report()
        assert report["simulator"] == name


def test_register_all_is_idempotent(runtime):
    register_all(runtime)
    register_all(runtime)
    assert set(runtime.simulators) == EXPECTED_NAMES
    assert len(runtime.simulators) == 4


def test_register_all_appends_audit_record(runtime):
    before = len(runtime.audit)
    register_all(runtime)
    records = runtime.audit.records(action="simulators.registered")
    assert len(runtime.audit) > before
    assert records, "expected a simulators.registered audit record"
    assert sorted(records[-1].payload["simulators"]) == sorted(EXPECTED_NAMES)
    assert records[-1].payload["count"] == 4
    assert runtime.audit.verify()


def test_treasury_simulator_is_wired_to_the_runtime(runtime):
    register_all(runtime)
    assert runtime.simulators["treasury"].runtime is runtime


def test_registered_simulators_run_end_to_end(runtime):
    register_all(runtime)
    quick_params = {
        "market": {"n_steps": 64, "n_paths": 4},
        "treasury": {"horizon_days": 10, "n_reps": 4},
        "compliance": {"n_steps": 20, "n_accounts": 60, "n_reps": 4},
        "customer_ops": {"horizon_hours": 2.0, "n_reps": 4},
    }
    for name, simulator in runtime.simulators.items():
        scenario = Scenario(name=f"smoke_{name}", params=quick_params[name])
        result = simulator.run(scenario, seed=77)
        assert result.ok is True
        assert result.simulator == name
        assert result.seed == 77
        assert result.metrics
        assert set(result.confidence) == set(result.metrics)
        for lo_hi in result.confidence.values():
            assert len(lo_hi) == 2 and lo_hi[0] <= lo_hi[1]
        assert "ks_stat" in result.calibration
        assert "calibrated" in result.calibration
