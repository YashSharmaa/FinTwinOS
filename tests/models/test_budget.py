"""Tests for the LLM budget guard: ceilings, decorator, context manager, audit."""

from __future__ import annotations

import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.models.llm_routing.budget import BudgetExceeded, BudgetGuard
from fintwinos.models.llm_routing.client import LLMResponse


def make_response(cost: float) -> LLMResponse:
    return LLMResponse(
        text="ok",
        model="gpt-5-mini",
        usage={"input_tokens": 100, "output_tokens": 50},
        cost_usd=cost,
    )


class TestCharging:
    def test_accumulates_below_ceiling(self):
        guard = BudgetGuard(ceiling_usd=0.01)
        remaining = guard.record(make_response(0.004))
        assert remaining == pytest.approx(0.006)
        guard.record(make_response(0.004))
        assert guard.spent_usd == pytest.approx(0.008)
        assert not guard.exhausted

    def test_trips_when_crossing_ceiling(self):
        guard = BudgetGuard(ceiling_usd=0.01)
        guard.record(make_response(0.004))
        guard.record(make_response(0.004))
        with pytest.raises(BudgetExceeded) as excinfo:
            guard.record(make_response(0.004))
        assert excinfo.value.spent_usd == pytest.approx(0.012)
        assert excinfo.value.ceiling_usd == 0.01
        # the spend that tripped the guard is still on the meter
        assert guard.spent_usd == pytest.approx(0.012)
        assert guard.remaining_usd == 0.0
        assert guard.exhausted

    def test_trips_exactly_at_the_ceiling(self):
        guard = BudgetGuard(ceiling_usd=0.01)
        with pytest.raises(BudgetExceeded):
            guard.charge(0.01)

    def test_exception_carries_the_response(self):
        guard = BudgetGuard(ceiling_usd=0.001)
        response = make_response(0.005)
        with pytest.raises(BudgetExceeded) as excinfo:
            guard.record(response)
        assert excinfo.value.payload is response

    def test_zero_cost_responses_are_free(self):
        guard = BudgetGuard(ceiling_usd=0.001)
        offline = LLMResponse(text="stub", offline=True)  # cost_usd defaults to 0
        for _ in range(10):
            guard.record(offline)
        assert guard.spent_usd == 0.0

    def test_objects_without_cost_are_free(self):
        guard = BudgetGuard(ceiling_usd=0.001)
        guard.record(object())
        assert guard.spent_usd == 0.0

    def test_negative_charge_rejected(self):
        with pytest.raises(ValueError):
            BudgetGuard(ceiling_usd=1.0).charge(-0.1)

    def test_invalid_ceiling_rejected(self):
        with pytest.raises(ValueError):
            BudgetGuard(ceiling_usd=0.0)

    def test_reset_restores_headroom(self):
        guard = BudgetGuard(ceiling_usd=0.01)
        with pytest.raises(BudgetExceeded):
            guard.charge(0.02)
        guard.reset()
        assert guard.spent_usd == 0.0
        assert not guard.exhausted
        guard.ensure_available()  # does not raise

    def test_summary_snapshot(self):
        guard = BudgetGuard(ceiling_usd=1.0)
        guard.charge(0.25)
        summary = guard.summary()
        assert summary["spent_usd"] == pytest.approx(0.25)
        assert summary["remaining_usd"] == pytest.approx(0.75)
        assert summary["exhausted"] is False


class TestDecorator:
    async def test_async_decorator_meters_and_trips(self):
        guard = BudgetGuard(ceiling_usd=0.01)
        calls = 0

        @guard.wrap
        async def ask() -> LLMResponse:
            nonlocal calls
            calls += 1
            return make_response(0.006)

        response = await ask()
        assert response.text == "ok"
        assert guard.spent_usd == pytest.approx(0.006)

        with pytest.raises(BudgetExceeded) as excinfo:
            await ask()  # 0.012 >= 0.01 -> trips while recording
        assert excinfo.value.payload is not None
        assert calls == 2

        with pytest.raises(BudgetExceeded):
            await ask()  # blocked before invocation this time
        assert calls == 2  # the function never ran

    def test_sync_decorator(self):
        guard = BudgetGuard(ceiling_usd=0.01)

        @guard.wrap
        def ask() -> LLMResponse:
            return make_response(0.004)

        ask()
        ask()
        with pytest.raises(BudgetExceeded):
            ask()
        assert guard.spent_usd == pytest.approx(0.012)

    def test_decorator_preserves_metadata(self):
        guard = BudgetGuard(ceiling_usd=1.0)

        @guard.wrap
        def documented_call() -> LLMResponse:
            """Docstring survives wrapping."""
            return make_response(0.001)

        assert documented_call.__name__ == "documented_call"
        assert "survives" in documented_call.__doc__


class TestContextManager:
    def test_context_allows_work_within_budget(self):
        guard = BudgetGuard(ceiling_usd=1.0)
        with guard as g:
            g.charge(0.5)
        assert guard.spent_usd == pytest.approx(0.5)

    def test_entry_blocked_when_exhausted(self):
        guard = BudgetGuard(ceiling_usd=0.01)
        with pytest.raises(BudgetExceeded):
            guard.charge(0.02)
        with pytest.raises(BudgetExceeded):
            with guard:
                pytest.fail("body must not run on an exhausted budget")

    def test_exceptions_are_not_suppressed(self):
        guard = BudgetGuard(ceiling_usd=1.0)
        with pytest.raises(RuntimeError):
            with guard:
                raise RuntimeError("boom")


class TestAuditIntegration:
    def test_charges_and_breach_are_audited(self):
        trail = AuditTrail()
        guard = BudgetGuard(ceiling_usd=0.01, audit=trail, name="case-budget")
        guard.charge(0.004)
        with pytest.raises(BudgetExceeded):
            guard.charge(0.02)

        charged = trail.records(action="budget.charged")
        exceeded = trail.records(action="budget.exceeded")
        assert len(charged) == 2
        assert len(exceeded) == 1
        assert charged[0].actor == "case-budget"
        assert exceeded[0].payload["ceiling_usd"] == 0.01
        assert trail.verify()
