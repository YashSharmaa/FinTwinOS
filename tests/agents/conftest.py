"""Offline fixtures for the agents module tests.

The sibling modules (``twin_core``, ``twin_sim``, ``tools.catalog``) are built in
parallel, so these tests run against:

- protocol-conforming in-memory fake stores assembled into a real ``TwinRuntime``;
- a real ``ToolRegistry`` populated with deterministic fake observe/simulate/propose
  tools, so the full governance path (schema validation, policy gate, audit chain)
  is exercised exactly as in production;
- a genuinely offline ``LLMClient`` (``offline=True`` settings), so no test can ever
  touch the network.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from fintwinos.agents.base import AgentContext, Blackboard
from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import (
    EntityRef,
    EventEnvelope,
    RiskTier,
    SideEffectClass,
    ToolBand,
    utcnow,
)
from fintwinos.models.llm_routing.client import LLMClient, LLMResponse
from fintwinos.policy.gates import PolicyGate
from fintwinos.tools.registry import ToolRegistry, ToolSpec

# ---------------------------------------------------------------------------
# Fake twin stores (conform to the fintwinos.core.interfaces Protocols)
# ---------------------------------------------------------------------------


class FakeGraph:
    """Minimal in-memory GraphStore."""

    def __init__(self) -> None:
        self._entities: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str]] = []

    def upsert_entity(self, ref: EntityRef, attributes: dict[str, Any] | None = None) -> None:
        self._entities[ref.key()] = dict(attributes or {})

    def get_entity(self, ref: EntityRef) -> dict[str, Any] | None:
        return self._entities.get(ref.key())

    def add_relationship(
        self, src: EntityRef, dst: EntityRef, kind: str, attributes: dict[str, Any] | None = None
    ) -> None:
        self._edges.append((src.key(), dst.key(), kind))

    def neighbors(self, ref: EntityRef, kind: str | None = None, depth: int = 1) -> list[EntityRef]:
        out: list[EntityRef] = []
        for src, dst, edge_kind in self._edges:
            if src == ref.key() and (kind is None or edge_kind == kind):
                entity_type, entity_id = dst.split(":", 1)
                out.append(EntityRef(entity_type=entity_type, entity_id=entity_id))
        return out

    def subgraph(self, refs: Any, depth: int = 1) -> Any:
        keys = {r.key() for r in refs}
        return {
            "nodes": sorted(keys),
            "edges": [e for e in self._edges if e[0] in keys or e[1] in keys],
        }

    def stats(self) -> dict[str, Any]:
        return {"entities": len(self._entities), "relationships": len(self._edges)}


class FakeTimeSeries:
    """Minimal in-memory TimeSeriesStore."""

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[datetime, float]]] = {}

    def append(
        self, series_key: str, ts: datetime, value: float, tags: dict[str, Any] | None = None
    ) -> None:
        self._series.setdefault(series_key, []).append((ts, value))
        self._series[series_key].sort(key=lambda p: p[0])

    def window(
        self, series_key: str, start: datetime | None = None, end: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        points = self._series.get(series_key, [])
        return [
            p for p in points
            if (start is None or p[0] >= start) and (end is None or p[0] <= end)
        ]

    def latest(self, series_key: str) -> tuple[datetime, float] | None:
        points = self._series.get(series_key)
        return points[-1] if points else None

    def keys(self) -> list[str]:
        return sorted(self._series)


class FakeDocuments:
    """Minimal in-memory DocumentStore."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._docs[doc_id] = {"doc_id": doc_id, "text": text, "metadata": metadata or {}}

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._docs.get(doc_id)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        terms = query.lower().split()
        scored = [
            (doc_id, float(sum(doc["text"].lower().count(t) for t in terms)))
            for doc_id, doc in self._docs.items()
        ]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:k]

    def count(self) -> int:
        return len(self._docs)


class FakeReplay:
    """Minimal in-memory ReplayEngine."""

    def __init__(self) -> None:
        self._episodes: dict[str, list[EventEnvelope]] = {}

    def record(self, envelope: EventEnvelope) -> None:
        episode_id = envelope.episode_id or "default"
        self._episodes.setdefault(episode_id, []).append(envelope)

    def episode(self, episode_id: str) -> list[EventEnvelope]:
        return list(self._episodes.get(episode_id, []))

    def episodes(self) -> list[str]:
        return sorted(self._episodes)

    def replay(self, episode_id: str, handler: Any) -> int:
        events = self.episode(episode_id)
        for envelope in events:
            handler(envelope)
        return len(events)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_settings(**overrides: Any) -> Settings:
    """Deterministic, fully offline settings regardless of the host environment."""
    values: dict[str, Any] = {
        "offline": True,
        "execute_tools_enabled": False,
        "dual_control_required": True,
        "seed": 7,
        "openai_api_key": None,
    }
    values.update(overrides)
    return Settings(**values)


def build_fake_runtime(*, with_stale_series: bool = True) -> TwinRuntime:
    """A small but populated twin: series, entities, documents, one episode."""
    graph = FakeGraph()
    timeseries = FakeTimeSeries()
    documents = FakeDocuments()
    replay = FakeReplay()
    now = utcnow()

    for hours_ago, value in [(3, 101.2), (2, 100.8), (1, 101.5)]:
        timeseries.append("market.eq_index", now - timedelta(hours=hours_ago), value)
    for hours_ago, value in [(26, 812.0), (2, 805.0)]:
        timeseries.append("treasury.deposits_total", now - timedelta(hours=hours_ago), value)
    stale_age = timedelta(hours=120 if with_stale_series else 1)
    timeseries.append("ops.queue_depth", now - stale_age, 42.0)

    customer = EntityRef(entity_type="customer", entity_id="cus_001")
    account = EntityRef(entity_type="account", entity_id="acc_001")
    graph.upsert_entity(customer, {"name": "Acme Holdings", "segment": "corporate"})
    graph.upsert_entity(account, {"currency": "USD", "balance": 500.0})
    graph.add_relationship(customer, account, "owns")

    documents.add("pol_liquidity", "Liquidity risk policy: maintain LCR above 1.1 at all times.")
    documents.add("pol_aml", "AML policy: high severity alerts escalate to the MLRO.")
    documents.add("pol_conduct", "Conduct policy: vulnerable customers are prioritised.")

    replay.record(
        EventEnvelope(kind="trade.executed", source="oms", episode_id="ep_001")
    )
    replay.record(
        EventEnvelope(kind="alert.raised", source="tm_screening", episode_id="ep_001")
    )

    return TwinRuntime(
        graph=graph,
        timeseries=timeseries,
        documents=documents,
        replay=replay,
        audit=AuditTrail(),
        simulators={},
        ingestor=None,
        metadata={"seed": 7, "fixture": "tests.agents"},
    )


def _spec(
    name: str,
    band: ToolBand,
    description: str,
    risk_tier: RiskTier = RiskTier.low,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        band=band,
        description=description,
        input_schema={"type": "object"},
        risk_tier=risk_tier,
        side_effect=SideEffectClass.read if band == ToolBand.observe else SideEffectClass.none,
        owner="tests",
    )


def _scenario_severity(arguments: dict[str, Any]) -> str:
    scenario = arguments.get("scenario") or {}
    params = scenario.get("params") or arguments.get("params") or {}
    return str(params.get("severity") or arguments.get("severity") or "baseline")


def register_fake_tools(
    registry: ToolRegistry,
    *,
    alert_severity: str = "medium",
    alert_count: int = 1,
    ring_members: tuple[str, ...] = (),
    sla_breach_rate: float = 0.03,
    queue_depth: float = 42.0,
) -> None:
    """A deterministic stand-in for the default tool catalog, all four domains."""

    # -- treasury -----------------------------------------------------------------
    registry.register(
        _spec("observe_liquidity_ladder", ToolBand.observe, "Liquidity ladder buckets."),
        lambda arguments, context: {
            "buckets": {"1d": 120.0, "7d": 340.0, "30d": 510.0},
            "cash_balance": 500.0,
        },
    )
    registry.register(
        _spec("observe_funding_profile", ToolBand.observe, "Funding mix snapshot."),
        lambda arguments, context: {"deposits": 800.0, "wholesale": 200.0},
    )

    def liquidity_stress(arguments: dict[str, Any], context: Any) -> dict[str, Any]:
        severe = _scenario_severity(arguments) == "severe"
        seed = int(arguments.get("seed", 7))
        if severe:
            return {
                "simulator": "liquidity_stress",
                "scenario_name": "treasury_rehearsal",
                "metrics": {"lcr": 0.82, "shortfall": 35.0},
                "confidence": {"lcr": [0.70, 0.94]},
                "calibration": {"coverage_0_9": 0.91},
                "seed": seed,
            }
        return {
            "simulator": "liquidity_stress",
            "scenario_name": "treasury_rehearsal",
            "metrics": {"lcr": 1.40, "shortfall": 0.0},
            "confidence": {"lcr": [1.25, 1.55]},
            "calibration": {"coverage_0_9": 0.91},
            "seed": seed,
        }

    registry.register(
        _spec("simulate_liquidity_stress", ToolBand.simulate, "Liquidity stress rehearsal."),
        liquidity_stress,
    )
    registry.register(
        _spec(
            "propose_funding_plan", ToolBand.propose, "Lodge a funding plan proposal.",
            risk_tier=RiskTier.medium,
        ),
        lambda arguments, context: {"plan_id": "fp_001", "accepted": True},
    )

    # -- risk ----------------------------------------------------------------------
    registry.register(
        _spec("observe_positions", ToolBand.observe, "Position and exposure snapshot."),
        lambda arguments, context: {"gross_exposure": 1000.0, "net_exposure": 250.0},
    )

    def market_shock(arguments: dict[str, Any], context: Any) -> dict[str, Any]:
        severe = _scenario_severity(arguments) == "severe"
        loss = 150.0 if severe else 20.0
        var_99 = 180.0 if severe else 60.0
        return {
            "simulator": "market_shock",
            "scenario_name": "risk_rehearsal",
            "metrics": {"shock_loss": loss, "var_99": var_99},
            "confidence": {"shock_loss": [round(loss * 0.8, 4), round(loss * 1.2, 4)]},
            "calibration": {"method": "bootstrap"},
            "seed": int(arguments.get("seed", 7)),
        }

    registry.register(
        _spec("simulate_market_shock", ToolBand.simulate, "Market shock rehearsal."),
        market_shock,
    )
    registry.register(
        _spec(
            "propose_hedge", ToolBand.propose, "Lodge a hedge proposal.",
            risk_tier=RiskTier.medium,
        ),
        lambda arguments, context: {"hedge_id": "hg_001", "accepted": True},
    )

    # -- compliance ------------------------------------------------------------------
    def alerts(arguments: dict[str, Any], context: Any) -> dict[str, Any]:
        return {
            "alerts": [
                {
                    "alert_id": f"alr_{i:03d}",
                    "kind": "aml_structuring",
                    "severity": alert_severity,
                    "score": round(0.4 + 0.1 * i, 2),
                }
                for i in range(alert_count)
            ]
        }

    registry.register(
        _spec("observe_alerts", ToolBand.observe, "Open transaction-monitoring alerts."),
        alerts,
    )
    registry.register(
        _spec("observe_case_graph", ToolBand.observe, "Entity ring around open cases."),
        lambda arguments, context: {"ring_members": list(ring_members)},
    )
    registry.register(
        _spec("simulate_alert_triage", ToolBand.simulate, "Alert triage rehearsal."),
        lambda arguments, context: {
            "simulator": "alert_triage",
            "scenario_name": "compliance_rehearsal",
            "metrics": {"expected_true_positive_rate": 0.42},
            "confidence": {"expected_true_positive_rate": [0.30, 0.55]},
            "calibration": {"method": "historical"},
            "seed": int(arguments.get("seed", 7)),
        },
    )
    registry.register(
        _spec(
            "propose_case_action", ToolBand.propose, "Lodge a case action proposal.",
            risk_tier=RiskTier.medium,
        ),
        lambda arguments, context: {"case_action_id": "ca_001", "accepted": True},
    )

    # -- customer ops -----------------------------------------------------------------
    registry.register(
        _spec("observe_ops_queues", ToolBand.observe, "Operational queue snapshot."),
        lambda arguments, context: {
            "queue_depth": queue_depth,
            "sla_breach_rate": sla_breach_rate,
            "complaints": 5.0,
        },
    )
    registry.register(
        _spec("simulate_queue_staffing", ToolBand.simulate, "Queue staffing rehearsal."),
        lambda arguments, context: {
            "simulator": "queue_staffing",
            "scenario_name": "customer_ops_rehearsal",
            "metrics": {"projected_sla_breach_rate": round(sla_breach_rate * 0.8, 6)},
            "confidence": {
                "projected_sla_breach_rate": [
                    round(sla_breach_rate * 0.6, 6),
                    round(sla_breach_rate * 1.0, 6),
                ]
            },
            "calibration": {"method": "queueing_mc"},
            "seed": int(arguments.get("seed", 7)),
        },
    )
    registry.register(
        _spec(
            "propose_queue_rebalance", ToolBand.propose, "Lodge a queue rebalance proposal.",
            risk_tier=RiskTier.medium,
        ),
        lambda arguments, context: {"rebalance_id": "qr_001", "accepted": True},
    )


def build_fake_registry(
    runtime: TwinRuntime | None,
    settings: Settings,
    *,
    policy_gate: Any | None = None,
    with_tools: bool = True,
    **tool_overrides: Any,
) -> ToolRegistry:
    """A real ToolRegistry sharing the runtime's audit chain, with fake tools."""
    registry = ToolRegistry(
        policy_gate=policy_gate if policy_gate is not None else PolicyGate(),
        audit=runtime.audit if runtime is not None else AuditTrail(),
        settings=settings,
    )
    if with_tools:
        register_fake_tools(registry, **tool_overrides)
    return registry


class ScriptedLLM:
    """Mimics an *online* LLMClient with a canned response and zero network."""

    def __init__(self, text: str):
        self.offline = False
        self.text = text
        self.calls = 0

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text, model="scripted", offline=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def offline_llm(settings: Settings) -> LLMClient:
    client = LLMClient(settings=settings)
    assert client.offline, "test LLM client must be offline"
    return client


@pytest.fixture
def runtime() -> TwinRuntime:
    return build_fake_runtime()


@pytest.fixture
def registry(runtime: TwinRuntime, settings: Settings) -> ToolRegistry:
    return build_fake_registry(runtime, settings)


@pytest.fixture
def ctx(
    runtime: TwinRuntime,
    registry: ToolRegistry,
    offline_llm: LLMClient,
    settings: Settings,
) -> AgentContext:
    return AgentContext(
        runtime=runtime,
        registry=registry,
        llm=offline_llm,
        blackboard=Blackboard(),
        settings=settings,
        audit=runtime.audit,
    )
