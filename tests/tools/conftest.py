"""Fixtures for the tools module tests.

Provides a fully deterministic ``FakeRuntime`` — in-memory implementations of every
core Protocol (`GraphStore`, `TimeSeriesStore`, `DocumentStore`, `ReplayEngine`,
`Simulator`, `EventIngestor`) seeded with a small but realistic book of demo data —
so the tool catalog is testable even while ``twin_core`` / ``twin_sim`` are being
built in parallel. ``tests/tools/test_integration.py`` exercises the real runtime
when those siblings are importable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import pytest

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import (
    ApprovalToken,
    EntityRef,
    EventEnvelope,
    Scenario,
    SimulationResult,
    ToolBand,
)
from fintwinos.policy.gates import PolicyGate, PolicyRule, RuleEffect, default_rules
from fintwinos.tools.catalog import build_default_registry

# ---------------------------------------------------------------------------
# Protocol implementations
# ---------------------------------------------------------------------------


class FakeGraphStore:
    """Networkx-backed GraphStore with the optional ``entities_of_type`` extension."""

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    def upsert_entity(self, ref: EntityRef, attributes: dict[str, Any] | None = None) -> None:
        key = ref.key()
        existing = dict(self._g.nodes[key]) if key in self._g else {}
        existing.update(attributes or {})
        self._g.add_node(key, **existing)

    def get_entity(self, ref: EntityRef) -> dict[str, Any] | None:
        key = ref.key()
        if key not in self._g:
            return None
        return dict(self._g.nodes[key])

    def add_relationship(
        self,
        src: EntityRef,
        dst: EntityRef,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        for ref in (src, dst):
            if ref.key() not in self._g:
                self._g.add_node(ref.key())
        self._g.add_edge(src.key(), dst.key(), kind=kind, **(attributes or {}))

    def neighbors(
        self, ref: EntityRef, kind: str | None = None, depth: int = 1
    ) -> list[EntityRef]:
        start = ref.key()
        if start not in self._g:
            return []
        undirected = self._g.to_undirected(as_view=True)
        seen = {start}
        frontier = [start]
        out: list[EntityRef] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for node in frontier:
                for neighbour in undirected.neighbors(node):
                    if neighbour in seen:
                        continue
                    if kind is not None:
                        edge_data = undirected.get_edge_data(node, neighbour) or {}
                        kinds = {d.get("kind") for d in edge_data.values()}
                        if kind not in kinds:
                            continue
                    seen.add(neighbour)
                    next_frontier.append(neighbour)
                    entity_type, entity_id = neighbour.split(":", 1)
                    out.append(EntityRef(entity_type=entity_type, entity_id=entity_id))
            frontier = next_frontier
        return out

    def subgraph(self, refs: Any, depth: int = 1) -> Any:
        keys: set[str] = set()
        for ref in refs:
            key = ref.key() if isinstance(ref, EntityRef) else str(ref)
            if key not in self._g:
                continue
            keys.add(key)
            for neighbour in self.neighbors(
                EntityRef(entity_type=key.split(":", 1)[0], entity_id=key.split(":", 1)[1]),
                depth=depth,
            ):
                keys.add(neighbour.key())
        return self._g.subgraph(keys)

    def stats(self) -> dict[str, Any]:
        return {"nodes": self._g.number_of_nodes(), "edges": self._g.number_of_edges()}

    # Optional extension probed by fintwinos.tools.domains.common.entities_of_type
    def entities_of_type(self, entity_type: str) -> list[tuple[EntityRef, dict[str, Any]]]:
        prefix = f"{entity_type}:"
        out = []
        for node, data in self._g.nodes(data=True):
            if node.startswith(prefix):
                ref = EntityRef(entity_type=entity_type, entity_id=node[len(prefix):])
                out.append((ref, dict(data)))
        return sorted(out, key=lambda pair: pair[0].entity_id)


class FakeTimeSeriesStore:
    """Minimal in-memory TimeSeriesStore."""

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[datetime, float]]] = {}

    def append(
        self, series_key: str, ts: datetime, value: float, tags: dict[str, Any] | None = None
    ) -> None:
        self._series.setdefault(series_key, []).append((ts, float(value)))
        self._series[series_key].sort(key=lambda pair: pair[0])

    def window(
        self, series_key: str, start: datetime | None = None, end: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        points = self._series.get(series_key, [])
        return [
            (ts, value)
            for ts, value in points
            if (start is None or ts >= start) and (end is None or ts <= end)
        ]

    def latest(self, series_key: str) -> tuple[datetime, float] | None:
        points = self._series.get(series_key)
        return points[-1] if points else None

    def keys(self) -> list[str]:
        return sorted(self._series)


class FakeDocumentStore:
    """Token-overlap keyword search over an in-memory document dict."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._docs[doc_id] = {"doc_id": doc_id, "text": text, "metadata": metadata or {}}

    def get(self, doc_id: str) -> dict[str, Any] | None:
        doc = self._docs.get(doc_id)
        return dict(doc) if doc else None

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        query_tokens = {t for t in query.lower().split() if t}
        if not query_tokens:
            return []
        scored: list[tuple[str, float]] = []
        for doc_id, doc in self._docs.items():
            doc_tokens = set(doc["text"].lower().split())
            overlap = len(query_tokens & doc_tokens)
            if overlap:
                scored.append((doc_id, overlap / len(query_tokens)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def count(self) -> int:
        return len(self._docs)


class FakeReplayEngine:
    """Episode recorder satisfying the ReplayEngine Protocol."""

    def __init__(self) -> None:
        self._episodes: dict[str, list[EventEnvelope]] = {}

    def record(self, envelope: EventEnvelope) -> None:
        self._episodes.setdefault(envelope.episode_id or "default", []).append(envelope)

    def episode(self, episode_id: str) -> list[EventEnvelope]:
        return list(self._episodes.get(episode_id, []))

    def episodes(self) -> list[str]:
        return sorted(self._episodes)

    def replay(self, episode_id: str, handler: Any) -> int:
        events = self._episodes.get(episode_id, [])
        for envelope in events:
            handler(envelope)
        return len(events)


class FakeIngestor:
    """Routes envelopes straight into the replay engine."""

    def __init__(self, replay: FakeReplayEngine) -> None:
        self._replay = replay

    def ingest(self, envelope: EventEnvelope) -> None:
        self._replay.record(envelope)


_BOUNDED_METRICS = {"precision", "recall", "sla_breach_rate"}


class FakeSimulator:
    """Deterministic Simulator: bootstrap-style CIs around fixed base metrics."""

    def __init__(self, name: str, base_metrics: dict[str, float]) -> None:
        self.name = name
        self._base = base_metrics

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        rng = np.random.default_rng(0 if seed is None else seed)
        metrics: dict[str, float] = {}
        confidence: dict[str, list[float]] = {}
        for key in sorted(self._base):
            base = self._base[key]
            draws = base * (1.0 + 0.05 * rng.standard_normal(512))
            if key in _BOUNDED_METRICS:
                draws = np.clip(draws, 0.0, 1.0)
            metrics[key] = round(float(np.mean(draws)), 6)
            lo, hi = np.percentile(draws, [5.0, 95.0])
            confidence[key] = [round(float(lo), 6), round(float(hi), 6)]
        series = {
            "sample_path": [round(float(v), 6) for v in np.cumsum(rng.standard_normal(8))]
        }
        return SimulationResult(
            simulator=self.name,
            scenario_name=scenario.name,
            ok=True,
            metrics=metrics,
            series=series,
            confidence=confidence,
            calibration={
                "method": "bootstrap",
                "coverage_80": 0.8,
                "coverage_95": 0.94,
                "crps": 0.11,
            },
            assumptions_version=scenario.assumptions_version,
            seed=seed,
        )

    def calibration_report(self) -> dict[str, Any]:
        return {"coverage_80": 0.8, "coverage_95": 0.94, "crps": 0.11}


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def build_fake_runtime(seed: int = 7) -> TwinRuntime:
    """Assemble a deterministic fake twin seeded with a small realistic book."""
    graph = FakeGraphStore()
    timeseries = FakeTimeSeriesStore()
    documents = FakeDocumentStore()
    replay = FakeReplayEngine()
    runtime = TwinRuntime(
        graph=graph,
        timeseries=timeseries,
        documents=documents,
        replay=replay,
        audit=AuditTrail(),
        ingestor=FakeIngestor(replay),
        metadata={"source": "fake_runtime", "seed": seed},
    )

    def ent(entity_type: str, entity_id: str, **attributes: Any) -> EntityRef:
        ref = EntityRef(entity_type=entity_type, entity_id=entity_id)
        graph.upsert_entity(ref, attributes)
        return ref

    # Customers / accounts / counterparty -------------------------------------
    cust1 = ent("customer", "CUST_1", name="Alice Howell", segment="retail",
                risk_rating="low", country="GB")
    cust2 = ent("customer", "CUST_2", name="Marlow Trading Ltd", segment="corporate",
                risk_rating="high", country="KY")
    acc1 = ent("account", "ACC_TRD_1", customer_id="CUST_1", currency="USD",
               account_type="trading", balance=2_000_000.0)
    acc2 = ent("account", "ACC_TRD_2", customer_id="CUST_2", currency="USD",
               account_type="trading", balance=7_500_000.0)
    counterparty = ent("counterparty", "CP_9", name="Nimbus Holdings", country="PA")
    graph.add_relationship(cust1, acc1, "owns")
    graph.add_relationship(cust2, acc2, "owns")
    graph.add_relationship(cust2, counterparty, "transacts_with")

    # Instruments and positions -------------------------------------------------
    ent("instrument", "INST_EQ_1", symbol="AAPL", asset_class="equity", currency="USD")
    ent("instrument", "INST_EQ_2", symbol="VOD", asset_class="equity", currency="GBP")
    ent("instrument", "INST_RT_1", symbol="UST10Y", asset_class="rates", currency="USD")
    ent("instrument", "INST_CR_1", symbol="XS_CORP_27", asset_class="credit", currency="EUR")
    ent("instrument", "INST_FX_1", symbol="EURUSD", asset_class="fx", currency="USD")
    positions = [
        ("POS_1", "ACC_TRD_1", "INST_EQ_1", 30_000, 4_500_000.0),
        ("POS_2", "ACC_TRD_1", "INST_RT_1", -12_000, -1_200_000.0),
        ("POS_3", "ACC_TRD_2", "INST_EQ_2", 800_000, 2_300_000.0),
        ("POS_4", "ACC_TRD_2", "INST_CR_1", 1_750, 1_750_000.0),
        ("POS_5", "ACC_TRD_2", "INST_FX_1", -500_000, -600_000.0),
    ]
    for position_id, account_id, instrument_id, quantity, market_value in positions:
        ref = ent(
            "position", position_id,
            account_id=account_id, instrument_id=instrument_id,
            quantity=quantity, market_value=market_value,
            as_of="2026-06-09T17:00:00+00:00",
        )
        account_ref = EntityRef(entity_type="account", entity_id=account_id)
        graph.add_relationship(account_ref, ref, "holds")

    # Limits ---------------------------------------------------------------------
    ent("limit", "LIM_EQ", scope_type="asset_class", scope_value="equity",
        limit_value=8_000_000.0)
    ent("limit", "LIM_CR", scope_type="asset_class", scope_value="credit",
        limit_value=1_500_000.0)  # deliberately breached: gross credit is 1.75m
    ent("limit", "LIM_GLOBAL", scope_type="global", scope_value="",
        limit_value=20_000_000.0)

    # Cash ladders ------------------------------------------------------------------
    ent(
        "cash_ladder", "LAD_USD",
        currency="USD", opening_balance=5_000_000.0,
        buckets=[
            {"day": 1, "inflow": 500_000.0, "outflow": 1_200_000.0},
            {"day": 2, "inflow": 300_000.0, "outflow": 2_500_000.0},
            {"day": 3, "inflow": 200_000.0, "outflow": 2_800_000.0},
            {"day": 4, "inflow": 1_500_000.0, "outflow": 200_000.0},
            {"day": 5, "inflow": 1_200_000.0, "outflow": 200_000.0},
            {"day": 6, "inflow": 400_000.0, "outflow": 300_000.0},
            {"day": 7, "inflow": 250_000.0, "outflow": 150_000.0},
        ],
    )
    ent(
        "cash_ladder", "LAD_EUR",
        currency="EUR", opening_balance=3_000_000.0,
        buckets=[
            {"day": 1, "inflow": 250_000.0, "outflow": 150_000.0},
            {"day": 2, "inflow": 200_000.0, "outflow": 180_000.0},
            {"day": 3, "inflow": 220_000.0, "outflow": 120_000.0},
        ],
    )

    # Compliance: alerts and a case ----------------------------------------------------
    alert_entities = [
        {"entity_type": "customer", "entity_id": "CUST_2"},
        {"entity_type": "account", "entity_id": "ACC_TRD_2"},
    ]
    ent("alert", "ALR_1", kind="structuring", severity="high", score=0.86, status="new",
        created_at="2026-06-08T09:15:00+00:00", entities=alert_entities)
    ent("alert", "ALR_2", kind="velocity", severity="medium", score=0.55, status="triaged",
        created_at="2026-06-07T14:00:00+00:00",
        entities=[{"entity_type": "customer", "entity_id": "CUST_2"}])
    ent("alert", "ALR_3", kind="sanctions_name_match", severity="low", score=0.20,
        status="dismissed", created_at="2026-06-05T10:30:00+00:00",
        entities=[{"entity_type": "customer", "entity_id": "CUST_1"}])
    case_ref = ent(
        "case", "CASE_7",
        kind="aml_alert", status="open", priority="high",
        opened_at="2026-06-08T10:00:00+00:00",
        narrative="System-generated cluster of structuring and velocity alerts.",
        entities=alert_entities,
    )
    graph.add_relationship(case_ref, cust2, "subject")
    graph.add_relationship(case_ref, acc2, "involves")

    # Customer-ops queues ------------------------------------------------------------
    ent("queue", "Q_COMPLAINTS", name="complaints", depth=42, arrival_rate_per_hour=18.0,
        avg_handle_minutes=14.0, agents_on_shift=6, sla_minutes=60.0)
    ent("queue", "Q_KYC", name="kyc_refresh", depth=11, arrival_rate_per_hour=6.0,
        avg_handle_minutes=22.0, agents_on_shift=4, sla_minutes=240.0)

    # Documents -------------------------------------------------------------------------
    documents.add(
        "filing_10k_acme",
        "Acme Bank annual report 10-K. Liquidity risk is managed via the liquidity "
        "coverage ratio and a buffer of high quality liquid assets. Funding "
        "concentration remains a key risk factor.",
        {"kind": "filing", "doc_type": "10-K", "issuer": "Acme Bank", "year": 2025},
    )
    documents.add(
        "filing_pillar3_acme",
        "Acme Bank Pillar 3 disclosure covering capital adequacy, leverage ratio and "
        "risk weighted assets across the trading and banking book.",
        {"kind": "filing", "doc_type": "Pillar3", "issuer": "Acme Bank", "year": 2025},
    )
    documents.add(
        "policy_aml_001",
        "AML policy on anti money laundering. Defines suspicious activity reporting "
        "thresholds, structuring detection rules and escalation paths for high risk "
        "customers.",
        {"kind": "policy", "tags": ["aml", "reporting"], "version": "2.1"},
    )
    documents.add(
        "policy_liq_001",
        "Liquidity risk management policy. Sets the internal liquidity buffer floor, "
        "cash ladder monitoring requirements and stress testing cadence for treasury.",
        {"kind": "policy", "tags": ["liquidity", "treasury"], "version": "1.4"},
    )
    documents.add(
        "proc_complaints_001",
        "Complaints handling procedure. Response time targets, final response letters "
        "and ombudsman referral rights for retail customers.",
        {"kind": "procedure", "tags": ["complaints"], "version": "3.0"},
    )

    # Timeseries ---------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    base = datetime.fromisoformat("2026-06-01T00:00:00+00:00")
    level = 100.0
    for day in range(9):
        level *= 1.0 + 0.01 * float(rng.standard_normal())
        timeseries.append("market.equity.spx", base.replace(day=1 + day), round(level, 4))

    # Simulators ------------------------------------------------------------------------
    runtime.register_simulator(
        FakeSimulator("market", {"pnl_p50": -2_400_000.0, "var_99": 5_100_000.0,
                                 "max_drawdown_pct": 7.5})
    )
    runtime.register_simulator(
        FakeSimulator("treasury", {"lcr": 1.32, "survival_days": 21.0,
                                   "peak_cumulative_outflow": 18_000_000.0})
    )
    runtime.register_simulator(
        FakeSimulator("compliance", {"precision": 0.62, "recall": 0.71,
                                     "alerts_per_day": 38.0})
    )
    runtime.register_simulator(
        FakeSimulator("customer_ops", {"avg_wait_minutes": 9.4, "sla_breach_rate": 0.08,
                                       "throughput_per_hour": 41.0})
    )
    return runtime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime() -> TwinRuntime:
    return build_fake_runtime()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Offline settings with the execute band disabled (the safe default)."""
    return Settings(
        offline=True, seed=7, data_dir=tmp_path / "data", execute_tools_enabled=False
    )


@pytest.fixture
def registry(fake_runtime, settings):
    """Default catalog: shipped policy gate, execute band disabled."""
    return build_default_registry(fake_runtime, settings=settings)


@pytest.fixture
def exec_settings(tmp_path) -> Settings:
    """Offline settings with the execute band explicitly enabled."""
    return Settings(
        offline=True, seed=7, data_dir=tmp_path / "data", execute_tools_enabled=True
    )


def allow_execute_gate(*patterns: str) -> PolicyGate:
    """Default policy pack plus explicit allow rules for the given tool patterns."""
    rules = default_rules()
    for pattern in patterns:
        rules.append(
            PolicyRule(
                name=f"allow-{pattern}",
                description=f"test allow rule for {pattern}",
                effect=RuleEffect.allow,
                bands=[ToolBand.execute],
                tools=[pattern],
            )
        )
    return PolicyGate(rules)


@pytest.fixture
def exec_registry(fake_runtime, exec_settings):
    """Catalog with the execute band enabled and every execute tool policy-allowed."""
    return build_default_registry(
        fake_runtime, policy_gate=allow_execute_gate("execute_*"), settings=exec_settings
    )


def approval_for(tool_name: str) -> ApprovalToken:
    """A valid human approval token scoped to one tool."""
    return ApprovalToken(subject=tool_name, granted_by="approver@bank.example", role="ops_lead")
