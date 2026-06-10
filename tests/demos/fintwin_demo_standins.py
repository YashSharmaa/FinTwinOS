"""Contract-faithful stand-ins for sibling modules built in parallel.

The demos code exclusively against the canonical entry points in CONTRACTS.md:

* ``fintwinos.twin_core.runtime.build_runtime``
* ``fintwinos.twin_sim.register_all``
* ``fintwinos.tools.catalog.build_default_registry``
* ``fintwinos.agents.runtime.handle_case``

Those modules are owned by other teams and may not exist yet in a partial
checkout. ``conftest.py`` installs the minimal implementations below — honouring
every hard rule (tool bands, execute gating, audit, SimulationResult shapes) —
**only** for entry points whose real module is missing, so this suite exercises
the demos end-to-end today and switches to the real platform automatically at
integration time. It doubles as an executable specification of what the demos
expect from their siblings.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fintwinos.core.audit import AuditTrail
from fintwinos.core.config import Settings, get_settings
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import (
    Decision,
    EntityRef,
    EventEnvelope,
    RiskTier,
    Scenario,
    SideEffectClass,
    SimulationResult,
    ToolBand,
    utcnow,
)
from fintwinos.demos import _fallbacks
from fintwinos.demos._corpus import tokens
from fintwinos.policy.gates import PolicyGate
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

# ---------------------------------------------------------------------------
# In-memory stores implementing the core Protocols
# ---------------------------------------------------------------------------


class MemoryGraph:
    """Minimal ``GraphStore``: dict-backed entities and typed relationships."""

    def __init__(self) -> None:
        self._entities: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str, dict[str, Any]]] = []

    def upsert_entity(self, ref: EntityRef, attributes: dict[str, Any] | None = None) -> None:
        self._entities.setdefault(ref.key(), {}).update(attributes or {})

    def get_entity(self, ref: EntityRef) -> dict[str, Any] | None:
        return self._entities.get(ref.key())

    def add_relationship(
        self, src: EntityRef, dst: EntityRef, kind: str, attributes: dict[str, Any] | None = None
    ) -> None:
        self._edges.append((src.key(), dst.key(), kind, attributes or {}))

    def neighbors(self, ref: EntityRef, kind: str | None = None, depth: int = 1) -> list[EntityRef]:
        out: list[EntityRef] = []
        for src, dst, edge_kind, _ in self._edges:
            if kind is not None and edge_kind != kind:
                continue
            if src == ref.key():
                etype, eid = dst.split(":", 1)
                out.append(EntityRef(entity_type=etype, entity_id=eid))
        return out

    def subgraph(self, refs: Any, depth: int = 1) -> dict[str, Any]:
        keys = {r.key() for r in refs}
        return {
            "nodes": sorted(keys),
            "edges": [e for e in self._edges if e[0] in keys or e[1] in keys],
        }

    def stats(self) -> dict[str, Any]:
        return {"entities": len(self._entities), "relationships": len(self._edges)}


class MemoryTimeSeries:
    """Minimal ``TimeSeriesStore`` keyed by series name."""

    def __init__(self) -> None:
        self._series: dict[str, list[tuple[datetime, float]]] = {}

    def append(
        self, series_key: str, ts: datetime, value: float, tags: dict[str, Any] | None = None
    ) -> None:
        self._series.setdefault(series_key, []).append((ts, float(value)))

    def window(
        self, series_key: str, start: datetime | None = None, end: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        points = self._series.get(series_key, [])
        return [
            (ts, v)
            for ts, v in points
            if (start is None or ts >= start) and (end is None or ts <= end)
        ]

    def latest(self, series_key: str) -> tuple[datetime, float] | None:
        points = self._series.get(series_key)
        return points[-1] if points else None

    def keys(self) -> list[str]:
        return sorted(self._series)


class MemoryDocs:
    """Minimal ``DocumentStore`` with token-overlap relevance scoring."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._docs[doc_id] = {"doc_id": doc_id, "text": text, "metadata": metadata or {}}

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._docs.get(doc_id)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        scored: list[tuple[str, float]] = []
        for doc_id, doc in self._docs.items():
            overlap = len(query_tokens & tokens(doc["text"]))
            if overlap:
                scored.append((doc_id, round(overlap / len(query_tokens), 4)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def count(self) -> int:
        return len(self._docs)


class MemoryReplay:
    """Minimal ``ReplayEngine`` recording envelopes into episodes."""

    def __init__(self) -> None:
        self._episodes: dict[str, list[EventEnvelope]] = {}

    def record(self, envelope: EventEnvelope) -> None:
        self._episodes.setdefault(envelope.episode_id or "default", []).append(envelope)

    def episode(self, episode_id: str) -> list[EventEnvelope]:
        return list(self._episodes.get(episode_id, []))

    def episodes(self) -> list[str]:
        return sorted(self._episodes)

    def replay(self, episode_id: str, handler: Any) -> int:
        events = self.episode(episode_id)
        for envelope in events:
            handler(envelope)
        return len(events)


class MemoryIngestor:
    """Minimal ``EventIngestor`` routing envelopes into the replay engine."""

    def __init__(self, replay: MemoryReplay, audit: AuditTrail) -> None:
        self._replay = replay
        self._audit = audit

    def ingest(self, envelope: EventEnvelope) -> None:
        self._replay.record(envelope)
        self._audit.append("ingestor", "event.ingested", {"kind": envelope.kind})


# ---------------------------------------------------------------------------
# Entry point: fintwinos.twin_core.runtime.build_runtime
# ---------------------------------------------------------------------------


def build_runtime(seed: int = 7, with_demo_data: bool = True) -> TwinRuntime:
    """Assemble an in-memory TwinRuntime with a sprinkle of demo data."""
    audit = AuditTrail()
    replay = MemoryReplay()
    runtime = TwinRuntime(
        graph=MemoryGraph(),
        timeseries=MemoryTimeSeries(),
        documents=MemoryDocs(),
        replay=replay,
        audit=audit,
        ingestor=MemoryIngestor(replay, audit),
        metadata={"seed": seed, "standin": True},
    )
    if with_demo_data:
        rng = np.random.default_rng(seed)
        now = utcnow()
        for day, value in enumerate(120.0 + np.cumsum(rng.normal(-0.8, 2.0, 30))):
            runtime.timeseries.append("treasury.usd.cash_position", now, float(value))
            runtime.timeseries.append("ops.complaints.daily", now, float(rng.poisson(90 + day)))
        for i in range(5):
            ref = EntityRef(entity_type="customer", entity_id=f"cust_{1000 + i}")
            runtime.graph.upsert_entity(ref, {"segment": "corporate", "seed": seed})
    audit.append("twin_core", "runtime.built", {"seed": seed, "with_demo_data": with_demo_data})
    return runtime


# ---------------------------------------------------------------------------
# Entry point: fintwinos.twin_sim.register_all
# ---------------------------------------------------------------------------


class _StandinSimulator:
    """Shared scaffolding: a named simulator returning ``SimulationResult``s."""

    name = "standin"

    def calibration_report(self) -> dict[str, Any]:
        return {"simulator": self.name, "status": "standin", "backtests": 0}


class LiquidityStandinSimulator(_StandinSimulator):
    name = "liquidity_stress"

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        payload = _fallbacks.liquidity_stress_model(
            severity=float(scenario.params.get("severity", 0.5)),
            seed=seed if seed is not None else 7,
            scenario_name=scenario.name,
        )
        return SimulationResult(**payload)


class ComplianceStandinSimulator(_StandinSimulator):
    name = "compliance_alerts"

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        n_alerts = int(scenario.params.get("n_alerts", 48))
        rng = np.random.default_rng(seed if seed is not None else 7)
        _, _, labels = _fallbacks.make_alert_population(n_alerts, 0.25, rng)
        return SimulationResult(
            simulator=self.name,
            scenario_name=scenario.name,
            metrics={
                "n_alerts": float(n_alerts),
                "ring_alerts": float(int(labels.sum())),
                "ring_rate": round(float(labels.mean()), 4),
            },
            confidence={"ring_rate": _fallbacks._binomial_ci(float(labels.mean()), n_alerts)},
            calibration={"method": "synthetic_population", "n_alerts": n_alerts},
            seed=seed,
        )


class OpsQueueStandinSimulator(_StandinSimulator):
    name = "ops_queue"

    def run(self, scenario: Scenario, *, seed: int | None = None) -> SimulationResult:
        payload = _fallbacks.queue_simulation(
            staff=int(scenario.params.get("staff", 8)),
            arrival_multiplier=float(scenario.params.get("arrival_multiplier", 1.6)),
            horizon_days=int(scenario.params.get("horizon_days", 10)),
            sla_hours=float(scenario.params.get("sla_hours", 24.0)),
            seed=seed if seed is not None else 7,
        )
        return SimulationResult(**payload)


def register_all(runtime: TwinRuntime) -> None:
    """Register the three stand-in simulators on the runtime."""
    for simulator in (
        LiquidityStandinSimulator(),
        ComplianceStandinSimulator(),
        OpsQueueStandinSimulator(),
    ):
        runtime.register_simulator(simulator)


# ---------------------------------------------------------------------------
# Entry point: fintwinos.tools.catalog.build_default_registry
# ---------------------------------------------------------------------------


def _obj_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def build_default_registry(
    runtime: TwinRuntime,
    policy_gate: Any | None = None,
    settings: Settings | None = None,
) -> ToolRegistry:
    """Build a registry carrying one tool per band used by the demos."""
    settings = settings or get_settings()
    registry = ToolRegistry(
        policy_gate=policy_gate or PolicyGate(), audit=runtime.audit, settings=settings
    )
    seed_default = int(runtime.metadata.get("seed", 7))

    # -- observe ---------------------------------------------------------------
    def observe_cash_ladder(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        return {
            "ladder": _fallbacks.cash_ladder(
                seed=int(args.get("seed", seed_default)),
                horizon_days=int(args.get("horizon_days", 14)),
                currency=str(args.get("currency", "USD")),
            )
        }

    registry.register(
        ToolSpec(
            name="observe_cash_ladder",
            description="Daily cash ladder for one currency from the treasury twin.",
            input_schema=_obj_schema(
                {
                    "currency": {"type": "string"},
                    "horizon_days": {"type": "integer", "minimum": 1},
                    "seed": {"type": "integer"},
                }
            ),
            band=ToolBand.observe,
            side_effect=SideEffectClass.read,
            provenance_required=True,
        ),
        observe_cash_ladder,
    )

    def observe_filing_search(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        hits = runtime.documents.search(str(args.get("query", "")), k=int(args.get("k", 5)))
        return {"hits": [{"doc_id": doc_id, "score": score} for doc_id, score in hits]}

    registry.register(
        ToolSpec(
            name="observe_filing_search",
            description="Relevance search over filings in the twin's document store.",
            input_schema=_obj_schema(
                {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1}},
                required=["query"],
            ),
            band=ToolBand.observe,
            side_effect=SideEffectClass.read,
            provenance_required=True,
        ),
        observe_filing_search,
    )

    # -- simulate ----------------------------------------------------------------
    def simulate_liquidity_stress(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        scenario = Scenario(
            name=str(args.get("scenario", "liquidity_stress")),
            params={"severity": float(args.get("severity", 0.5))},
        )
        sim = runtime.simulators["liquidity_stress"].run(
            scenario, seed=int(args.get("seed", seed_default))
        )
        return sim.model_dump()

    registry.register(
        ToolSpec(
            name="simulate_liquidity_stress",
            description="Monte Carlo liquidity stress with survival-day confidence intervals.",
            input_schema=_obj_schema(
                {
                    "scenario": {"type": "string"},
                    "severity": {"type": "number", "minimum": 0, "maximum": 1},
                    "horizon_days": {"type": "integer"},
                    "currency": {"type": "string"},
                    "seed": {"type": "integer"},
                }
            ),
            band=ToolBand.simulate,
            risk_tier=RiskTier.medium,
        ),
        simulate_liquidity_stress,
    )

    def simulate_aml_ring(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        seed = int(args.get("seed", seed_default))
        n_alerts = int(args.get("n_alerts", 48))
        scenario = Scenario(name=str(args.get("scenario", "ring_surge")), params={"n_alerts": n_alerts})
        sim = runtime.simulators["compliance_alerts"].run(scenario, seed=seed)
        alerts, _, _ = _fallbacks.make_alert_population(
            n_alerts, 0.25, np.random.default_rng(seed)
        )
        return {**sim.model_dump(), "alerts": alerts}

    registry.register(
        ToolSpec(
            name="simulate_aml_ring",
            description="Simulate a laundering-ring alert surge; returns a labelled alert queue.",
            input_schema=_obj_schema(
                {
                    "scenario": {"type": "string"},
                    "n_alerts": {"type": "integer", "minimum": 1},
                    "horizon_days": {"type": "integer"},
                    "seed": {"type": "integer"},
                }
            ),
            band=ToolBand.simulate,
            risk_tier=RiskTier.medium,
        ),
        simulate_aml_ring,
    )

    def simulate_ops_queue(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        scenario = Scenario(
            name=f"staff_{args.get('staff', 8)}",
            params={
                "staff": int(args.get("staff", 8)),
                "arrival_multiplier": float(args.get("arrival_multiplier", 1.6)),
                "horizon_days": int(args.get("horizon_days", 10)),
                "sla_hours": float(args.get("sla_hours", 24.0)),
            },
        )
        sim = runtime.simulators["ops_queue"].run(scenario, seed=int(args.get("seed", seed_default)))
        return sim.model_dump()

    registry.register(
        ToolSpec(
            name="simulate_ops_queue",
            description="Fluid-queue Monte Carlo of a service queue under staffing scenarios.",
            input_schema=_obj_schema(
                {
                    "staff": {"type": "integer", "minimum": 1},
                    "arrival_multiplier": {"type": "number", "minimum": 0},
                    "horizon_days": {"type": "integer", "minimum": 1},
                    "sla_hours": {"type": "number", "minimum": 1},
                    "seed": {"type": "integer"},
                }
            ),
            band=ToolBand.simulate,
            risk_tier=RiskTier.low,
        ),
        simulate_ops_queue,
    )

    # -- propose ----------------------------------------------------------------
    def propose_case_narrative(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        summary = str(args.get("summary", "")).strip()
        narrative = (
            f"Case {args.get('case_id', '?')} / alert {args.get('alert_id', '?')}: pattern "
            f"consistent with layering ({summary or 'see attached features'}). Counterparty "
            "fan-in concentrated within 72 hours and transfer sizes cluster below the "
            "reporting threshold. Recommend escalation to a case, payment restraint pending "
            "review, and SAR preparation for MLRO sign-off."
        )
        return {"narrative": narrative, "case_id": args.get("case_id")}

    registry.register(
        ToolSpec(
            name="propose_case_narrative",
            description="Draft a case narrative for an alert; propose-only, side-effect free.",
            input_schema=_obj_schema(
                {
                    "case_id": {"type": "string"},
                    "alert_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "summary": {"type": "string"},
                },
                required=["case_id"],
            ),
            band=ToolBand.propose,
            risk_tier=RiskTier.medium,
        ),
        propose_case_narrative,
    )

    def propose_routing_change(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        return {"proposal": dict(args), "status": "drafted", "requires": "operations sign-off"}

    registry.register(
        ToolSpec(
            name="propose_routing_change",
            description="Draft a queue routing/staffing change; propose-only.",
            input_schema=_obj_schema(
                {
                    "queue": {"type": "string"},
                    "add_staff": {"type": "integer"},
                    "reroute_overflow_to": {"type": "string"},
                    "effective": {"type": "string"},
                    "sla_hours": {"type": "number"},
                }
            ),
            band=ToolBand.propose,
            risk_tier=RiskTier.medium,
        ),
        propose_routing_change,
    )

    # -- execute ------------------------------------------------------------------
    def execute_close_case(args: dict[str, Any], context: CallContext) -> dict[str, Any]:
        outbox = Path(settings.ensure_data_dir()) / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        path = outbox / "case_closures.jsonl"
        record = {
            "case_id": args.get("case_id"),
            "resolution": args.get("resolution"),
            "closed_by": context.caller,
            "approval_token": context.approval.token_id if context.approval else None,
            "ts": utcnow().isoformat(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return {"status": "closed", "outbox_path": str(path), "case_id": record["case_id"]}

    registry.register(
        ToolSpec(
            name="execute_close_case",
            description="Close an AML case; writes a closure record to the local outbox.",
            input_schema=_obj_schema(
                {
                    "case_id": {"type": "string"},
                    "resolution": {"type": "string"},
                    "narrative": {"type": "string"},
                },
                required=["case_id", "resolution"],
            ),
            band=ToolBand.execute,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
            provenance_required=True,
            idempotent=False,
        ),
        execute_close_case,
    )

    runtime.audit.append("tools.catalog", "registry.built", {"tools": len(registry)})
    return registry


# ---------------------------------------------------------------------------
# Entry point: fintwinos.agents.runtime.handle_case
# ---------------------------------------------------------------------------


async def handle_case(
    case_id: str,
    objective: str,
    runtime: TwinRuntime,
    registry: ToolRegistry,
    llm: Any,
) -> dict[str, Any]:
    """Plan → simulate → decide → policy-check, all through the registry."""
    from fintwinos.demos.stack import schema_args

    planned: list[dict[str, Any]] = []
    risk = RiskTier.medium
    if "liquidity" in objective.lower() and "simulate_liquidity_stress" in registry:
        risk = RiskTier.high
        spec = registry.spec("simulate_liquidity_stress")
        arguments = schema_args(
            spec,
            {
                "scenario": "severe_squeeze",
                "stress_name": "severe_squeeze",
                "severity": 0.75,
                "shock_bps": 300.0,
                "horizon_days": 30,
                "currencies": ["USD"],
                "assumptions_version": "v1",
                "ticket_id": case_id,
                "seed": int(runtime.metadata.get("seed", 7)),
            },
        )
        result = await registry.call(
            "simulate_liquidity_stress",
            arguments,
            CallContext(caller="agents.runtime", ticket_id=case_id),
        )
        if result.ok:
            planned.append(
                {"tool": "propose_funding_plan", "arguments": {"currency": "USD", "basis": result.tool}}
            )

    decision = Decision(
        objective=objective,
        action_type="propose_only",
        planned_tool_calls=planned,
        rationale="Standin planner: rehearse in the twin, propose to humans, execute nothing.",
        risk_tier=risk,
        owner="agents.runtime.standin",
        ticket_id=case_id,
    )
    gate = registry.policy_gate or PolicyGate()
    verdict = gate.check_decision(decision)
    runtime.audit.append(
        "agents.runtime",
        "case.handled",
        {
            "case_id": case_id,
            "decision_id": decision.decision_id,
            "allowed": verdict.allowed,
            "requires_human_review": verdict.requires_human_review,
        },
    )
    status = "awaiting_human" if verdict.requires_human_review else "complete"
    if not verdict.allowed:
        status = "blocked"
    return {
        "status": status,
        "decision": decision.model_dump(mode="json"),
        "policy": verdict.model_dump(),
        "case_id": case_id,
    }
