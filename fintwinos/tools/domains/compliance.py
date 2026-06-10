"""Financial-crime & compliance tool pack.

Tools registered here:

- ``observe_alerts`` — filterable view over transaction-monitoring alerts.
- ``observe_case`` — one case with its linked entities and related alerts.
- ``observe_entity_graph`` — egonet around an entity for network investigation.
- ``simulate_alert_threshold`` — runs the ``compliance`` simulator for a candidate
  threshold and surfaces the precision/recall trade-off with confidence intervals.
- ``propose_case_narrative`` — deterministic, template-based case narrative built from
  case, alert and graph context (no LLM required; offline-safe by construction).
- ``execute_close_case`` — appends a closure record to the local outbox ledger and
  marks the case closed in the twin; reversible, human approval always required.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import (
    RISK_ORDER,
    EntityRef,
    RiskTier,
    Scenario,
    SideEffectClass,
    ToolBand,
)
from fintwinos.tools.domains.common import (
    append_outbox_jsonl,
    derive_seed,
    entities_of_type,
    get_entity_attributes,
    json_safe,
    now_iso,
    object_schema,
    provenance_entry,
    run_simulator,
    simulation_payload,
)
from fintwinos.tools.registry import CallContext, ToolRegistry, ToolSpec

OWNER = "compliance"

_SEVERITY_ORDER = {tier.value: rank for tier, rank in RISK_ORDER.items()}


def _entity_keys(entities: Any) -> set[str]:
    """Normalise a list of entity-ref dicts into ``"type:id"`` keys."""
    keys: set[str] = set()
    for entry in entities or []:
        if isinstance(entry, dict) and entry.get("entity_type") and entry.get("entity_id"):
            keys.add(f"{entry['entity_type']}:{entry['entity_id']}")
        elif isinstance(entry, EntityRef):
            keys.add(entry.key())
    return keys


def _load_alerts(runtime: TwinRuntime) -> list[dict[str, Any]]:
    """Load alert entities sorted by score (desc) then alert id."""
    rows: list[dict[str, Any]] = []
    for ref, attrs in entities_of_type(runtime.graph, "alert"):
        rows.append(
            {
                "alert_id": ref.entity_id,
                "kind": attrs.get("kind", "unknown"),
                "severity": str(attrs.get("severity", "medium")),
                "score": float(attrs.get("score", 0.0)),
                "status": str(attrs.get("status", "new")),
                "created_at": str(attrs.get("created_at")) if attrs.get("created_at") else None,
                "entities": json_safe(attrs.get("entities") or []),
            }
        )
    rows.sort(key=lambda r: (-r["score"], r["alert_id"]))
    return rows


def _alerts_touching(runtime: TwinRuntime, keys: set[str]) -> list[dict[str, Any]]:
    """Alerts whose linked entities overlap the given entity keys."""
    return [a for a in _load_alerts(runtime) if _entity_keys(a["entities"]) & keys]


def _parse_node_key(node: Any) -> tuple[str, str]:
    """Split a graph node key into ``(entity_type, entity_id)``."""
    if isinstance(node, EntityRef):
        return node.entity_type, node.entity_id
    text = str(node)
    if ":" in text:
        entity_type, entity_id = text.split(":", 1)
        return entity_type, entity_id
    return "unknown", text


def _egonet(
    runtime: TwinRuntime, centre: EntityRef, depth: int, kind: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect the egonet around ``centre`` as JSON-safe node and edge lists.

    Prefers ``graph.subgraph`` when it yields a ``networkx`` graph (full edge data);
    otherwise falls back to breadth-first expansion through ``graph.neighbors`` —
    which stays strictly inside the ``GraphStore`` Protocol.
    """
    node_keys: set[str] = {centre.key()}
    edges: set[tuple[str, str, str]] = set()

    subgraph: Any = None
    if kind is None:  # subgraph cannot filter by relationship kind
        try:
            subgraph = runtime.graph.subgraph([centre], depth=depth)
        except Exception:  # noqa: BLE001 — optional fast path only
            subgraph = None

    if isinstance(subgraph, nx.Graph):
        for node in subgraph.nodes:
            entity_type, entity_id = _parse_node_key(node)
            node_keys.add(f"{entity_type}:{entity_id}")
        edge_iter = (
            subgraph.edges(data=True, keys=False)
            if subgraph.is_multigraph()
            else subgraph.edges(data=True)
        )
        for src, dst, data in edge_iter:
            src_key = ":".join(_parse_node_key(src))
            dst_key = ":".join(_parse_node_key(dst))
            edges.add((src_key, dst_key, str((data or {}).get("kind", ""))))
    else:
        frontier = [centre]
        for _ in range(depth):
            next_frontier: list[EntityRef] = []
            for ref in frontier:
                for neighbour in runtime.graph.neighbors(ref, kind=kind, depth=1):
                    edges.add((ref.key(), neighbour.key(), kind or ""))
                    if neighbour.key() not in node_keys:
                        node_keys.add(neighbour.key())
                        next_frontier.append(neighbour)
            frontier = next_frontier

    nodes: list[dict[str, Any]] = []
    for key in sorted(node_keys):
        entity_type, entity_id = key.split(":", 1)
        attrs = get_entity_attributes(runtime.graph, entity_type, entity_id) or {}
        nodes.append(
            {
                "key": key,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "attributes": json_safe(attrs),
            }
        )
    edge_rows = [
        {"src": src, "dst": dst, "kind": edge_kind or None}
        for src, dst, edge_kind in sorted(edges)
    ]
    return nodes, edge_rows


def _signal_strength(alerts: list[dict[str, Any]]) -> str:
    """Deterministic overall signal strength from the strongest alert score."""
    top = max((a["score"] for a in alerts), default=0.0)
    if top >= 0.8:
        return "high"
    if top >= 0.5:
        return "moderate"
    return "low"


_RECOMMENDATIONS = {
    "high": (
        "Escalate for enhanced due diligence and prepare a draft regulatory filing; "
        "route to a senior investigator for human review before any submission."
    ),
    "moderate": (
        "Continue the investigation: request refreshed KYC information and 90 days of "
        "transaction history before a disposition is proposed."
    ),
    "low": (
        "Propose closure as a false positive, subject to human approval via "
        "execute_close_case with a documented closure note."
    ),
}


def register(registry: ToolRegistry, runtime: TwinRuntime) -> None:
    """Register the compliance tool pack on the given registry against the runtime."""

    # -- observe_alerts ----------------------------------------------------------

    def observe_alerts(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        rows = _load_alerts(runtime)
        if arguments.get("status"):
            rows = [r for r in rows if r["status"] == arguments["status"]]
        if arguments.get("kind"):
            rows = [r for r in rows if r["kind"] == arguments["kind"]]
        if arguments.get("min_severity"):
            floor = _SEVERITY_ORDER.get(arguments["min_severity"], 0)
            rows = [r for r in rows if _SEVERITY_ORDER.get(r["severity"], 0) >= floor]
        limit = int(arguments.get("limit", 50))
        return {
            "count": len(rows),
            "alerts": rows[:limit],
            "provenance": [provenance_entry("twin_core.graph", "entity_type:alert")],
        }

    registry.register(
        ToolSpec(
            name="observe_alerts",
            description=(
                "List transaction-monitoring alerts from the twin, sorted by score, with "
                "optional filters on status, kind and minimum severity."
            ),
            input_schema=object_schema(
                {
                    "status": {
                        "type": "string",
                        "enum": ["new", "triaged", "investigating", "dismissed", "confirmed"],
                        "description": "Only alerts in this workflow status.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Only alerts of this typology, e.g. 'structuring'.",
                    },
                    "min_severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Only alerts at or above this severity.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum alerts to return (default 50).",
                    },
                }
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_alerts,
    )

    # -- observe_case -------------------------------------------------------------

    def observe_case(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        case_id = arguments["case_id"]
        attrs = get_entity_attributes(runtime.graph, "case", case_id)
        if attrs is None:
            raise ValueError(f"case '{case_id}' not found in the twin")
        case_ref = EntityRef(entity_type="case", entity_id=case_id)
        linked = []
        for neighbour in runtime.graph.neighbors(case_ref, depth=1):
            linked.append(
                {
                    "key": neighbour.key(),
                    "entity_type": neighbour.entity_type,
                    "entity_id": neighbour.entity_id,
                    "attributes": json_safe(
                        get_entity_attributes(
                            runtime.graph, neighbour.entity_type, neighbour.entity_id
                        )
                        or {}
                    ),
                }
            )
        linked.sort(key=lambda n: n["key"])
        keys = _entity_keys(attrs.get("entities")) | {n["key"] for n in linked}
        related_alerts = _alerts_touching(runtime, keys)
        return {
            "case_id": case_id,
            "case": json_safe(attrs),
            "linked_entities": linked,
            "related_alerts": related_alerts,
            "provenance": [provenance_entry("twin_core.graph", f"case:{case_id}")],
        }

    registry.register(
        ToolSpec(
            name="observe_case",
            description=(
                "Fetch one case with its full attributes, the entities linked to it in the "
                "twin graph, and every alert touching those entities."
            ),
            input_schema=object_schema(
                {
                    "case_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Identifier of the case to fetch.",
                    }
                },
                ["case_id"],
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_case,
    )

    # -- observe_entity_graph --------------------------------------------------------

    def observe_entity_graph(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        entity_type = arguments["entity_type"]
        entity_id = arguments["entity_id"]
        depth = int(arguments.get("depth", 1))
        kind = arguments.get("kind")
        centre = EntityRef(entity_type=entity_type, entity_id=entity_id)
        if runtime.graph.get_entity(centre) is None:
            raise ValueError(f"entity '{centre.key()}' not found in the twin")
        nodes, edges = _egonet(runtime, centre, depth, kind)
        return {
            "centre": centre.key(),
            "depth": depth,
            "relationship_kind": kind,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
            "provenance": [
                provenance_entry("twin_core.graph", f"egonet:{centre.key()}:depth={depth}")
            ],
        }

    registry.register(
        ToolSpec(
            name="observe_entity_graph",
            description=(
                "Return the egonet (neighbourhood subgraph) around an entity up to a given "
                "depth — nodes with attributes plus typed edges — for network investigation."
            ),
            input_schema=object_schema(
                {
                    "entity_type": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Canonical entity type, e.g. 'customer'.",
                    },
                    "entity_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Identifier of the centre entity.",
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "description": "Hops to expand from the centre (default 1).",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Only traverse relationships of this kind.",
                    },
                },
                ["entity_type", "entity_id"],
            ),
            band=ToolBand.observe,
            risk_tier=RiskTier.low,
            side_effect=SideEffectClass.read,
            owner=OWNER,
        ),
        observe_entity_graph,
    )

    # -- simulate_alert_threshold -------------------------------------------------------

    def simulate_alert_threshold(
        arguments: dict[str, Any], context: CallContext
    ) -> dict[str, Any]:
        threshold = float(arguments["threshold"])
        alert_kind = arguments.get("alert_kind")
        scenario = Scenario(
            name=f"alert_threshold_{threshold:g}",
            kind="what_if",
            params={
                "threshold": threshold,
                "alert_kind": alert_kind,
                "lookback_days": int(arguments.get("lookback_days", 90)),
            },
        )
        seed = arguments.get("seed")
        if seed is None:
            seed = derive_seed(
                registry.settings.seed, "compliance", threshold, alert_kind or "all"
            )
        result = run_simulator(runtime, "compliance", scenario, int(seed))
        payload = simulation_payload(result, scenario, "twin_sim.compliance")
        precision = result.metrics.get("precision")
        recall = result.metrics.get("recall")
        f1 = result.metrics.get("f1")
        if f1 is None and precision and recall and (precision + recall) > 0:
            f1 = round(2 * precision * recall / (precision + recall), 6)
        payload["trade_off"] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_ci": result.confidence.get("precision"),
            "recall_ci": result.confidence.get("recall"),
        }
        return payload

    registry.register(
        ToolSpec(
            name="simulate_alert_threshold",
            description=(
                "Run the compliance simulator for a candidate alerting threshold and return "
                "the precision/recall trade-off with confidence intervals and calibration. "
                "Deterministic for identical inputs."
            ),
            input_schema=object_schema(
                {
                    "threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Candidate alert score threshold in [0, 1].",
                    },
                    "alert_kind": {
                        "type": "string",
                        "description": "Restrict the backtest to one alert typology.",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 365,
                        "description": "Backtest window in days (default 90).",
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Override the deterministic derived seed.",
                    },
                },
                ["threshold"],
            ),
            band=ToolBand.simulate,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.none,
            provenance_required=True,
            owner=OWNER,
        ),
        simulate_alert_threshold,
    )

    # -- propose_case_narrative -----------------------------------------------------------

    def propose_case_narrative(
        arguments: dict[str, Any], context: CallContext
    ) -> dict[str, Any]:
        case_id = arguments["case_id"]
        include_graph = bool(arguments.get("include_graph", True))
        max_related = int(arguments.get("max_related_entities", 10))

        attrs = get_entity_attributes(runtime.graph, "case", case_id)
        if attrs is None:
            raise ValueError(f"case '{case_id}' not found in the twin")
        case_ref = EntityRef(entity_type="case", entity_id=case_id)
        subjects = sorted(runtime.graph.neighbors(case_ref, depth=1), key=lambda r: r.key())
        subjects = subjects[:max_related]
        keys = _entity_keys(attrs.get("entities")) | {s.key() for s in subjects}
        alerts = _alerts_touching(runtime, keys)
        strength = _signal_strength(alerts)

        sections: dict[str, str] = {}
        sections["summary"] = (
            f"Case {case_id} ({attrs.get('kind', 'unknown')}, priority "
            f"{attrs.get('priority', 'medium')}) was opened on "
            f"{attrs.get('opened_at', 'an unknown date')} and is currently "
            f"{attrs.get('status', 'open')}. {len(alerts)} related alert(s) and "
            f"{len(subjects)} linked entit(y/ies) were reviewed for this narrative."
        )

        subject_lines = []
        for ref in subjects:
            subject_attrs = (
                get_entity_attributes(runtime.graph, ref.entity_type, ref.entity_id) or {}
            )
            described = ", ".join(
                f"{field}={subject_attrs[field]}"
                for field in ("name", "segment", "risk_rating", "country", "jurisdiction")
                if field in subject_attrs
            )
            subject_lines.append(f"- {ref.key()}" + (f": {described}" if described else ""))
        sections["subjects"] = "\n".join(subject_lines) or "- no linked entities on record"

        activity_lines = [
            f"- alert {a['alert_id']} ({a['kind']}, severity {a['severity']}, "
            f"score {a['score']:.2f}, status {a['status']})"
            for a in alerts
        ]
        sections["activity"] = "\n".join(activity_lines) or "- no related alerts on record"

        if include_graph:
            network_lines = []
            for ref in subjects:
                neighbours = sorted(
                    (n.key() for n in runtime.graph.neighbors(ref, depth=1)),
                )
                neighbours = [k for k in neighbours if k != case_ref.key()]
                preview = ", ".join(neighbours[:3])
                network_lines.append(
                    f"- {ref.key()} is connected to {len(neighbours)} counterpart(s) "
                    f"within 1 hop" + (f", including {preview}" if preview else "")
                )
            sections["network"] = "\n".join(network_lines) or "- network context not requested"
        else:
            sections["network"] = "- network context not requested"

        top_kinds = sorted({a["kind"] for a in alerts}) or ["none"]
        sections["assessment"] = (
            f"Indicators are consistent with the {', '.join(top_kinds)} typolog(y/ies); "
            f"overall signal strength is assessed as {strength} based on the maximum alert "
            f"score across the linked entity set."
        )
        sections["recommended_action"] = _RECOMMENDATIONS[strength]

        narrative = "\n\n".join(
            [
                f"== CASE NARRATIVE: {case_id} ==",
                "1. SUMMARY\n" + sections["summary"],
                "2. SUBJECTS\n" + sections["subjects"],
                "3. ACTIVITY OBSERVED\n" + sections["activity"],
                "4. NETWORK CONTEXT\n" + sections["network"],
                "5. ASSESSMENT\n" + sections["assessment"],
                "6. RECOMMENDED ACTION\n" + sections["recommended_action"],
                "Generated deterministically by FinTwinOS compliance tooling; pending "
                "human review. This draft is not a regulatory filing.",
            ]
        )
        return {
            "case_id": case_id,
            "narrative": narrative,
            "sections": sections,
            "signal_strength": strength,
            "sources": sorted(keys) + [a["alert_id"] for a in alerts],
            "generated_by": "rule_based_template",
            "provenance": [
                provenance_entry(
                    "fintwinos.tools.compliance",
                    f"deterministic narrative template over case:{case_id}",
                )
            ],
        }

    registry.register(
        ToolSpec(
            name="propose_case_narrative",
            description=(
                "Draft a structured, deterministic case narrative (summary, subjects, "
                "activity, network context, assessment, recommended action) from the case "
                "record, related alerts and graph context. Template-based — works fully "
                "offline; an LLM may optionally enrich it downstream."
            ),
            input_schema=object_schema(
                {
                    "case_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Identifier of the case to narrate.",
                    },
                    "include_graph": {
                        "type": "boolean",
                        "description": "Include the network-context section (default true).",
                    },
                    "max_related_entities": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Cap on linked entities described (default 10).",
                    },
                },
                ["case_id"],
            ),
            band=ToolBand.propose,
            risk_tier=RiskTier.medium,
            side_effect=SideEffectClass.none,
            owner=OWNER,
        ),
        propose_case_narrative,
    )

    # -- execute_close_case -----------------------------------------------------------------

    def execute_close_case(arguments: dict[str, Any], context: CallContext) -> dict[str, Any]:
        case_id = arguments["case_id"]
        attrs = get_entity_attributes(runtime.graph, "case", case_id)
        if attrs is None:
            raise ValueError(f"case '{case_id}' not found in the twin")

        approval = context.approval
        closure = {
            "record_type": "case_closure",
            "case_id": case_id,
            "disposition": arguments["disposition"],
            "closure_note": arguments["closure_note"],
            "ticket_id": arguments["ticket_id"],
            "previous_status": attrs.get("status", "open"),
            "closed_by": context.caller,
            "approved_by": approval.granted_by if approval else None,
            "approval_token_id": approval.token_id if approval else None,
            "reversible": True,
            "closed_at": now_iso(),
        }
        path = append_outbox_jsonl(registry.settings, "case_closures.jsonl", closure)

        case_ref = EntityRef(entity_type="case", entity_id=case_id)
        runtime.graph.upsert_entity(
            case_ref,
            {
                **attrs,
                "status": "closed",
                "disposition": arguments["disposition"],
                "closed_at": closure["closed_at"],
                "closure_ticket_id": arguments["ticket_id"],
            },
        )
        runtime.audit.append(
            context.caller,
            "case.closed",
            {
                "case_id": case_id,
                "disposition": arguments["disposition"],
                "ticket_id": arguments["ticket_id"],
                "outbox_path": str(path),
            },
        )
        return {
            "case_id": case_id,
            "status": "closed",
            "disposition": arguments["disposition"],
            "outbox_path": str(path),
            "provenance": [provenance_entry("fintwinos.outbox", f"case_closure:{case_id}")],
        }

    registry.register(
        ToolSpec(
            name="execute_close_case",
            description=(
                "Close a case: append a closure record to the local outbox ledger and mark "
                "the case closed in the twin. Reversible (cases can be reopened); always "
                "requires a human approval token, a policy allow rule and the execute-band "
                "enable flag."
            ),
            input_schema=object_schema(
                {
                    "case_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Identifier of the case to close.",
                    },
                    "disposition": {
                        "type": "string",
                        "enum": ["true_positive", "false_positive", "inconclusive", "escalated"],
                        "description": "Final disposition of the investigation.",
                    },
                    "closure_note": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 4000,
                        "description": "Mandatory free-text justification for the closure.",
                    },
                    "ticket_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Workflow ticket authorising the closure.",
                    },
                },
                ["case_id", "disposition", "closure_note", "ticket_id"],
            ),
            band=ToolBand.execute,
            risk_tier=RiskTier.high,
            side_effect=SideEffectClass.reversible,
            requires_human_approval=True,
            provenance_required=True,
            idempotent=False,
            owner=OWNER,
        ),
        execute_close_case,
    )
