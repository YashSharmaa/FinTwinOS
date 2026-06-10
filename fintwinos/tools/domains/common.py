"""Shared plumbing for the FinTwinOS domain tool packs.

Every domain module (`risk`, `treasury`, `compliance`, `customer_ops`, `filings`)
builds its :class:`~fintwinos.tools.registry.ToolSpec` schemas and handlers on top of
the helpers here, so the conventions stay uniform:

- JSON Schemas are always closed objects (``additionalProperties: false``) with an
  explicit ``required`` list, built via :func:`object_schema`.
- Twin state is reached **only** through the :class:`~fintwinos.core.interfaces.TwinRuntime`
  stores. Because the :class:`~fintwinos.core.interfaces.GraphStore` Protocol has no
  enumeration method, :func:`entities_of_type` probes a small set of widely used
  capability shapes (an ``entities_of_type``-style method, a ``networkx`` graph whose
  nodes are keyed ``"<type>:<id>"``, or a plain entity dict) and degrades to an empty
  list rather than failing.
- Simulations run through :func:`run_simulator` with seeds derived deterministically by
  :func:`derive_seed`, so identical inputs always reproduce identical results.
- Execute-band side effects only ever land in the local outbox directory
  (``settings.data_dir / "outbox"``) via :func:`write_outbox_json` /
  :func:`append_outbox_jsonl` — never in a real external system.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

from fintwinos.core.config import Settings
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.core.types import EntityRef, Scenario, SimulationResult, utcnow

# ---------------------------------------------------------------------------
# JSON Schema construction
# ---------------------------------------------------------------------------


def object_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
    *,
    description: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a closed JSON-Schema object: explicit ``required``, no extra properties.

    Args:
        properties: Property name to schema fragment mapping.
        required: Names that must be present; defaults to an empty list.
        description: Optional human-readable schema description.
        extra: Additional top-level schema keywords (e.g. ``anyOf``).

    Returns:
        A Draft 2020-12 compatible object schema.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    if extra:
        schema.update(extra)
    return schema


# ---------------------------------------------------------------------------
# Graph access
# ---------------------------------------------------------------------------


def _merge_node_attributes(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten a node-data dict that may nest its payload under ``attributes``."""
    merged = dict(data or {})
    nested = merged.pop("attributes", None)
    if isinstance(nested, dict):
        merged.update(nested)
    return merged


def _normalise_entity_items(
    graph: Any, entity_type: str, items: Any
) -> list[tuple[EntityRef, dict[str, Any]]]:
    """Coerce assorted enumeration result shapes into ``(EntityRef, attrs)`` pairs."""
    out: list[tuple[EntityRef, dict[str, Any]]] = []
    for item in items or []:
        if isinstance(item, tuple | list) and len(item) == 2:
            ref_like, attrs = item
            if isinstance(ref_like, EntityRef):
                out.append((ref_like, dict(attrs or {})))
                continue
            if isinstance(ref_like, str):
                ref = EntityRef(entity_type=entity_type, entity_id=ref_like)
                out.append((ref, dict(attrs or {})))
                continue
        if isinstance(item, EntityRef):
            out.append((item, dict(graph.get_entity(item) or {})))
            continue
        if isinstance(item, dict):
            entity_id = item.get("entity_id") or item.get("id")
            if entity_id is not None:
                ref = EntityRef(entity_type=entity_type, entity_id=str(entity_id))
                attrs = {k: v for k, v in item.items() if k not in {"entity_id", "id"}}
                out.append((ref, _merge_node_attributes(attrs)))
    return sorted(out, key=lambda pair: pair[0].entity_id)


def entities_of_type(graph: Any, entity_type: str) -> list[tuple[EntityRef, dict[str, Any]]]:
    """Enumerate all entities of one type from a ``GraphStore``-like object.

    The core ``GraphStore`` Protocol deliberately omits a listing method, so this
    helper probes — in order — a listing method (``entities_of_type``,
    ``entities_by_type``, ``list_entities``, ``entities``), a ``networkx`` graph
    attribute whose node keys follow the canonical ``"<entity_type>:<entity_id>"``
    convention, and finally a plain internal entity dict. Stores supporting none of
    these yield an empty list, never an exception.

    Args:
        graph: Any object satisfying (at least) the ``GraphStore`` Protocol.
        entity_type: Canonical entity type, e.g. ``"position"`` or ``"alert"``.

    Returns:
        ``(EntityRef, attributes)`` pairs sorted by ``entity_id`` for determinism.
    """
    for method_name in ("entities_of_type", "entities_by_type", "list_entities", "entities"):
        method = getattr(graph, method_name, None)
        if callable(method):
            try:
                return _normalise_entity_items(graph, entity_type, method(entity_type))
            except Exception:  # noqa: BLE001 — fall through to the next capability probe
                continue

    prefix = f"{entity_type}:"
    for attr_name in ("g", "graph", "nx_graph", "_graph", "_g"):
        candidate = getattr(graph, attr_name, None)
        if isinstance(candidate, nx.Graph):
            out = []
            for node, data in candidate.nodes(data=True):
                if isinstance(node, str) and node.startswith(prefix):
                    ref = EntityRef(entity_type=entity_type, entity_id=node[len(prefix):])
                    out.append((ref, _merge_node_attributes(data or {})))
            return sorted(out, key=lambda pair: pair[0].entity_id)

    for attr_name in ("_entities", "_nodes"):
        candidate = getattr(graph, attr_name, None)
        if isinstance(candidate, dict):
            out = []
            for key, attrs in candidate.items():
                if isinstance(key, EntityRef) and key.entity_type == entity_type:
                    out.append((key, _merge_node_attributes(dict(attrs or {}))))
                elif isinstance(key, str) and key.startswith(prefix):
                    ref = EntityRef(entity_type=entity_type, entity_id=key[len(prefix):])
                    out.append((ref, _merge_node_attributes(dict(attrs or {}))))
            return sorted(out, key=lambda pair: pair[0].entity_id)

    return []


def get_entity_attributes(graph: Any, entity_type: str, entity_id: str) -> dict[str, Any] | None:
    """Fetch one entity's attributes, or ``None`` when it does not exist."""
    attrs = graph.get_entity(EntityRef(entity_type=entity_type, entity_id=entity_id))
    if attrs is None:
        return None
    return _merge_node_attributes(dict(attrs))


# ---------------------------------------------------------------------------
# Determinism and simulation plumbing
# ---------------------------------------------------------------------------


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a reproducible sub-seed from the platform seed plus scenario inputs.

    Identical ``(base_seed, parts)`` always produce the same seed, while distinct
    scenarios get well-separated streams — keeping every simulation reproducible
    without callers having to thread explicit seeds through tool arguments.
    """
    body = json.dumps([base_seed, *[str(p) for p in parts]], sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def run_simulator(
    runtime: TwinRuntime, name: str, scenario: Scenario, seed: int
) -> SimulationResult:
    """Run a named simulator from the runtime, with a clear error when absent."""
    simulator = runtime.simulators.get(name)
    if simulator is None:
        available = ", ".join(sorted(runtime.simulators)) or "none"
        raise ValueError(
            f"simulator '{name}' is not registered on this runtime (available: {available}); "
            "call fintwinos.twin_sim.register_all(runtime) before using simulate_* tools"
        )
    return simulator.run(scenario, seed=seed)


def simulation_payload(
    result: SimulationResult, scenario: Scenario, source_system: str
) -> dict[str, Any]:
    """Standard tool-data envelope for a simulation: result, uncertainty, provenance."""
    return {
        "scenario": scenario.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "metrics": dict(result.metrics),
        "confidence": {k: list(v) for k, v in result.confidence.items()},
        "calibration": json_safe(result.calibration),
        "warnings": list(result.warnings),
        "assumptions_version": result.assumptions_version,
        "seed": result.seed,
        "provenance": [provenance_entry(source_system, f"scenario:{scenario.name}")],
    }


# ---------------------------------------------------------------------------
# Output hygiene
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Round-trip a value through JSON so tool outputs never carry live objects."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def provenance_entry(source_system: str, notes: str) -> dict[str, str]:
    """Provenance stub the registry inflates into a full ``Provenance`` record."""
    return {"source_system": source_system, "notes": notes}


# ---------------------------------------------------------------------------
# Outbox (the ONLY place execute-band tools may write)
# ---------------------------------------------------------------------------


def outbox_dir(settings: Settings) -> Path:
    """Ensure and return the local outbox directory under ``settings.data_dir``."""
    path = settings.ensure_data_dir() / "outbox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_outbox_json(settings: Settings, filename: str, record: dict[str, Any]) -> Path:
    """Write one instruction document into the outbox as pretty-printed JSON."""
    path = outbox_dir(settings) / filename
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def append_outbox_jsonl(settings: Settings, filename: str, record: dict[str, Any]) -> Path:
    """Append one record to a JSONL ledger file in the outbox."""
    path = outbox_dir(settings) / filename
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return path


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (tool outputs stay JSON-native)."""
    return utcnow().isoformat()
