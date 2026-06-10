"""The default FinTwinOS tool catalog.

:func:`build_default_registry` is the contracted entry point (see ``CONTRACTS.md``):
it assembles a :class:`~fintwinos.tools.registry.ToolRegistry` over a
:class:`~fintwinos.core.interfaces.TwinRuntime`, wires the shared audit trail and
policy gate, and registers every domain tool pack from
:mod:`fintwinos.tools.domains` — yielding 18+ tools across all four governance bands
(observe / simulate / propose / execute).
"""

from __future__ import annotations

from typing import Any

from fintwinos.core.config import Settings, get_settings
from fintwinos.core.interfaces import TwinRuntime
from fintwinos.policy.gates import PolicyGate
from fintwinos.tools.domains import ALL_DOMAINS
from fintwinos.tools.registry import ToolRegistry


def build_default_registry(
    runtime: TwinRuntime,
    policy_gate: Any | None = None,
    settings: Settings | None = None,
) -> ToolRegistry:
    """Build the default typed tool registry over a twin runtime.

    Args:
        runtime: The assembled federated twin (graph, timeseries, documents, replay,
            audit and simulators). The registry shares ``runtime.audit`` so every tool
            call lands on the same hash-chained trail as the rest of the platform.
        policy_gate: Optional policy gate; defaults to the shipped conservative
            :class:`~fintwinos.policy.gates.PolicyGate` (execute band default-deny).
        settings: Optional settings; defaults to the process-wide
            :func:`~fintwinos.core.config.get_settings`.

    Returns:
        A :class:`~fintwinos.tools.registry.ToolRegistry` with every domain tool pack
        registered (risk, treasury, compliance, customer_ops, filings).
    """
    effective_settings = settings if settings is not None else get_settings()
    gate = policy_gate if policy_gate is not None else PolicyGate()
    registry = ToolRegistry(
        policy_gate=gate, audit=runtime.audit, settings=effective_settings
    )
    # ``ToolRegistry.__init__`` uses ``audit or AuditTrail()``, and an *empty*
    # AuditTrail is falsy (it defines __len__) — assign explicitly so the registry
    # always shares the runtime's hash chain, even when it has no records yet.
    registry.audit = runtime.audit
    for domain in ALL_DOMAINS:
        domain.register(registry, runtime)

    band_counts: dict[str, int] = {}
    for spec in registry.list_specs():
        band_counts[spec.band.value] = band_counts.get(spec.band.value, 0) + 1
    registry.audit.append(
        "tools.catalog",
        "catalog.built",
        {
            "tool_count": len(registry),
            "bands": band_counts,
            "domains": [domain.__name__.rsplit(".", 1)[-1] for domain in ALL_DOMAINS],
        },
    )
    return registry


def catalog_summary(registry: ToolRegistry) -> dict[str, Any]:
    """Summarise a registry for dashboards and demos: counts by band and by owner."""
    by_band: dict[str, list[str]] = {}
    by_owner: dict[str, list[str]] = {}
    for spec in registry.list_specs():
        by_band.setdefault(spec.band.value, []).append(spec.name)
        by_owner.setdefault(spec.owner, []).append(spec.name)
    return {
        "tool_count": len(registry),
        "by_band": {band: sorted(names) for band, names in sorted(by_band.items())},
        "by_owner": {owner: sorted(names) for owner, names in sorted(by_owner.items())},
    }
