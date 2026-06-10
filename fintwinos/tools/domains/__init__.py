"""Domain tool packs for the FinTwinOS typed tool catalog.

Each module in this package exposes ``register(registry, runtime)`` and contributes
a coherent set of ``observe_*`` / ``simulate_*`` / ``propose_*`` / ``execute_*`` tools
for one business domain. Handlers read and write twin state **only** through the
:class:`~fintwinos.core.interfaces.TwinRuntime` stores, and execute-band side effects
are confined to the local outbox directory.

``ALL_DOMAINS`` is the canonical registration order used by
:func:`fintwinos.tools.catalog.build_default_registry`.
"""

from fintwinos.tools.domains import compliance, customer_ops, filings, risk, treasury

#: Canonical registration order for the default catalog.
ALL_DOMAINS = (risk, treasury, compliance, customer_ops, filings)

__all__ = [
    "ALL_DOMAINS",
    "compliance",
    "customer_ops",
    "filings",
    "risk",
    "treasury",
]
