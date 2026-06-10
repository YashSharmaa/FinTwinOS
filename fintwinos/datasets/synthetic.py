"""Seeded synthetic data generators for FinTwinOS.

Every generator here is fully deterministic per seed: all randomness flows through a
single ``numpy.random.default_rng(seed)`` instance, event identifiers and timestamps
are derived arithmetically (never from wall clocks or ``uuid``), and re-running a
generator with the same arguments produces byte-identical envelopes.

Each generator returns a :class:`SyntheticDataset` containing:

- ``envelopes`` — a list of :class:`~fintwinos.core.types.EventEnvelope` ready to feed
  straight into ``runtime.ingestor`` (envelopes in, envelopes out);
- ``entities`` — the canonical entities behind the events (``Customer``, ``Account``,
  ``Instrument``, ``CaseRecord``) for seeding graph stores;
- ``labels`` — a ground-truth dictionary for supervised work and eval scoring. Labels
  are **never** leaked into envelope payloads: a detector consuming the envelopes sees
  exactly what a production system would see;
- ``graph`` — where the dataset has a natural graph shape (money flows, fraud rings),
  a ``networkx`` graph annotated with the ground truth for analysis convenience.

The generators cover the five demo domains: retail transactions with embedded
money-laundering motifs (fan-in, fan-out, cycles), layered fraud rings, multi-channel
customer journeys with complaint-escalation paths, market price paths (GBM with jumps
and GARCH-style volatility clustering), and treasury cash-flow ladders.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import networkx as nx
import numpy as np

from fintwinos.core.types import (
    Account,
    CaseRecord,
    CaseStatus,
    Customer,
    EntityRef,
    EventEnvelope,
    Instrument,
    Provenance,
    RiskTier,
)

#: Fixed epoch for all synthetic timestamps; never derived from the wall clock so that
#: identical seeds produce byte-identical envelopes across runs and machines.
BASE_TIME = datetime(2025, 1, 6, 8, 0, 0, tzinfo=UTC)

SYNTHETIC_LICENCE = "MIT (fully synthetic; no real persons, accounts or institutions)"

_FIRST_NAMES = [
    "Asha", "Bruno", "Carmen", "Deepak", "Elena", "Farid", "Greta", "Hiro",
    "Ines", "Jonas", "Kavya", "Liam", "Mona", "Nadia", "Otto", "Priya",
    "Quentin", "Rosa", "Stefan", "Tariq", "Uma", "Viktor", "Wendy", "Yara",
]
_LAST_NAMES = [
    "Abara", "Bergstrom", "Castellanos", "Duarte", "Eriksen", "Fontaine",
    "Gallagher", "Hassan", "Iwata", "Jansen", "Kowalski", "Lindqvist",
    "Marchetti", "Novak", "Okafor", "Petrov", "Quinn", "Rahman", "Silva",
    "Tanaka", "Ueda", "Varga", "Whitfield", "Zhou",
]
_COUNTRIES = ["GB", "US", "DE", "FR", "NL", "IE", "ES", "SG", "HK", "AE"]
_TXN_CHANNELS = ["wire", "ach", "card", "internal", "cash_deposit"]
_JOURNEY_CHANNELS = ["mobile_app", "web", "branch", "call_centre", "email", "chat"]
_ISSUE_TYPES = [
    "card_declined", "payment_delay", "fee_dispute", "app_outage",
    "fraud_concern", "account_lockout",
]


@dataclass
class SyntheticDataset:
    """The uniform return type of every generator in this module.

    Attributes:
        name: Generator name, e.g. ``"synthetic-transactions"``.
        seed: The seed the dataset was generated with.
        envelopes: Ingestion-ready :class:`EventEnvelope` list, sorted by
            ``occurred_at`` (ties broken by ``event_id``) so replay order is stable.
        labels: Ground-truth dictionary for supervised work; structure is documented
            per generator. Labels never appear inside envelope payloads.
        entities: Canonical entities keyed by plural type name (``"customers"``,
            ``"accounts"``, ``"instruments"``, ``"cases"``).
        graph: Optional ``networkx`` graph view of the dataset (money-flow or ring
            structure), annotated with ground-truth edge/node attributes.
        metadata: Generation parameters, for provenance and data cards.
    """

    name: str
    seed: int
    envelopes: list[EventEnvelope]
    labels: dict[str, Any]
    entities: dict[str, list[Any]] = field(default_factory=dict)
    graph: nx.MultiDiGraph | nx.DiGraph | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _provenance(generator: str, seed: int) -> Provenance:
    """Provenance block stamped on every synthetic envelope."""
    return Provenance(
        source_system=f"fintwinos.datasets.synthetic.{generator}",
        ingested_at=BASE_TIME,
        licence=SYNTHETIC_LICENCE,
        notes=f"deterministic synthetic data, seed={seed}",
    )


def _envelope(
    *,
    event_id: str,
    kind: str,
    source: str,
    occurred_at: datetime,
    entities: list[EntityRef],
    payload: dict[str, Any],
    generator: str,
    seed: int,
) -> EventEnvelope:
    """Build a fully deterministic envelope (no uuid, no wall-clock timestamps)."""
    env = EventEnvelope(
        event_id=event_id,
        kind=kind,
        occurred_at=occurred_at,
        recorded_at=occurred_at,
        source=source,
        entities=entities,
        payload=payload,
        provenance=_provenance(generator, seed),
    )
    assert env.provenance is not None
    env.provenance.record_hash = env.content_hash()
    return env


def _sort_envelopes(envelopes: list[EventEnvelope]) -> list[EventEnvelope]:
    return sorted(envelopes, key=lambda e: (e.occurred_at, e.event_id))


def _make_customer(rng: np.random.Generator, customer_id: str) -> Customer:
    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    risk = rng.choice([RiskTier.low, RiskTier.low, RiskTier.low, RiskTier.medium])
    return Customer(
        customer_id=customer_id,
        name=str(name),
        segment=str(rng.choice(["retail", "retail", "retail", "sme"])),
        risk_rating=RiskTier(str(risk)),
        country=str(rng.choice(_COUNTRIES)),
    )


def _make_account(rng: np.random.Generator, account_id: str, customer_id: str) -> Account:
    return Account(
        account_id=account_id,
        customer_id=customer_id,
        currency=str(rng.choice(["USD", "GBP", "EUR"])),
        account_type=str(rng.choice(["deposit", "deposit", "current", "savings"])),
        balance=round(float(rng.uniform(500.0, 250_000.0)), 2),
        opened_at=BASE_TIME - timedelta(days=int(rng.integers(30, 2000))),
    )


def _population(
    rng: np.random.Generator, n_accounts: int, seed: int, prefix: str
) -> tuple[list[Customer], list[Account]]:
    """One customer per account; ids are deterministic per (seed, prefix, index)."""
    customers: list[Customer] = []
    accounts: list[Account] = []
    for i in range(n_accounts):
        cust_id = f"cust_{prefix}_{seed}_{i:04d}"
        acc_id = f"acc_{prefix}_{seed}_{i:04d}"
        customers.append(_make_customer(rng, cust_id))
        accounts.append(_make_account(rng, acc_id, cust_id))
    return customers, accounts


# ---------------------------------------------------------------------------
# 1. Transactions with embedded laundering motifs
# ---------------------------------------------------------------------------


def gen_transactions(
    n: int = 400,
    ring_fraction: float = 0.1,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate ``n`` retail transactions with laundering motifs embedded.

    A fraction ``ring_fraction`` of the transactions belong to laundering rings using
    three classic motifs, rotated in turn:

    - **fan-in** — several dedicated source accounts each send one sub-threshold
      payment (structuring, just below 10,000) into a single collector ("hub");
    - **fan-out** — one hub disperses sub-threshold payments to several mule accounts;
    - **cycle** — funds traverse a closed loop ``A1 -> A2 -> ... -> A1``, each hop
      retaining 97–99% of the previous amount.

    Ring accounts are dedicated (never used for background traffic), so ground-truth
    motif structure is exact: a fan-in hub's in-degree in the returned graph equals
    the number of labelled ring transactions, and cycle members form a closed loop.
    Ring *customers* look ordinary — risk ratings and segments are drawn from the same
    distribution as everyone else — so labels never leak through the entities either.

    Args:
        n: Total number of transactions (background + ring).
        ring_fraction: Fraction of transactions that belong to laundering motifs,
            in ``[0, 0.9]``.
        seed: RNG seed; identical seeds produce byte-identical output.

    Returns:
        :class:`SyntheticDataset` with ``kind="transaction.posted"`` envelopes, a
        ``networkx.MultiDiGraph`` of account-to-account flows (edge attributes:
        ``txn_id``, ``amount``, ``is_ring``, ``motif``, ``ring_id``) and labels::

            labels["transactions"][txn_id] -> {"is_ring", "motif", "ring_id"}
            labels["rings"] -> [{"ring_id", "motif", "accounts", "hub", "txn_ids"}]
            labels["ring_accounts"] -> sorted list of all ring account ids
            labels["n_ring_transactions"], labels["n_background_transactions"]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 <= ring_fraction <= 0.9:
        raise ValueError("ring_fraction must be in [0, 0.9]")

    rng = np.random.default_rng(seed)
    n_ring_target = int(round(n * ring_fraction))
    n_background = n - n_ring_target

    n_accounts = max(12, n_background // 6)
    customers, accounts = _population(rng, n_accounts, seed, prefix="txn")
    country_by_customer = {c.customer_id: c.country for c in customers}
    customer_by_account = {a.account_id: a.customer_id for a in accounts}
    currency_by_account = {a.account_id: a.currency for a in accounts}

    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    for acc in accounts:
        graph.add_node(acc.account_id, is_ring=False, ring_id=None)

    envelopes: list[EventEnvelope] = []
    txn_labels: dict[str, dict[str, Any]] = {}
    rings: list[dict[str, Any]] = []
    ring_account_ids: list[str] = []
    txn_counter = 0

    def _add_account(acc_id: str, cust_id: str) -> None:
        customers.append(_make_customer(rng, cust_id))
        acc = _make_account(rng, acc_id, cust_id)
        accounts.append(acc)
        country_by_customer[cust_id] = customers[-1].country
        customer_by_account[acc_id] = cust_id
        currency_by_account[acc_id] = acc.currency

    def _emit(
        src: str, dst: str, amount: float, occurred_at: datetime,
        motif: str | None, ring_id: str | None,
    ) -> str:
        nonlocal txn_counter
        txn_id = f"txn_{seed}_{txn_counter:06d}"
        txn_counter += 1
        is_ring = ring_id is not None
        cross = (
            country_by_customer[customer_by_account[src]]
            != country_by_customer[customer_by_account[dst]]
        )
        payload = {
            "transaction_id": txn_id,
            "src_account": src,
            "dst_account": dst,
            "amount": round(float(amount), 2),
            "currency": currency_by_account[src],
            "channel": str(rng.choice(_TXN_CHANNELS)),
            "is_cross_border": bool(cross),
        }
        envelopes.append(
            _envelope(
                event_id=f"evt_{txn_id}",
                kind="transaction.posted",
                source="synthetic.transactions",
                occurred_at=occurred_at,
                entities=[
                    EntityRef(entity_type="account", entity_id=src),
                    EntityRef(entity_type="account", entity_id=dst),
                ],
                payload=payload,
                generator="transactions",
                seed=seed,
            )
        )
        txn_labels[txn_id] = {"is_ring": is_ring, "motif": motif, "ring_id": ring_id}
        graph.add_edge(
            src, dst, key=txn_id, txn_id=txn_id, amount=round(float(amount), 2),
            is_ring=is_ring, motif=motif, ring_id=ring_id,
        )
        return txn_id

    # -- laundering rings (dedicated accounts, generated first) -----------------
    motif_iter = itertools.cycle(["fan_in", "fan_out", "cycle"])
    remaining = n_ring_target
    ring_idx = 0
    while remaining > 0:
        motif = next(motif_iter)
        if motif == "cycle":
            k = int(rng.integers(3, 7))
            if k > remaining:  # a cycle needs at least 3 hops; degrade gracefully
                motif, k = "fan_in", remaining
        else:
            k = min(int(rng.integers(4, 9)), remaining)
        ring_id = f"ring_{seed}_{ring_idx:03d}"
        ring_idx += 1

        n_members = k + 1 if motif in {"fan_in", "fan_out"} else k
        members = []
        for j in range(n_members):
            acc_id = f"acc_ring_{seed}_{ring_idx - 1:03d}_{j:02d}"
            _add_account(acc_id, f"cust_ring_{seed}_{ring_idx - 1:03d}_{j:02d}")
            graph.add_node(acc_id, is_ring=True, ring_id=ring_id)
            members.append(acc_id)

        start = BASE_TIME + timedelta(
            days=int(rng.integers(0, 26)), hours=int(rng.integers(0, 12))
        )
        ts = start
        txn_ids: list[str] = []
        hub: str | None = None
        if motif == "fan_in":
            hub, sources = members[0], members[1:]
            for src in sources:
                ts = ts + timedelta(minutes=int(rng.integers(3, 180)))
                txn_ids.append(_emit(src, hub, rng.uniform(8200, 9850), ts, motif, ring_id))
        elif motif == "fan_out":
            hub, mules = members[0], members[1:]
            for dst in mules:
                ts = ts + timedelta(minutes=int(rng.integers(3, 180)))
                txn_ids.append(_emit(hub, dst, rng.uniform(8200, 9850), ts, motif, ring_id))
        else:  # cycle
            amount = float(rng.uniform(20_000, 90_000))
            for j in range(k):
                src, dst = members[j], members[(j + 1) % k]
                ts = ts + timedelta(minutes=int(rng.integers(10, 360)))
                txn_ids.append(_emit(src, dst, amount, ts, motif, ring_id))
                amount *= float(rng.uniform(0.97, 0.99))  # layering fee at each hop

        rings.append(
            {"ring_id": ring_id, "motif": motif, "accounts": members, "hub": hub,
             "txn_ids": txn_ids}
        )
        ring_account_ids.extend(members)
        remaining -= len(txn_ids)

    # -- background traffic (never touches ring accounts) -----------------------
    background_ids = [a.account_id for a in accounts if a.account_id not in set(ring_account_ids)]
    for _ in range(n_background):
        src, dst = (str(x) for x in rng.choice(background_ids, size=2, replace=False))
        amount = float(np.exp(rng.normal(5.5, 1.2)))
        occurred = BASE_TIME + timedelta(minutes=int(rng.integers(0, 28 * 24 * 60)))
        _emit(src, dst, amount, occurred, None, None)

    labels = {
        "transactions": txn_labels,
        "rings": rings,
        "ring_accounts": sorted(ring_account_ids),
        "n_ring_transactions": sum(1 for v in txn_labels.values() if v["is_ring"]),
        "n_background_transactions": sum(1 for v in txn_labels.values() if not v["is_ring"]),
    }
    return SyntheticDataset(
        name="synthetic-transactions",
        seed=seed,
        envelopes=_sort_envelopes(envelopes),
        labels=labels,
        entities={"customers": customers, "accounts": accounts},
        graph=graph,
        metadata={"n": n, "ring_fraction": ring_fraction, "base_time": BASE_TIME.isoformat()},
    )


# ---------------------------------------------------------------------------
# 2. Layered fraud ring
# ---------------------------------------------------------------------------


def gen_fraud_ring(size: int = 9, layers: int = 3, seed: int = 7) -> SyntheticDataset:
    """Generate one layered laundering ring as a graph plus transfer envelopes.

    ``size`` accounts are distributed across ``layers`` layers: layer 0 holds the
    placement accounts (where illicit cash enters), the middle layers hold mules, and
    the final layer holds the integration accounts. Every account forwards its
    incoming funds to one or two accounts in the next layer, skimming a 1–3% fee per
    hop, so value provably flows source -> integration and the graph is a layered DAG.

    Args:
        size: Total number of accounts in the ring (``size >= layers``).
        layers: Number of layers (``layers >= 2``).
        seed: RNG seed.

    Returns:
        :class:`SyntheticDataset` whose ``graph`` is a ``networkx.DiGraph`` with node
        attributes ``layer`` and ``role`` (``source`` / ``mule`` / ``integrator``) and
        edge attributes ``txn_id`` / ``amount``; one ``transaction.posted`` envelope
        per edge; and labels::

            labels["ring_id"], labels["layers"] -> {layer_index: [account_ids]}
            labels["roles"] -> {account_id: role}
            labels["total_placed"], labels["total_integrated"], labels["txn_ids"]
    """
    if layers < 2:
        raise ValueError("layers must be >= 2")
    if size < layers:
        raise ValueError("size must be >= layers (every layer needs an account)")

    rng = np.random.default_rng(seed)
    ring_id = f"fraudring_{seed}_{size}x{layers}"

    # Distribute accounts across layers, extra capacity towards the early layers.
    per_layer = [size // layers] * layers
    for i in range(size % layers):
        per_layer[i] += 1

    layer_accounts: dict[int, list[str]] = {}
    customers: list[Customer] = []
    accounts: list[Account] = []
    graph: nx.DiGraph = nx.DiGraph()
    roles: dict[str, str] = {}
    idx = 0
    for layer in range(layers):
        ids = []
        role = "source" if layer == 0 else ("integrator" if layer == layers - 1 else "mule")
        for _ in range(per_layer[layer]):
            acc_id = f"acc_fr_{seed}_{layer}_{idx:03d}"
            cust_id = f"cust_fr_{seed}_{layer}_{idx:03d}"
            idx += 1
            customers.append(_make_customer(rng, cust_id))
            accounts.append(_make_account(rng, acc_id, cust_id))
            graph.add_node(acc_id, layer=layer, role=role, ring_id=ring_id)
            roles[acc_id] = role
            ids.append(acc_id)
        layer_accounts[layer] = ids

    incoming: dict[str, float] = {acc: 0.0 for acc in roles}
    total_placed = 0.0
    for acc in layer_accounts[0]:
        placed = float(rng.uniform(15_000, 60_000))
        incoming[acc] = placed
        total_placed += placed

    envelopes: list[EventEnvelope] = []
    txn_ids: list[str] = []
    counter = 0
    for layer in range(layers - 1):
        nxt = layer_accounts[layer + 1]
        chosen_targets: dict[str, list[str]] = {}
        for src in layer_accounts[layer]:
            n_targets = int(rng.integers(1, min(2, len(nxt)) + 1))
            picks = rng.choice(nxt, size=n_targets, replace=False)
            chosen_targets[src] = [str(p) for p in picks]
        # Guarantee every next-layer account receives funds (no orphan mules).
        covered = {t for targets in chosen_targets.values() for t in targets}
        for orphan in [t for t in nxt if t not in covered]:
            donor = str(rng.choice(layer_accounts[layer]))
            chosen_targets[donor].append(orphan)

        for src in layer_accounts[layer]:
            targets = chosen_targets[src]
            fee = float(rng.uniform(0.01, 0.03))
            forwarded = incoming[src] * (1.0 - fee)
            share = forwarded / len(targets)
            for hop, dst in enumerate(targets):
                txn_id = f"txn_fr_{seed}_{counter:05d}"
                counter += 1
                occurred = BASE_TIME + timedelta(
                    days=layer, hours=int(rng.integers(0, 18)), minutes=int(rng.integers(0, 60))
                )
                graph.add_edge(src, dst, txn_id=txn_id, amount=round(share, 2), hop=hop)
                envelopes.append(
                    _envelope(
                        event_id=f"evt_{txn_id}",
                        kind="transaction.posted",
                        source="synthetic.fraud_ring",
                        occurred_at=occurred,
                        entities=[
                            EntityRef(entity_type="account", entity_id=src),
                            EntityRef(entity_type="account", entity_id=dst),
                        ],
                        payload={
                            "transaction_id": txn_id,
                            "src_account": src,
                            "dst_account": dst,
                            "amount": round(share, 2),
                            "currency": "USD",
                            "channel": str(rng.choice(["wire", "ach", "internal"])),
                            "is_cross_border": bool(rng.random() < 0.3),
                        },
                        generator="fraud_ring",
                        seed=seed,
                    )
                )
                txn_ids.append(txn_id)
                incoming[dst] += share

    total_integrated = sum(incoming[acc] for acc in layer_accounts[layers - 1])
    labels = {
        "ring_id": ring_id,
        "layers": layer_accounts,
        "roles": roles,
        "total_placed": round(total_placed, 2),
        "total_integrated": round(total_integrated, 2),
        "txn_ids": txn_ids,
    }
    return SyntheticDataset(
        name="fraud-rings",
        seed=seed,
        envelopes=_sort_envelopes(envelopes),
        labels=labels,
        entities={"customers": customers, "accounts": accounts},
        graph=graph,
        metadata={"size": size, "layers": layers, "base_time": BASE_TIME.isoformat()},
    )


# ---------------------------------------------------------------------------
# 3. Customer journeys with complaint escalation
# ---------------------------------------------------------------------------


def gen_customer_journeys(n: int = 40, seed: int = 7) -> SyntheticDataset:
    """Generate ``n`` multi-channel customer journeys with complaint-escalation paths.

    Each journey starts with onboarding stages, then with calibrated probabilities an
    issue occurs, a complaint is raised and acknowledged, escalates internally and —
    for a small tail — reaches the ombudsman, before resolving or closing unresolved:

    ``onboarding -> kyc_check -> first_deposit -> regular_usage
    [-> issue_reported -> complaint_raised -> complaint_acknowledged
    -> escalation_internal -> escalation_ombudsman] -> resolved | closed_unresolved``

    Every touchpoint is one ``journey.touchpoint`` envelope carrying the journey id,
    stage, channel, a sentiment score that deteriorates along the complaint path, and
    a channel-dependent wait time. Complaint journeys additionally yield a canonical
    :class:`CaseRecord` under ``entities["cases"]``.

    Args:
        n: Number of journeys.
        seed: RNG seed.

    Returns:
        :class:`SyntheticDataset` with labels::

            labels["journeys"][journey_id] -> {"customer_id", "had_issue",
                "issue_type", "complained", "escalated", "ombudsman", "resolved",
                "stages", "channels"}
            labels["n_complaints"], labels["n_escalated"], labels["n_ombudsman"]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    customers, _accounts = _population(rng, n, seed, prefix="jr")

    envelopes: list[EventEnvelope] = []
    cases: list[CaseRecord] = []
    journeys: dict[str, dict[str, Any]] = {}

    channel_weights = {
        "default": [0.30, 0.25, 0.15, 0.10, 0.10, 0.10],
        "complaint": [0.05, 0.05, 0.15, 0.45, 0.15, 0.15],
    }
    complaint_stages = {
        "issue_reported", "complaint_raised", "complaint_acknowledged",
        "escalation_internal", "escalation_ombudsman",
    }

    for i in range(n):
        journey_id = f"jr_{seed}_{i:04d}"
        customer = customers[i]
        stages: list[tuple[str, str | None]] = [
            ("onboarding", None), ("kyc_check", None),
            ("first_deposit", None), ("regular_usage", None),
        ]
        had_issue = bool(rng.random() < 0.45)
        issue_type: str | None = None
        complained = escalated = ombudsman = False
        resolved: bool | None = None
        if had_issue:
            issue_type = str(rng.choice(_ISSUE_TYPES))
            stages.append(("issue_reported", issue_type))
            complained = bool(rng.random() < 0.7)
            if complained:
                stages.append(("complaint_raised", issue_type))
                if rng.random() < 0.8:
                    stages.append(("complaint_acknowledged", issue_type))
                escalated = bool(rng.random() < 0.4)
                if escalated:
                    stages.append(("escalation_internal", issue_type))
                    ombudsman = bool(rng.random() < 0.3)
                    if ombudsman:
                        stages.append(("escalation_ombudsman", issue_type))
            resolved = bool(rng.random() < (0.6 if escalated else 0.85))
            stages.append(("resolved" if resolved else "closed_unresolved", issue_type))

        sentiment = round(float(rng.uniform(0.3, 0.8)), 3)
        ts = BASE_TIME + timedelta(days=int(rng.integers(0, 20)))
        stage_names: list[str] = []
        channels: list[str] = []
        for step, (stage, detail) in enumerate(stages):
            weights = channel_weights["complaint" if stage in complaint_stages else "default"]
            channel = str(rng.choice(_JOURNEY_CHANNELS, p=weights))
            if stage == "issue_reported":
                sentiment -= 0.3
            elif stage == "complaint_raised":
                sentiment -= 0.2
            elif stage in {"escalation_internal", "escalation_ombudsman"}:
                sentiment -= 0.2
            elif stage == "complaint_acknowledged":
                sentiment += 0.05
            elif stage == "resolved":
                sentiment += 0.4
            sentiment = round(max(-1.0, min(1.0, sentiment)), 3)
            wait = (
                int(rng.integers(5, 45)) if channel == "call_centre"
                else int(rng.integers(5, 30)) if channel == "branch"
                else int(rng.integers(0, 5))
            )
            ts = ts + timedelta(hours=int(rng.integers(2, 96)))
            envelopes.append(
                _envelope(
                    event_id=f"evt_{journey_id}_{step:02d}",
                    kind="journey.touchpoint",
                    source="synthetic.journeys",
                    occurred_at=ts,
                    entities=[EntityRef(entity_type="customer", entity_id=customer.customer_id)],
                    payload={
                        "journey_id": journey_id,
                        "customer_id": customer.customer_id,
                        "stage": stage,
                        "detail": detail,
                        "channel": channel,
                        "step_index": step,
                        "sentiment": sentiment,
                        "wait_minutes": wait,
                    },
                    generator="journeys",
                    seed=seed,
                )
            )
            stage_names.append(stage)
            channels.append(channel)

        if complained:
            status = (
                CaseStatus.escalated if ombudsman
                else CaseStatus.closed if resolved
                else CaseStatus.open
            )
            cases.append(
                CaseRecord(
                    case_id=f"case_{journey_id}",
                    kind="complaint",
                    status=status,
                    priority=RiskTier.high if escalated else RiskTier.medium,
                    opened_at=ts,
                    entities=[EntityRef(entity_type="customer", entity_id=customer.customer_id)],
                    narrative=(
                        f"Customer {customer.customer_id} complained about {issue_type}; "
                        f"escalated={escalated}, ombudsman={ombudsman}, resolved={resolved}."
                    ),
                )
            )
        journeys[journey_id] = {
            "customer_id": customer.customer_id,
            "had_issue": had_issue,
            "issue_type": issue_type,
            "complained": complained,
            "escalated": escalated,
            "ombudsman": ombudsman,
            "resolved": resolved,
            "stages": stage_names,
            "channels": channels,
        }

    labels = {
        "journeys": journeys,
        "n_complaints": sum(1 for j in journeys.values() if j["complained"]),
        "n_escalated": sum(1 for j in journeys.values() if j["escalated"]),
        "n_ombudsman": sum(1 for j in journeys.values() if j["ombudsman"]),
    }
    return SyntheticDataset(
        name="journeys",
        seed=seed,
        envelopes=_sort_envelopes(envelopes),
        labels=labels,
        entities={"customers": customers, "cases": cases},
        metadata={"n": n, "base_time": BASE_TIME.isoformat()},
    )


# ---------------------------------------------------------------------------
# 4. Market paths: GBM + jumps + volatility clustering
# ---------------------------------------------------------------------------


def gen_market_paths(
    n_instruments: int = 4,
    n_steps: int = 252,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate daily price paths with drift, jumps and volatility clustering.

    Per instrument, daily log-returns follow a jump-diffusion with a GARCH(1,1)
    variance recursion (implemented directly in numpy)::

        r_t = mu*dt - v_t/2 + sqrt(v_t) * z_t + J_t
        v_{t+1} = omega + alpha * (r_t - mu*dt)^2 + beta * v_t

    where ``z_t ~ N(0,1)``, jumps ``J_t`` arrive via a Poisson process (2–6 per year)
    with negative-mean lognormal-style sizes, and ``omega`` is set so the long-run
    variance matches each instrument's drawn annual volatility. Prices start at 100.

    Args:
        n_instruments: Number of instruments (synthetic tickers ``SYN01``...).
        n_steps: Number of daily steps per path.
        seed: RNG seed.

    Returns:
        :class:`SyntheticDataset` with one ``market.bar`` envelope per instrument per
        step (payload: symbol, step, date, close, log_return) and labels::

            labels["params"][symbol] -> {"mu", "sigma_annual", "alpha", "beta",
                                          "jump_intensity_per_year"}
            labels["jump_steps"][symbol] -> [step indices where a jump occurred]
            labels["prices"][symbol] -> [n_steps close prices]
            labels["realised_vol_annual"][symbol] -> float
    """
    if n_instruments < 1 or n_steps < 2:
        raise ValueError("need n_instruments >= 1 and n_steps >= 2")
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    mu = rng.uniform(0.02, 0.10, n_instruments)
    sigma = rng.uniform(0.12, 0.35, n_instruments)
    alpha = rng.uniform(0.05, 0.12, n_instruments)
    beta = np.minimum(rng.uniform(0.80, 0.92, n_instruments), 0.99 - alpha)
    v_long = (sigma**2) * dt  # long-run daily variance
    omega = v_long * (1.0 - alpha - beta)
    jump_intensity = rng.uniform(2.0, 6.0, n_instruments)  # jumps per year
    jump_mu = rng.uniform(-0.06, -0.01, n_instruments)
    jump_sigma = rng.uniform(0.02, 0.06, n_instruments)

    z = rng.standard_normal((n_steps, n_instruments))
    n_jumps = rng.poisson(jump_intensity * dt, size=(n_steps, n_instruments))
    jump_noise = rng.standard_normal((n_steps, n_instruments))
    jumps = n_jumps * jump_mu + np.sqrt(np.maximum(n_jumps, 0)) * jump_sigma * jump_noise

    returns = np.zeros((n_steps, n_instruments))
    v = v_long.copy()
    for t in range(n_steps):
        returns[t] = mu * dt - 0.5 * v + np.sqrt(v) * z[t] + jumps[t]
        innovation = returns[t] - mu * dt
        v = omega + alpha * innovation**2 + beta * v
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))

    instruments = [
        Instrument(
            instrument_id=f"ins_mkt_{seed}_{i:02d}",
            symbol=f"SYN{i + 1:02d}",
            asset_class="equity",
            currency="USD",
        )
        for i in range(n_instruments)
    ]

    envelopes: list[EventEnvelope] = []
    for i, instrument in enumerate(instruments):
        for t in range(n_steps):
            occurred = BASE_TIME + timedelta(days=t)
            envelopes.append(
                _envelope(
                    event_id=f"evt_bar_{seed}_{i:02d}_{t:04d}",
                    kind="market.bar",
                    source="synthetic.market_paths",
                    occurred_at=occurred,
                    entities=[
                        EntityRef(entity_type="instrument", entity_id=instrument.instrument_id)
                    ],
                    payload={
                        "symbol": instrument.symbol,
                        "step": t,
                        "date": occurred.date().isoformat(),
                        "close": round(float(prices[t, i]), 6),
                        "log_return": round(float(returns[t, i]), 8),
                    },
                    generator="market_paths",
                    seed=seed,
                )
            )

    labels: dict[str, Any] = {
        "params": {}, "jump_steps": {}, "prices": {}, "realised_vol_annual": {},
    }
    for i, instrument in enumerate(instruments):
        sym = instrument.symbol
        labels["params"][sym] = {
            "mu": float(mu[i]),
            "sigma_annual": float(sigma[i]),
            "alpha": float(alpha[i]),
            "beta": float(beta[i]),
            "jump_intensity_per_year": float(jump_intensity[i]),
        }
        labels["jump_steps"][sym] = [int(t) for t in np.flatnonzero(n_jumps[:, i] > 0)]
        labels["prices"][sym] = [round(float(p), 6) for p in prices[:, i]]
        labels["realised_vol_annual"][sym] = float(np.std(returns[:, i], ddof=1) * np.sqrt(252.0))

    return SyntheticDataset(
        name="market-paths",
        seed=seed,
        envelopes=_sort_envelopes(envelopes),
        labels=labels,
        entities={"instruments": instruments},
        metadata={
            "n_instruments": n_instruments, "n_steps": n_steps,
            "model": "GBM + Poisson jumps + GARCH(1,1) volatility clustering",
            "base_time": BASE_TIME.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# 5. Treasury cash-flow ladder
# ---------------------------------------------------------------------------


def gen_treasury_ladder(
    currencies: Sequence[str] = ("USD", "EUR", "GBP"),
    days: int = 30,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate a daily treasury cash-flow ladder per currency.

    Each currency book starts with an opening balance and accrues daily contractual
    inflows (maturing assets, scheduled receipts) and outflows (deposit run-off,
    debt maturities, scheduled payments), including occasional lumpy maturities that
    stress the cumulative position. One ``treasury.ladder_slot`` envelope is emitted
    per currency-day with inflow, outflow, net and the running cumulative position.

    Args:
        currencies: Currency codes, one ladder per code.
        days: Ladder horizon in days (``>= 1``).
        seed: RNG seed.

    Returns:
        :class:`SyntheticDataset` with labels::

            labels["opening_balance"][ccy] -> float
            labels["cumulative"][ccy] -> [running position per day]
            labels["first_negative_day"][ccy] -> int | None
            labels["worst_day"][ccy] -> {"day", "cumulative"}
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    if not currencies:
        raise ValueError("at least one currency is required")
    rng = np.random.default_rng(seed)

    envelopes: list[EventEnvelope] = []
    labels: dict[str, Any] = {
        "opening_balance": {}, "cumulative": {},
        "first_negative_day": {}, "worst_day": {},
    }
    for ccy in currencies:
        book_id = f"book_{str(ccy).lower()}"
        opening = round(float(rng.uniform(40e6, 160e6)), 2)
        base_in = float(rng.uniform(3e6, 9e6))
        base_out = float(rng.uniform(3e6, 9e6))
        cumulative = opening
        cumulative_series: list[float] = []
        first_negative: int | None = None
        worst = {"day": 0, "cumulative": opening}
        for day in range(days):
            inflow = base_in * float(rng.lognormal(0.0, 0.35))
            outflow = base_out * float(rng.lognormal(0.0, 0.35))
            if rng.random() < 0.08:  # lumpy debt maturity / collateral call
                outflow += float(rng.uniform(10e6, 30e6))
            if rng.random() < 0.06:  # lumpy asset maturity
                inflow += float(rng.uniform(8e6, 20e6))
            inflow, outflow = round(inflow, 2), round(outflow, 2)
            net = round(inflow - outflow, 2)
            cumulative = round(cumulative + net, 2)
            cumulative_series.append(cumulative)
            if cumulative < 0 and first_negative is None:
                first_negative = day
            if cumulative < worst["cumulative"]:
                worst = {"day": day, "cumulative": cumulative}
            occurred = BASE_TIME + timedelta(days=day)
            envelopes.append(
                _envelope(
                    event_id=f"evt_tl_{seed}_{str(ccy).lower()}_{day:03d}",
                    kind="treasury.ladder_slot",
                    source="synthetic.treasury_ladder",
                    occurred_at=occurred,
                    entities=[EntityRef(entity_type="treasury_book", entity_id=book_id)],
                    payload={
                        "currency": str(ccy),
                        "day": day,
                        "date": occurred.date().isoformat(),
                        "opening_balance": opening,
                        "inflow": inflow,
                        "outflow": outflow,
                        "net": net,
                        "cumulative": cumulative,
                    },
                    generator="treasury_ladder",
                    seed=seed,
                )
            )
        labels["opening_balance"][str(ccy)] = opening
        labels["cumulative"][str(ccy)] = cumulative_series
        labels["first_negative_day"][str(ccy)] = first_negative
        labels["worst_day"][str(ccy)] = worst

    return SyntheticDataset(
        name="treasury-ladders",
        seed=seed,
        envelopes=_sort_envelopes(envelopes),
        labels=labels,
        metadata={
            "currencies": [str(c) for c in currencies], "days": days,
            "base_time": BASE_TIME.isoformat(),
        },
    )
