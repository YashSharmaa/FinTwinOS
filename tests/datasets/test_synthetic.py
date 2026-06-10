"""Tests for the seeded synthetic generators: determinism, envelope validity and
ground-truth label consistency (the motifs claimed in labels actually exist in the
emitted graph/envelopes)."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from fintwinos.core.types import EventEnvelope
from fintwinos.datasets.synthetic import (
    SyntheticDataset,
    gen_customer_journeys,
    gen_fraud_ring,
    gen_market_paths,
    gen_transactions,
    gen_treasury_ladder,
)

GENERATORS = [
    ("transactions", lambda seed: gen_transactions(n=120, ring_fraction=0.15, seed=seed)),
    ("fraud_ring", lambda seed: gen_fraud_ring(size=10, layers=3, seed=seed)),
    ("journeys", lambda seed: gen_customer_journeys(n=25, seed=seed)),
    ("market_paths", lambda seed: gen_market_paths(n_instruments=3, n_steps=60, seed=seed)),
    ("treasury", lambda seed: gen_treasury_ladder(currencies=("USD", "EUR"), days=15, seed=seed)),
]


def _dump_envelopes(ds: SyntheticDataset) -> list[dict]:
    return [e.model_dump(mode="json") for e in ds.envelopes]


class TestDeterminism:
    @pytest.mark.parametrize("name,factory", GENERATORS, ids=[g[0] for g in GENERATORS])
    def test_same_seed_identical(self, name, factory):
        a, b = factory(11), factory(11)
        assert _dump_envelopes(a) == _dump_envelopes(b)
        assert json.dumps(a.labels, sort_keys=True, default=str) == json.dumps(
            b.labels, sort_keys=True, default=str
        )

    @pytest.mark.parametrize("name,factory", GENERATORS, ids=[g[0] for g in GENERATORS])
    def test_different_seed_differs(self, name, factory):
        a, b = factory(11), factory(12)
        assert _dump_envelopes(a) != _dump_envelopes(b)


class TestEnvelopeValidity:
    @pytest.mark.parametrize("name,factory", GENERATORS, ids=[g[0] for g in GENERATORS])
    def test_envelopes_parse_and_serialise(self, name, factory):
        ds = factory(7)
        assert ds.envelopes, "generator emitted no envelopes"
        for env in ds.envelopes:
            # Round-trips through the canonical model and plain JSON.
            parsed = EventEnvelope.model_validate(env.model_dump())
            assert parsed.kind == env.kind
            json.dumps(env.payload)  # payload must be plain-JSON serialisable
            assert env.entities, "every synthetic envelope must reference entities"
            assert env.provenance is not None
            assert env.provenance.record_hash == env.content_hash()
            assert "synthetic" in env.provenance.source_system

    @pytest.mark.parametrize("name,factory", GENERATORS, ids=[g[0] for g in GENERATORS])
    def test_envelopes_sorted_and_unique(self, name, factory):
        ds = factory(7)
        ids = [e.event_id for e in ds.envelopes]
        assert len(ids) == len(set(ids))
        times = [e.occurred_at for e in ds.envelopes]
        assert times == sorted(times)


class TestTransactionsLabels:
    @pytest.fixture(scope="class")
    def ds(self) -> SyntheticDataset:
        return gen_transactions(n=200, ring_fraction=0.2, seed=21)

    def test_counts_add_up(self, ds):
        assert len(ds.envelopes) == 200
        labels = ds.labels
        assert labels["n_ring_transactions"] + labels["n_background_transactions"] == 200
        assert labels["n_ring_transactions"] == pytest.approx(200 * 0.2, abs=3)
        assert len(labels["transactions"]) == 200

    def test_every_envelope_labelled(self, ds):
        for env in ds.envelopes:
            txn_id = env.payload["transaction_id"]
            assert txn_id in ds.labels["transactions"]

    def test_labels_not_leaked_into_payloads(self, ds):
        for env in ds.envelopes:
            assert "is_ring" not in env.payload
            assert "ring_id" not in env.payload
            assert "motif" not in env.payload

    def test_fan_in_hub_in_degree(self, ds):
        """The fan-in motif must actually exist: hub in-degree == labelled txns."""
        fan_ins = [r for r in ds.labels["rings"] if r["motif"] == "fan_in"]
        assert fan_ins, "expected at least one fan-in ring at this size"
        for ring in fan_ins:
            hub = ring["hub"]
            assert ds.graph.in_degree(hub) == len(ring["txn_ids"])
            assert ds.graph.out_degree(hub) == 0  # collector only receives

    def test_fan_out_hub_out_degree(self, ds):
        fan_outs = [r for r in ds.labels["rings"] if r["motif"] == "fan_out"]
        assert fan_outs, "expected at least one fan-out ring at this size"
        for ring in fan_outs:
            hub = ring["hub"]
            assert ds.graph.out_degree(hub) == len(ring["txn_ids"])
            assert ds.graph.in_degree(hub) == 0

    def test_cycles_form_closed_loops(self, ds):
        cycles = [r for r in ds.labels["rings"] if r["motif"] == "cycle"]
        assert cycles, "expected at least one cycle ring at this size"
        for ring in cycles:
            sub = ds.graph.subgraph(ring["accounts"])
            found = nx.find_cycle(sub)
            assert len(found) == len(ring["accounts"])

    def test_structuring_amounts_below_threshold(self, ds):
        by_txn = {e.payload["transaction_id"]: e for e in ds.envelopes}
        for ring in ds.labels["rings"]:
            if ring["motif"] in {"fan_in", "fan_out"}:
                for txn_id in ring["txn_ids"]:
                    assert by_txn[txn_id].payload["amount"] < 10_000

    def test_ring_accounts_dedicated(self, ds):
        """Background traffic never touches ring accounts, so labels are exact."""
        ring_accounts = set(ds.labels["ring_accounts"])
        for env in ds.envelopes:
            txn_id = env.payload["transaction_id"]
            if not ds.labels["transactions"][txn_id]["is_ring"]:
                assert env.payload["src_account"] not in ring_accounts
                assert env.payload["dst_account"] not in ring_accounts

    def test_zero_ring_fraction(self):
        ds = gen_transactions(n=40, ring_fraction=0.0, seed=5)
        assert ds.labels["n_ring_transactions"] == 0
        assert ds.labels["rings"] == []
        assert len(ds.envelopes) == 40

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            gen_transactions(n=0)
        with pytest.raises(ValueError):
            gen_transactions(n=10, ring_fraction=0.95)


class TestFraudRing:
    @pytest.fixture(scope="class")
    def ds(self) -> SyntheticDataset:
        return gen_fraud_ring(size=11, layers=4, seed=9)

    def test_structure(self, ds):
        graph = ds.graph
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 11
        assert nx.is_directed_acyclic_graph(graph)
        layers = ds.labels["layers"]
        assert sorted(layers) == [0, 1, 2, 3]
        assert sum(len(v) for v in layers.values()) == 11
        # Edges only flow to the immediately next layer.
        for src, dst in graph.edges():
            assert graph.nodes[dst]["layer"] == graph.nodes[src]["layer"] + 1

    def test_roles_match_layers(self, ds):
        for acc, role in ds.labels["roles"].items():
            layer = ds.graph.nodes[acc]["layer"]
            expected = "source" if layer == 0 else ("integrator" if layer == 3 else "mule")
            assert role == expected

    def test_no_orphan_mules(self, ds):
        """Every non-source account receives funds; every non-integrator forwards."""
        for node in ds.graph.nodes():
            layer = ds.graph.nodes[node]["layer"]
            if layer > 0:
                assert ds.graph.in_degree(node) >= 1
            if layer < 3:
                assert ds.graph.out_degree(node) >= 1

    def test_value_flows_to_integration(self, ds):
        placed, integrated = ds.labels["total_placed"], ds.labels["total_integrated"]
        assert 0 < integrated < placed  # fees skimmed each hop
        assert integrated > placed * (1 - 0.03) ** 4  # bounded by max fee per layer

    def test_one_envelope_per_edge(self, ds):
        assert len(ds.envelopes) == ds.graph.number_of_edges()
        assert len(ds.labels["txn_ids"]) == len(ds.envelopes)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            gen_fraud_ring(size=2, layers=3)
        with pytest.raises(ValueError):
            gen_fraud_ring(size=5, layers=1)


class TestJourneys:
    @pytest.fixture(scope="class")
    def ds(self) -> SyntheticDataset:
        return gen_customer_journeys(n=60, seed=13)

    def test_label_consistency_with_envelopes(self, ds):
        stages_by_journey: dict[str, list[str]] = {}
        for env in ds.envelopes:
            stages_by_journey.setdefault(env.payload["journey_id"], []).append(
                env.payload["stage"]
            )
        for journey_id, label in ds.labels["journeys"].items():
            stages = stages_by_journey[journey_id]
            assert sorted(stages) == sorted(label["stages"])
            assert label["complained"] == ("complaint_raised" in stages)
            assert label["escalated"] == ("escalation_internal" in stages)
            assert label["ombudsman"] == ("escalation_ombudsman" in stages)
            if label["complained"]:
                assert label["had_issue"]  # complaints only follow issues
            if label["escalated"]:
                assert label["complained"]

    def test_escalation_paths_present(self, ds):
        assert ds.labels["n_complaints"] > 0
        assert ds.labels["n_escalated"] > 0
        assert ds.labels["n_escalated"] <= ds.labels["n_complaints"]

    def test_complaint_cases_created(self, ds):
        cases = ds.entities["cases"]
        assert len(cases) == ds.labels["n_complaints"]
        for case in cases:
            assert case.kind == "complaint"
            assert case.entities[0].entity_type == "customer"

    def test_sentiment_bounded(self, ds):
        for env in ds.envelopes:
            assert -1.0 <= env.payload["sentiment"] <= 1.0


class TestMarketPaths:
    @pytest.fixture(scope="class")
    def ds(self) -> SyntheticDataset:
        return gen_market_paths(n_instruments=3, n_steps=120, seed=17)

    def test_shapes(self, ds):
        assert len(ds.envelopes) == 3 * 120
        assert len(ds.entities["instruments"]) == 3
        for sym, prices in ds.labels["prices"].items():
            assert len(prices) == 120
            assert all(p > 0 for p in prices)
            assert sym in ds.labels["params"]

    def test_prices_match_envelopes(self, ds):
        for env in ds.envelopes:
            sym, step = env.payload["symbol"], env.payload["step"]
            assert env.payload["close"] == pytest.approx(ds.labels["prices"][sym][step])

    def test_jump_ground_truth(self, ds):
        for steps in ds.labels["jump_steps"].values():
            assert all(0 <= s < 120 for s in steps)
        # With 2-6 jumps/year over ~half a year across 3 instruments, expect >= 1.
        total_jumps = sum(len(s) for s in ds.labels["jump_steps"].values())
        assert total_jumps >= 1

    def test_vol_clustering_params_stationary(self, ds):
        for params in ds.labels["params"].values():
            assert params["alpha"] + params["beta"] < 1.0
            assert 0.12 <= params["sigma_annual"] <= 0.35

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            gen_market_paths(n_instruments=0)
        with pytest.raises(ValueError):
            gen_market_paths(n_steps=1)


class TestTreasuryLadder:
    @pytest.fixture(scope="class")
    def ds(self) -> SyntheticDataset:
        return gen_treasury_ladder(currencies=("USD", "EUR", "GBP"), days=40, seed=23)

    def test_shapes(self, ds):
        assert len(ds.envelopes) == 3 * 40
        for ccy in ("USD", "EUR", "GBP"):
            assert len(ds.labels["cumulative"][ccy]) == 40

    def test_cumulative_reconciles(self, ds):
        """labels.cumulative must equal opening + running sum of per-day nets."""
        by_ccy: dict[str, list] = {}
        for env in ds.envelopes:
            by_ccy.setdefault(env.payload["currency"], []).append(env)
        for ccy, envs in by_ccy.items():
            envs.sort(key=lambda e: e.payload["day"])
            running = ds.labels["opening_balance"][ccy]
            for env in envs:
                assert env.payload["net"] == pytest.approx(
                    env.payload["inflow"] - env.payload["outflow"], abs=0.011
                )
                running = round(running + env.payload["net"], 2)
                assert env.payload["cumulative"] == pytest.approx(running, abs=0.01)
                assert ds.labels["cumulative"][ccy][env.payload["day"]] == pytest.approx(
                    running, abs=0.01
                )

    def test_worst_day_is_minimum(self, ds):
        for ccy in ("USD", "EUR", "GBP"):
            series = ds.labels["cumulative"][ccy]
            opening = ds.labels["opening_balance"][ccy]
            worst = ds.labels["worst_day"][ccy]
            # worst tracks the running minimum, seeded with the opening balance.
            assert worst["cumulative"] == min([opening, *series])
            if worst["cumulative"] != opening:
                assert series[worst["day"]] == worst["cumulative"]

    def test_first_negative_day_consistent(self, ds):
        for ccy in ("USD", "EUR", "GBP"):
            series = ds.labels["cumulative"][ccy]
            first_negative = ds.labels["first_negative_day"][ccy]
            negatives = [i for i, v in enumerate(series) if v < 0]
            assert first_negative == (negatives[0] if negatives else None)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            gen_treasury_ladder(days=0)
        with pytest.raises(ValueError):
            gen_treasury_ladder(currencies=())
