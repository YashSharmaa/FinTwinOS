"""Named stress-scenario presets for the FinTwinOS simulators.

The :class:`ScenarioCatalog` ships a curated set of stress presets — each a fully
parameterised :class:`~fintwinos.core.types.Scenario` whose ``params["simulator"]``
names the twin_sim simulator it targets — plus calm baselines for severity
comparisons. Catalogs are mutable registries: demos, evals and the RL module can
:meth:`~ScenarioCatalog.register` their own presets without touching this file.

``get`` always returns a *fresh* Scenario instance (new ``scenario_id``) so two runs
of the same preset remain distinguishable in the audit trail.
"""

from __future__ import annotations

from fintwinos.core.types import Scenario, new_id


def _presets() -> list[Scenario]:
    """Build the shipped preset list (fresh instances on every call)."""
    return [
        Scenario(
            name="usd_liquidity_squeeze",
            kind="stress",
            narrative=(
                "A funding squeeze concentrated in USD: wholesale outflows triple, "
                "EUR/GBP books wobble in sympathy, and the funding spread gaps 150bp. "
                "Tests how many days the counterbalancing capacity buys."
            ),
            params={
                "simulator": "treasury",
                "horizon_days": 30,
                "stress_outflow_multiplier": {"USD": 3.0, "EUR": 1.3, "GBP": 1.2},
                "shock_bps": 150.0,
            },
        ),
        Scenario(
            name="flash_crash",
            kind="stress",
            narrative=(
                "An intraday flash crash: a 300bp exogenous shock hits a stressed, "
                "thin-liquidity market with aggressive momentum flow, testing depth "
                "withdrawal, fat tails and the speed of mean reversion."
            ),
            params={
                "simulator": "market",
                "n_steps": 240,
                "regime": "stressed",
                "shock_bps": 300.0,
                "shock_step": 80,
            },
        ),
        Scenario(
            name="ring_surge",
            kind="stress",
            narrative=(
                "A laundering-ring surge: new fan-in/fan-out/cycle/layering motifs "
                "spawn at nearly four times the baseline rate, stressing alert "
                "precision and the analyst triage budget."
            ),
            params={
                "simulator": "compliance",
                "n_steps": 60,
                "ring_rate": 0.45,
            },
        ),
        Scenario(
            name="complaints_spike",
            kind="stress",
            narrative=(
                "A complaints spike after a product incident: arrivals jump to 140/hr "
                "against a normal staffing plan with elevated escalations, stressing "
                "SLA breaches and abandonment."
            ),
            params={
                "simulator": "customer_ops",
                "arrival_rate_per_hr": 140.0,
                "n_agents": 8,
                "sla_minutes": 15.0,
                "escalation_prob": 0.15,
            },
        ),
        Scenario(
            name="rates_shock_200bp",
            kind="stress",
            narrative=(
                "A 200bp rates shock: funding spreads reprice instantly and deposit "
                "outflows accelerate moderately across all currencies as money chases "
                "yield."
            ),
            params={
                "simulator": "treasury",
                "horizon_days": 30,
                "stress_outflow_multiplier": 1.6,
                "shock_bps": 200.0,
            },
        ),
        Scenario(
            name="calm_market_baseline",
            kind="what_if",
            narrative="Calm-regime market baseline for comparing stressed runs against.",
            params={"simulator": "market", "n_steps": 250, "regime": "calm", "shock_bps": 0.0},
        ),
        Scenario(
            name="treasury_baseline",
            kind="what_if",
            narrative="Business-as-usual cash ladder: no stress multiplier, no spread shock.",
            params={
                "simulator": "treasury",
                "horizon_days": 30,
                "stress_outflow_multiplier": 1.0,
                "shock_bps": 0.0,
            },
        ),
        Scenario(
            name="ops_baseline",
            kind="what_if",
            narrative="Normal contact-centre day: 60 contacts/hr against 8 agents.",
            params={
                "simulator": "customer_ops",
                "arrival_rate_per_hr": 60.0,
                "n_agents": 8,
                "sla_minutes": 15.0,
            },
        ),
    ]


class ScenarioCatalog:
    """Registry of named scenario presets.

    Args:
        include_defaults: When True (default) the shipped presets are pre-registered.
    """

    def __init__(self, include_defaults: bool = True):
        self._scenarios: dict[str, Scenario] = {}
        if include_defaults:
            for scenario in _presets():
                self._scenarios[scenario.name] = scenario

    def list(self) -> list[str]:
        """Sorted names of every registered scenario."""
        return sorted(self._scenarios)

    def get(self, name: str) -> Scenario:
        """Return a fresh copy of the named scenario.

        The copy carries a new ``scenario_id`` so repeated runs of the same preset
        stay distinguishable in audit and replay records.

        Raises:
            KeyError: If the name is not registered (message lists what is).
        """
        if name not in self._scenarios:
            available = ", ".join(self.list()) or "<empty catalog>"
            raise KeyError(f"unknown scenario '{name}'; available: {available}")
        return self._scenarios[name].model_copy(
            deep=True, update={"scenario_id": new_id("scn")}
        )

    def register(self, scenario: Scenario, *, overwrite: bool = False) -> None:
        """Register a scenario under its ``name``.

        Args:
            scenario: The preset to store (stored as-is; ``get`` hands out copies).
            overwrite: Allow replacing an existing preset of the same name.

        Raises:
            ValueError: If the name exists and ``overwrite`` is False.
        """
        if scenario.name in self._scenarios and not overwrite:
            raise ValueError(
                f"scenario '{scenario.name}' is already registered; pass overwrite=True"
            )
        self._scenarios[scenario.name] = scenario

    def for_simulator(self, simulator: str) -> list[str]:
        """Names of presets targeting the given simulator (via params['simulator'])."""
        return sorted(
            name
            for name, scn in self._scenarios.items()
            if scn.params.get("simulator") == simulator
        )

    def __contains__(self, name: str) -> bool:
        return name in self._scenarios

    def __len__(self) -> int:
        return len(self._scenarios)


def default_catalog() -> ScenarioCatalog:
    """A freshly built catalog containing the shipped presets."""
    return ScenarioCatalog(include_defaults=True)
