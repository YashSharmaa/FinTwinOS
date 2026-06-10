"""Seeded synthetic market data: a GBM-plus-jumps tick stream.

``SyntheticMarketConnector`` generates ``market.tick`` envelopes from a Merton-style
jump-diffusion: per tick, the log-mid moves by

    (mu - sigma^2 / 2) * dt  +  sigma * sqrt(dt) * Z  +  m * N + s * sqrt(N) * Z'

where ``Z, Z'`` are standard normals, ``N ~ Poisson(jump_intensity)`` is the jump
count for the tick, and jumps are ``Normal(jump_mean, jump_sigma)`` in log space
(the compound sum is sampled via its exact conditional distribution given ``N``).
All randomness flows through one ``numpy.random.default_rng(seed)``, so a given
``(seed, instruments, n_ticks, parameters)`` tuple always reproduces the identical
tick stream — which is exactly what replayable twin episodes need.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from fintwinos.connectors.base import BaseConnector, ConnectorError
from fintwinos.core.types import EntityRef, EventEnvelope, Provenance

TRADING_SECONDS_PER_YEAR = 252 * 6.5 * 3600.0
"""Seconds in a trading year (252 days of 6.5 hours), used to scale annual params."""

DEFAULT_START_TIME = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
"""Deterministic default first-tick timestamp (a market open), aiding reproducibility."""


class SyntheticMarketConnector(BaseConnector):
    """Stream a deterministic GBM+jump tick path as ``market.tick`` envelopes.

    Ticks are interleaved by time: tick 0 for every instrument, then tick 1, and
    so on, with timestamps spaced ``tick_interval_s`` apart starting at
    ``start_time``. Each payload carries ``instrument``, ``mid`` (rounded to six
    decimals) and ``ts`` (ISO-8601), plus the tick index and seed for traceability.

    Args:
        instruments: Instrument symbols, streamed in the given order (the order is
            part of the deterministic draw sequence).
        n_ticks: Ticks per instrument.
        seed: Seed for ``numpy.random.default_rng``.
        start_prices: A single price applied to all instruments, or a
            ``symbol -> price`` map (missing symbols use 100.0).
        mu: Annualised drift.
        sigma: Annualised diffusion volatility.
        jump_intensity: Expected jumps per tick (Poisson rate per tick).
        jump_mean: Mean log-jump size.
        jump_sigma: Std-dev of a single log-jump.
        tick_interval_s: Wall-clock seconds between consecutive ticks; also sets
            ``dt = tick_interval_s / TRADING_SECONDS_PER_YEAR`` for the dynamics.
        start_time: Timestamp of tick 0 (timezone-aware); defaults to a fixed
            constant rather than "now" so runs are reproducible end to end.
        name: Connector name.
    """

    def __init__(
        self,
        instruments: Sequence[str] = ("EQ.ACME",),
        *,
        n_ticks: int = 100,
        seed: int = 7,
        start_prices: float | dict[str, float] = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        jump_intensity: float = 0.02,
        jump_mean: float = 0.0,
        jump_sigma: float = 0.02,
        tick_interval_s: float = 60.0,
        start_time: datetime | None = None,
        name: str = "synthetic_market",
    ):
        if n_ticks < 1:
            raise ConnectorError("n_ticks must be >= 1")
        if not instruments:
            raise ConnectorError("at least one instrument is required")
        if sigma < 0 or jump_intensity < 0 or jump_sigma < 0:
            raise ConnectorError("sigma, jump_intensity and jump_sigma must be >= 0")
        self.instruments = list(instruments)
        self.n_ticks = int(n_ticks)
        self.seed = int(seed)
        self.start_prices = start_prices
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.jump_intensity = float(jump_intensity)
        self.jump_mean = float(jump_mean)
        self.jump_sigma = float(jump_sigma)
        self.tick_interval_s = float(tick_interval_s)
        self.start_time = start_time or DEFAULT_START_TIME
        if self.start_time.tzinfo is None:
            self.start_time = self.start_time.replace(tzinfo=UTC)
        super().__init__(name=name, source="synthetic_market")

    def _start_price(self, symbol: str) -> float:
        """Resolve the initial mid for one symbol."""
        if isinstance(self.start_prices, dict):
            return float(self.start_prices.get(symbol, 100.0))
        return float(self.start_prices)

    def paths(self) -> dict[str, np.ndarray]:
        """Generate the full deterministic mid-price path per instrument.

        Returns:
            ``symbol -> array of n_ticks mids``. Tick ``i`` already includes the
            ``i``-th increment, i.e. ``mid[i] = s0 * exp(sum(increments[: i + 1]))``.
        """
        rng = np.random.default_rng(self.seed)
        dt = self.tick_interval_s / TRADING_SECONDS_PER_YEAR
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)
        out: dict[str, np.ndarray] = {}
        for symbol in self.instruments:
            z = rng.standard_normal(self.n_ticks)
            jump_counts = rng.poisson(self.jump_intensity, self.n_ticks)
            z_jump = rng.standard_normal(self.n_ticks)
            jumps = self.jump_mean * jump_counts + self.jump_sigma * np.sqrt(jump_counts) * z_jump
            increments = drift + vol * z + jumps
            out[symbol] = self._start_price(symbol) * np.exp(np.cumsum(increments))
        return out

    async def stream(self) -> AsyncIterator[EventEnvelope]:
        """Yield ``n_ticks * len(instruments)`` envelopes, interleaved by time."""
        paths = self.paths()
        for tick in range(self.n_ticks):
            ts = self.start_time + timedelta(seconds=tick * self.tick_interval_s)
            for symbol in self.instruments:
                mid = round(float(paths[symbol][tick]), 6)
                yield EventEnvelope(
                    kind="market.tick",
                    source=self.source,
                    occurred_at=ts,
                    entities=[EntityRef(entity_type="instrument", entity_id=symbol)],
                    payload={
                        "instrument": symbol,
                        "mid": mid,
                        "ts": ts.isoformat(),
                        "tick": tick,
                        "seed": self.seed,
                    },
                    provenance=Provenance(
                        source_system=self.source,
                        notes=f"gbm+jump seed={self.seed} tick={tick}",
                    ),
                )
            await asyncio.sleep(0)
