"""Time-series models in pure numpy: AR(p) forecasting with prediction intervals,
RiskMetrics EWMA volatility, rolling-origin backtesting and regime detection."""

from fintwinos.models.time_series.forecast import (
    ARForecaster,
    EwmaVol,
    ForecastResult,
    NaiveForecaster,
    backtest,
)
from fintwinos.models.time_series.regime import RegimeDetector, RegimeResult

__all__ = [
    "ARForecaster",
    "EwmaVol",
    "ForecastResult",
    "NaiveForecaster",
    "RegimeDetector",
    "RegimeResult",
    "backtest",
]
