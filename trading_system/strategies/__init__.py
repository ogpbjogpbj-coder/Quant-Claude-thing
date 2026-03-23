"""Trading strategies module."""

from trading_system.strategies.base import BaseStrategy, Signal
from trading_system.strategies.momentum import MomentumStrategy
from trading_system.strategies.mean_reversion import MeanReversionStrategy
from trading_system.strategies.ml_ensemble import MLEnsembleStrategy
from trading_system.strategies.volatility_breakout import VolatilityBreakoutStrategy
from trading_system.strategies.trend_following import TrendFollowingStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "MLEnsembleStrategy",
    "VolatilityBreakoutStrategy",
    "TrendFollowingStrategy",
]
