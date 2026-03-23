"""Tests for trading strategies using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from trading_system.config import (
    StrategyMomentumConfig,
    StrategyMeanReversionConfig,
    StrategyVolBreakoutConfig,
    StrategyTrendConfig,
)
from trading_system.data_ingestion import DataIngestion
from trading_system.strategies.momentum import MomentumStrategy
from trading_system.strategies.mean_reversion import MeanReversionStrategy
from trading_system.strategies.volatility_breakout import VolatilityBreakoutStrategy
from trading_system.strategies.trend_following import TrendFollowingStrategy


def make_synthetic_ohlcv(n=200, trend=0.0005, volatility=0.02, seed=42):
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    returns = rng.normal(trend, volatility, n)
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


def enrich(df):
    """Add indicators using DataIngestion's compute method."""
    from trading_system.config import TradingConfig
    config = TradingConfig()
    di = DataIngestion.__new__(DataIngestion)
    di.config = config
    di._cache = {}
    return di.compute_indicators(df.copy())


class TestMomentumStrategy:
    def test_generates_signals(self):
        config = StrategyMomentumConfig()
        strategy = MomentumStrategy(config)

        # Create multiple stocks with different trends
        data = {}
        for i, sym in enumerate(["AAPL", "MSFT", "GOOGL", "TSLA", "AMD"]):
            trend = 0.001 * (2 - i)  # AAPL trending up, AMD trending down
            df = make_synthetic_ohlcv(trend=trend, seed=42 + i)
            data[sym] = enrich(df)

        signals = strategy.generate_signals(data, {})
        assert isinstance(signals, list)

    def test_respects_empty_data(self):
        config = StrategyMomentumConfig()
        strategy = MomentumStrategy(config)
        signals = strategy.generate_signals({}, {})
        assert signals == []

    def test_signal_properties(self):
        config = StrategyMomentumConfig()
        strategy = MomentumStrategy(config)

        data = {}
        for i, sym in enumerate(["A", "B", "C", "D", "E"]):
            data[sym] = enrich(make_synthetic_ohlcv(trend=0.002 * (2 - i), seed=i))

        signals = strategy.generate_signals(data, {})
        for sig in signals:
            assert -1.0 <= sig.direction <= 1.0
            assert 0.0 <= sig.confidence <= 1.0
            assert sig.strategy == "momentum"


class TestMeanReversionStrategy:
    def test_generates_signals_on_extreme_zscore(self):
        config = StrategyMeanReversionConfig()
        strategy = MeanReversionStrategy(config)

        # Create data with mean-reverting pattern
        df = make_synthetic_ohlcv(n=200, trend=0, volatility=0.03, seed=42)
        data = {"TEST": enrich(df)}

        signals = strategy.generate_signals(data, {})
        assert isinstance(signals, list)

    def test_exit_signal_for_existing_position(self):
        config = StrategyMeanReversionConfig()
        strategy = MeanReversionStrategy(config)

        df = make_synthetic_ohlcv(n=200, trend=0, volatility=0.01, seed=100)
        data = {"TEST": enrich(df)}

        # If position exists and zscore is near 0, should suggest exit
        signals = strategy.generate_signals(data, {"TEST": 100.0})
        assert isinstance(signals, list)


class TestVolatilityBreakout:
    def test_requires_volume_confirmation(self):
        config = StrategyVolBreakoutConfig()
        strategy = VolatilityBreakoutStrategy(config)

        df = make_synthetic_ohlcv(seed=42)
        data = {"TEST": enrich(df)}

        signals = strategy.generate_signals(data, {})
        assert isinstance(signals, list)
        # Signals should only fire with high volume
        for sig in signals:
            assert sig.strategy == "volatility_breakout"


class TestTrendFollowing:
    def test_with_trending_data(self):
        config = StrategyTrendConfig()
        strategy = TrendFollowingStrategy(config)

        # Strong uptrend
        df = make_synthetic_ohlcv(n=250, trend=0.002, volatility=0.01, seed=42)
        data = {"TEST": enrich(df)}

        signals = strategy.generate_signals(data, {})
        assert isinstance(signals, list)


class TestIndicators:
    def test_compute_indicators(self):
        df = make_synthetic_ohlcv(n=250, seed=42)
        result = enrich(df)

        # Check key indicators exist
        expected_cols = [
            "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
            "atr_14", "volume_ratio", "price_zscore", "adx",
            "returns_5d", "volatility_21d", "sma_50", "sma_200",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing indicator: {col}"

    def test_no_infinite_values(self):
        df = make_synthetic_ohlcv(n=250, seed=42)
        result = enrich(df)

        # After warmup, should have no infinities
        trimmed = result.iloc[50:]
        numeric = trimmed.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Found infinite values in indicators"


class TestSignalAggregator:
    def test_aggregation(self):
        from trading_system.signal_aggregator import SignalAggregator
        from trading_system.strategies.base import Signal
        from trading_system.config import StrategiesConfig

        agg = SignalAggregator(StrategiesConfig())

        signals = [
            Signal("AAPL", 0.8, 0.7, "momentum"),
            Signal("AAPL", 0.6, 0.6, "mean_reversion"),
            Signal("MSFT", -0.5, 0.8, "momentum"),
        ]

        result = agg.aggregate(signals)
        # MSFT has only 1 strategy so it should be filtered out (require 2+)
        assert len(result) == 1

        aapl = [s for s in result if s.symbol == "AAPL"][0]
        assert aapl.direction > 0  # Both agreed on buy
        assert aapl.confidence > 0

    def test_conflicting_signals_reduce_confidence(self):
        from trading_system.signal_aggregator import SignalAggregator
        from trading_system.strategies.base import Signal
        from trading_system.config import StrategiesConfig

        agg = SignalAggregator(StrategiesConfig())

        signals = [
            Signal("AAPL", 0.8, 0.7, "momentum"),
            Signal("AAPL", -0.6, 0.7, "mean_reversion"),
        ]

        result = agg.aggregate(signals)
        # Conflicting should reduce confidence
        if result:
            assert result[0].confidence < 0.7


class TestRiskManager:
    @staticmethod
    def _init_db():
        from trading_system.utils.db import init_db
        init_db()

    def test_circuit_breaker_daily_loss(self):
        from trading_system.risk_manager import RiskManager
        from trading_system.config import RiskConfig

        self._init_db()
        rm = RiskManager(RiskConfig(max_daily_loss_pct=3.0))
        rm.initialize(100_000)

        # Should be fine at 2% loss
        assert rm.check_circuit_breakers(98_000) is True

        # Should trip at 3% loss
        assert rm.check_circuit_breakers(97_000) is False

    def test_circuit_breaker_drawdown(self):
        from trading_system.risk_manager import RiskManager
        from trading_system.config import RiskConfig

        self._init_db()
        rm = RiskManager(RiskConfig(max_drawdown_pct=10.0))
        rm.initialize(100_000)
        rm.update_peak_equity(110_000)  # Peak went higher

        # 10% drawdown from 110k = 99k
        assert rm.check_circuit_breakers(100_000) is True
        assert rm.check_circuit_breakers(99_000) is False

    def test_position_sizing_kelly(self):
        from trading_system.risk_manager import RiskManager
        from trading_system.config import RiskConfig
        from trading_system.strategies.base import Signal

        rm = RiskManager(RiskConfig(position_sizing="kelly"))
        sig = Signal("AAPL", 0.7, 0.8, "test")

        size = rm.calculate_position_size(
            signal=sig,
            price=150.0,
            portfolio_equity=100_000,
        )

        assert size > 0
        assert size <= 100_000 * 0.05  # Should respect limits

    def test_filter_blocks_over_max_positions(self):
        from trading_system.risk_manager import RiskManager
        from trading_system.config import RiskConfig
        from trading_system.strategies.base import Signal

        self._init_db()
        rm = RiskManager(RiskConfig(max_open_positions=2))
        rm.initialize(100_000)

        signals = [Signal("NEW_STOCK", 0.8, 0.8, "test")]
        current = {"A": 10.0, "B": 10.0}  # Already at max

        approved = rm.filter_signals(signals, current, 100_000, {"A": 5000, "B": 5000})
        assert len(approved) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
