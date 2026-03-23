"""Trend following strategy using dual moving average crossovers.

Combines short-term and long-term moving averages with ADX trend
strength confirmation.
"""

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import StrategyTrendConfig
from trading_system.strategies.base import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"

    def __init__(self, config: StrategyTrendConfig):
        self.config = config

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        signals = []

        for sym, df in data.items():
            if df.empty or len(df) < self.config.signal_ma + 5:
                continue

            try:
                signal = self._evaluate(sym, df, current_positions.get(sym, 0))
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Trend following failed for {sym}: {e}")

        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, position: float) -> Signal | None:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = last["close"]
        atr = last.get("atr_14", 0)

        fast_ma_col = f"sma_{self.config.fast_ma}"
        slow_ma_col = f"sma_{self.config.slow_ma}"
        signal_ma_col = f"sma_{self.config.signal_ma}"

        # Ensure MAs exist
        for col in [fast_ma_col, slow_ma_col, signal_ma_col]:
            if col not in df.columns or pd.isna(last.get(col)):
                return None

        fast_now = last[fast_ma_col]
        slow_now = last[slow_ma_col]
        signal_now = last[signal_ma_col]
        fast_prev = prev.get(fast_ma_col, fast_now)
        slow_prev = prev.get(slow_ma_col, slow_now)

        # ADX filter — only trade when trend is strong enough
        adx = last.get("adx", 0)
        if pd.isna(adx):
            adx = 0

        # Bullish crossover: fast crosses above slow, price above 200 SMA
        bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
        bearish_cross = fast_prev >= slow_prev and fast_now < slow_now
        above_signal = close > signal_now
        below_signal = close < signal_now

        if bullish_cross and above_signal and adx > 20:
            direction = np.clip(0.5 + (adx - 20) / 60, 0.3, 1.0)
            confidence = np.clip(0.4 + (adx - 20) / 80, 0.3, 0.85)

            stop = close - 2 * atr if atr > 0 else close * 0.95
            tp = close + 4 * atr if atr > 0 else close * 1.12

            return Signal(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                strategy=self.name,
                stop_loss=stop,
                take_profit=tp,
                metadata={"crossover": "bullish", "adx": adx, "above_200sma": above_signal},
            )

        elif bearish_cross and below_signal and adx > 20:
            direction = -np.clip(0.5 + (adx - 20) / 60, 0.3, 1.0)
            confidence = np.clip(0.4 + (adx - 20) / 80, 0.3, 0.85)

            return Signal(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                strategy=self.name,
                metadata={"crossover": "bearish", "adx": adx, "below_200sma": below_signal},
            )

        # Trend continuation — existing position with confirmed trend
        elif position > 0 and fast_now > slow_now and close > signal_now and adx > 25:
            # Trend strengthening — mild add signal
            return Signal(
                symbol=sym,
                direction=0.2,
                confidence=0.4,
                strategy=self.name,
                metadata={"action": "trend_continuation", "adx": adx},
            )

        return None
