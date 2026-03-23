"""Volatility breakout strategy.

Enters when price breaks above/below the ATR-based channel with
volume confirmation. Good at catching sudden moves.
"""

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import StrategyVolBreakoutConfig
from trading_system.strategies.base import BaseStrategy, Signal


class VolatilityBreakoutStrategy(BaseStrategy):
    name = "volatility_breakout"

    def __init__(self, config: StrategyVolBreakoutConfig):
        self.config = config

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        signals = []

        for sym, df in data.items():
            if df.empty or len(df) < self.config.lookback + 5:
                continue

            try:
                signal = self._evaluate(sym, df, current_positions.get(sym, 0))
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Vol breakout failed for {sym}: {e}")

        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, position: float) -> Signal | None:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = last["close"]
        prev_close = prev["close"]

        atr = last.get("atr_14", 0)
        vol_ratio = last.get("volume_ratio", 1.0)

        if pd.isna(atr) or atr <= 0:
            return None

        # Compute breakout levels from yesterday
        upper = prev_close + self.config.atr_multiplier * atr
        lower = prev_close - self.config.atr_multiplier * atr

        # Volume must confirm (at least 1.2x average)
        if pd.isna(vol_ratio) or vol_ratio < 1.2:
            return None

        if close > upper:
            # Bullish breakout
            excess = (close - upper) / atr
            direction = min(0.5 + excess * 0.25, 1.0)
            confidence = np.clip(0.5 + (vol_ratio - 1.2) * 0.15 + excess * 0.1, 0.3, 0.85)

            stop = close - 2 * atr
            tp = close + 3 * atr

            return Signal(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                strategy=self.name,
                stop_loss=stop,
                take_profit=tp,
                metadata={"breakout": "upper", "excess_atr": excess, "vol_ratio": vol_ratio},
            )

        elif close < lower:
            # Bearish breakout (sell/short signal)
            excess = (lower - close) / atr
            direction = -min(0.5 + excess * 0.25, 1.0)
            confidence = np.clip(0.5 + (vol_ratio - 1.2) * 0.15 + excess * 0.1, 0.3, 0.85)

            return Signal(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                strategy=self.name,
                metadata={"breakout": "lower", "excess_atr": excess, "vol_ratio": vol_ratio},
            )

        return None
