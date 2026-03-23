"""Multi-timeframe momentum strategy.

Ranks stocks by momentum across multiple lookback periods, buys winners
and sells losers. Uses RSI and volume confirmation.
"""

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import StrategyMomentumConfig
from trading_system.strategies.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(self, config: StrategyMomentumConfig):
        self.config = config
        self.lookback_periods = config.lookback_periods  # e.g. [5, 10, 21, 63]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        signals = []
        momentum_scores = {}

        for sym, df in data.items():
            if df.empty or len(df) < max(self.lookback_periods) + 5:
                continue

            try:
                score = self._compute_momentum_score(df)
                momentum_scores[sym] = score
            except Exception as e:
                logger.debug(f"Momentum calc failed for {sym}: {e}")

        if not momentum_scores:
            return signals

        # Rank by composite momentum score
        sorted_scores = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_scores)

        for rank, (sym, score) in enumerate(sorted_scores):
            df = data[sym]
            last = df.iloc[-1]

            # Top quintile = buy, bottom quintile = sell
            percentile = 1 - (rank / n)

            # Volume confirmation: skip if volume is anemic
            vol_ratio = last.get("volume_ratio", 1.0)
            if pd.isna(vol_ratio):
                vol_ratio = 1.0
            if vol_ratio < 0.5:
                continue

            # RSI confirmation: avoid extreme RSI for new entries
            rsi = last.get("rsi_14", 50)
            if pd.isna(rsi):
                rsi = 50

            atr = last.get("atr_14", 0)
            close = last["close"]

            if percentile >= 0.8 and score > 0 and rsi < 75:
                # Strong momentum buy
                direction = min(score / 0.15, 1.0)  # Normalize
                confidence = percentile * (1 - abs(rsi - 50) / 50) * min(vol_ratio, 2) / 2
                confidence = np.clip(confidence, 0.1, 0.95)

                stop = close - 2 * atr if atr > 0 else close * 0.95
                tp = close + 4 * atr if atr > 0 else close * 1.10

                signals.append(Signal(
                    symbol=sym,
                    direction=np.clip(direction, 0.1, 1.0),
                    confidence=confidence,
                    strategy=self.name,
                    stop_loss=stop,
                    take_profit=tp,
                    metadata={"momentum_score": score, "rank_percentile": percentile},
                ))

            elif percentile <= 0.2 and score < 0 and rsi > 25:
                # Weak momentum — sell signal (close longs)
                direction = max(score / 0.15, -1.0)
                confidence = (1 - percentile) * 0.7
                confidence = np.clip(confidence, 0.1, 0.9)

                signals.append(Signal(
                    symbol=sym,
                    direction=np.clip(direction, -1.0, -0.1),
                    confidence=confidence,
                    strategy=self.name,
                    metadata={"momentum_score": score, "rank_percentile": percentile},
                ))

        return signals

    def _compute_momentum_score(self, df: pd.DataFrame) -> float:
        """Composite momentum = weighted average of multi-period returns."""
        close = df["close"]
        weights = [0.1, 0.2, 0.3, 0.4]  # More weight on longer periods
        score = 0.0

        for period, weight in zip(self.lookback_periods, weights):
            if len(close) > period:
                ret = (close.iloc[-1] / close.iloc[-period]) - 1
                score += weight * ret

        # Adjust by recent acceleration
        if len(close) > 10:
            recent = (close.iloc[-1] / close.iloc[-5]) - 1
            older = (close.iloc[-5] / close.iloc[-10]) - 1
            acceleration = recent - older
            score += 0.2 * acceleration

        return score
