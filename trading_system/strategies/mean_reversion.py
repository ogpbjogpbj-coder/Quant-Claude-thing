"""Mean reversion strategy using Bollinger Band z-scores.

Identifies overbought/oversold conditions and trades reversals with
volume and RSI confirmation.
"""

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import StrategyMeanReversionConfig
from trading_system.strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def __init__(self, config: StrategyMeanReversionConfig):
        self.config = config

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        signals = []

        for sym, df in data.items():
            if df.empty or len(df) < self.config.lookback + 10:
                continue

            try:
                signal = self._evaluate(sym, df, current_positions.get(sym, 0))
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Mean reversion failed for {sym}: {e}")

        return signals

    def _evaluate(self, sym: str, df: pd.DataFrame, position: float) -> Signal | None:
        last = df.iloc[-1]
        zscore = last.get("price_zscore", 0)
        rsi = last.get("rsi_14", 50)
        vol_ratio = last.get("volume_ratio", 1.0)
        atr = last.get("atr_14", 0)
        close = last["close"]
        bb_pos = last.get("bb_position", 0.5)

        if pd.isna(zscore) or pd.isna(rsi):
            return None

        # Check trending regime — skip mean reversion in strong trends
        adx = last.get("adx", 0)
        if not pd.isna(adx) and adx > 35:
            return None  # Market trending too strongly

        # Oversold — buy signal
        if zscore <= -self.config.z_score_entry and rsi < 35:
            # Deeper oversold = stronger signal
            direction = min(abs(zscore) / 3.0, 1.0)
            confidence = self._confidence(zscore, rsi, vol_ratio, is_buy=True)

            stop = close - 2.5 * atr if atr > 0 else close * 0.96
            tp = last.get("bb_mid", close * 1.03)

            return Signal(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                strategy=self.name,
                stop_loss=stop,
                take_profit=tp,
                metadata={"zscore": zscore, "rsi": rsi, "bb_position": bb_pos},
            )

        # Overbought — sell/exit signal
        elif zscore >= self.config.z_score_entry and rsi > 65:
            direction = -min(abs(zscore) / 3.0, 1.0)
            confidence = self._confidence(zscore, rsi, vol_ratio, is_buy=False)

            return Signal(
                symbol=sym,
                direction=direction,
                confidence=confidence,
                strategy=self.name,
                metadata={"zscore": zscore, "rsi": rsi, "bb_position": bb_pos},
            )

        # Exit signal for existing positions: z-score reverted
        elif position > 0 and abs(zscore) <= self.config.z_score_exit:
            return Signal(
                symbol=sym,
                direction=-0.3,  # Mild sell to close
                confidence=0.6,
                strategy=self.name,
                metadata={"zscore": zscore, "action": "exit_reversion_complete"},
            )

        return None

    def _confidence(self, zscore: float, rsi: float, vol_ratio: float, is_buy: bool) -> float:
        """Confidence based on extremity + volume confirmation."""
        extremity = min(abs(zscore) / 3.5, 1.0)

        if is_buy:
            rsi_factor = max(0, (40 - rsi) / 40)
        else:
            rsi_factor = max(0, (rsi - 60) / 40)

        vol_factor = np.clip(vol_ratio / 2.0, 0.3, 1.0)

        conf = 0.4 * extremity + 0.3 * rsi_factor + 0.3 * vol_factor
        return np.clip(conf, 0.1, 0.9)
