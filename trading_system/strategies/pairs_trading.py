"""Cointegration-based pairs trading strategy.

Identifies cointegrated pairs from the trading universe, computes spreads
using OLS hedge ratios, and generates mean-reversion signals when the
z-scored spread deviates beyond configurable thresholds.
"""

import time
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
from statsmodels.tools.tools import add_constant

from trading_system.strategies.base import BaseStrategy, Signal


class StrategyPairsConfig(BaseModel):
    """Configuration for the pairs trading strategy."""

    enabled: bool = True
    weight: float = 0.15
    lookback: int = 60
    z_score_entry: float = 2.0
    z_score_exit: float = 0.5
    min_half_life: int = 5
    max_half_life: int = 60
    max_pairs: int = 10


class _PairInfo:
    """Internal container for a validated cointegrated pair."""

    __slots__ = (
        "sym_a",
        "sym_b",
        "hedge_ratio",
        "half_life",
        "p_value",
        "spread_mean",
        "spread_std",
    )

    def __init__(
        self,
        sym_a: str,
        sym_b: str,
        hedge_ratio: float,
        half_life: float,
        p_value: float,
        spread_mean: float,
        spread_std: float,
    ):
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.hedge_ratio = hedge_ratio
        self.half_life = half_life
        self.p_value = p_value
        self.spread_mean = spread_mean
        self.spread_std = spread_std


class PairsTradingStrategy(BaseStrategy):
    """Pairs trading strategy driven by Engle-Granger cointegration tests."""

    name = "pairs_trading"

    def __init__(self, config: StrategyPairsConfig):
        self.config = config
        self._pairs: list[_PairInfo] = []
        self._last_selection_time: float = 0.0
        # Recompute pairs at most once per day (86400 seconds).
        self._selection_interval: float = 86400.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        if not self.config.enabled:
            return []

        # Refresh pair selection if stale.
        now = time.time()
        if now - self._last_selection_time >= self._selection_interval:
            self._select_pairs(data)
            self._last_selection_time = now

        signals: list[Signal] = []
        for pair in self._pairs:
            try:
                pair_signals = self._generate_pair_signals(
                    pair, data, current_positions
                )
                signals.extend(pair_signals)
            except Exception as e:
                logger.debug(
                    f"Pairs trading signal generation failed for "
                    f"{pair.sym_a}/{pair.sym_b}: {e}"
                )

        return signals

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def _select_pairs(self, data: dict[str, pd.DataFrame]) -> None:
        """Run cointegration tests on all symbol pairs and cache the results."""
        symbols = [
            sym
            for sym, df in data.items()
            if not df.empty and len(df) >= self.config.lookback
        ]

        if len(symbols) < 2:
            logger.debug("Pairs trading: fewer than 2 eligible symbols, skipping.")
            self._pairs = []
            return

        candidates: list[_PairInfo] = []

        for sym_a, sym_b in combinations(symbols, 2):
            try:
                pair_info = self._test_pair(sym_a, sym_b, data[sym_a], data[sym_b])
                if pair_info is not None:
                    candidates.append(pair_info)
            except Exception as e:
                logger.debug(
                    f"Cointegration test failed for {sym_a}/{sym_b}: {e}"
                )

        # Rank by cointegration strength (lowest p-value first).
        candidates.sort(key=lambda p: p.p_value)
        self._pairs = candidates[: self.config.max_pairs]

        if self._pairs:
            pair_names = [f"{p.sym_a}/{p.sym_b}" for p in self._pairs]
            logger.info(
                f"Pairs trading selected {len(self._pairs)} pairs: {pair_names}"
            )
        else:
            logger.debug("Pairs trading: no cointegrated pairs found.")

    def _test_pair(
        self,
        sym_a: str,
        sym_b: str,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> _PairInfo | None:
        """Test a single pair for cointegration and compute spread statistics."""
        lookback = self.config.lookback

        close_a = df_a["close"].iloc[-lookback:].values.astype(float)
        close_b = df_b["close"].iloc[-lookback:].values.astype(float)

        if len(close_a) < lookback or len(close_b) < lookback:
            return None

        # Engle-Granger cointegration test.
        _, p_value, _ = coint(close_a, close_b)
        if p_value >= 0.05:
            return None

        # OLS hedge ratio: close_a = beta * close_b + alpha + eps
        x = add_constant(close_b)
        model = OLS(close_a, x).fit()
        hedge_ratio: float = float(model.params[1])

        # Spread and its half-life of mean reversion.
        spread = close_a - hedge_ratio * close_b

        half_life = self._compute_half_life(spread)
        if half_life is None:
            return None
        if not (self.config.min_half_life <= half_life <= self.config.max_half_life):
            return None

        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread, ddof=1))
        if spread_std < 1e-10:
            return None

        return _PairInfo(
            sym_a=sym_a,
            sym_b=sym_b,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            p_value=p_value,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

    @staticmethod
    def _compute_half_life(spread: np.ndarray) -> float | None:
        """Estimate mean-reversion half-life via OLS on spread changes.

        Model: delta_spread(t) = lambda * spread(t-1) + eps
        Half-life = -ln(2) / lambda
        """
        lagged = spread[:-1]
        delta = np.diff(spread)

        if len(lagged) < 3:
            return None

        x = add_constant(lagged)
        model = OLS(delta, x).fit()
        lam = model.params[1]

        if lam >= 0:
            # No mean reversion detected.
            return None

        half_life = -np.log(2) / lam
        return float(half_life)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_pair_signals(
        self,
        pair: _PairInfo,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        """Generate entry/exit signals for a single pair."""
        df_a = data.get(pair.sym_a)
        df_b = data.get(pair.sym_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            return []

        lookback = self.config.lookback
        close_a = df_a["close"].iloc[-lookback:].values.astype(float)
        close_b = df_b["close"].iloc[-lookback:].values.astype(float)

        if len(close_a) < lookback or len(close_b) < lookback:
            return []

        # Recompute spread statistics over the lookback window.
        spread = close_a - pair.hedge_ratio * close_b
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread, ddof=1))
        if spread_std < 1e-10:
            return []

        current_spread = spread[-1]
        z_score = (current_spread - spread_mean) / spread_std

        pos_a = current_positions.get(pair.sym_a, 0.0)
        pos_b = current_positions.get(pair.sym_b, 0.0)
        has_position = abs(pos_a) > 1e-9 or abs(pos_b) > 1e-9

        signals: list[Signal] = []

        # Stop-loss: z-score exceeded 4.0, force close.
        z_stop = 4.0
        if has_position and abs(z_score) > z_stop:
            logger.info(
                f"Pairs stop-loss triggered for {pair.sym_a}/{pair.sym_b}, "
                f"z={z_score:.2f}"
            )
            signals.extend(self._exit_signals(pair, z_score))
            return signals

        # Exit: z-score reverted within exit band.
        if has_position and abs(z_score) < self.config.z_score_exit:
            logger.info(
                f"Pairs exit signal for {pair.sym_a}/{pair.sym_b}, z={z_score:.2f}"
            )
            signals.extend(self._exit_signals(pair, z_score))
            return signals

        # Entry: z-score beyond entry threshold.
        if not has_position and abs(z_score) >= self.config.z_score_entry:
            # Clamp direction strength to [-1, 1] proportional to z-score.
            raw_strength = min(abs(z_score) / z_stop, 1.0)
            confidence = min(abs(z_score) / z_stop, 1.0)

            if z_score > 0:
                # Spread is above mean: short A, long B.
                dir_a = -raw_strength
                dir_b = raw_strength
            else:
                # Spread is below mean: long A, short B.
                dir_a = raw_strength
                dir_b = -raw_strength

            meta = {
                "pair": f"{pair.sym_a}/{pair.sym_b}",
                "z_score": round(z_score, 4),
                "hedge_ratio": round(pair.hedge_ratio, 4),
                "half_life": round(pair.half_life, 2),
                "p_value": round(pair.p_value, 4),
            }

            signals.append(
                Signal(
                    symbol=pair.sym_a,
                    direction=float(np.clip(dir_a, -1.0, 1.0)),
                    confidence=confidence,
                    strategy=self.name,
                    stop_loss=z_stop,
                    metadata=meta,
                )
            )
            signals.append(
                Signal(
                    symbol=pair.sym_b,
                    direction=float(np.clip(dir_b, -1.0, 1.0)),
                    confidence=confidence,
                    strategy=self.name,
                    stop_loss=z_stop,
                    metadata=meta,
                )
            )

            logger.info(
                f"Pairs entry signal for {pair.sym_a}/{pair.sym_b}, "
                f"z={z_score:.2f}, hedge={pair.hedge_ratio:.4f}"
            )

        return signals

    def _exit_signals(self, pair: _PairInfo, z_score: float) -> list[Signal]:
        """Generate zero-direction signals to close both legs of a pair."""
        meta = {
            "pair": f"{pair.sym_a}/{pair.sym_b}",
            "z_score": round(z_score, 4),
            "action": "exit",
        }
        return [
            Signal(
                symbol=pair.sym_a,
                direction=0.0,
                confidence=1.0,
                strategy=self.name,
                metadata=meta,
            ),
            Signal(
                symbol=pair.sym_b,
                direction=0.0,
                confidence=1.0,
                strategy=self.name,
                metadata=meta,
            ),
        ]
