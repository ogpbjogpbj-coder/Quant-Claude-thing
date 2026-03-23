"""Signal aggregation across multiple strategies.

Combines signals from all strategies using configurable weights,
resolves conflicts, and produces a final ranked list of trade ideas.
"""

from collections import defaultdict

import numpy as np
from loguru import logger

from trading_system.config import StrategiesConfig
from trading_system.strategies.base import Signal


class SignalAggregator:
    """Combines signals from multiple strategies into consensus signals."""

    def __init__(self, config: StrategiesConfig):
        self.config = config
        self.weights = {
            "momentum": config.momentum.weight,
            "mean_reversion": config.mean_reversion.weight,
            "ml_ensemble": config.ml_ensemble.weight,
            "volatility_breakout": config.volatility_breakout.weight,
            "trend_following": config.trend_following.weight,
            "pairs_trading": config.pairs_trading.weight,
            "sentiment": config.sentiment.weight,
            "adaptive": config.adaptive.weight,
            "catalyst": config.catalyst.weight,
        }
        self._regime_adjustments: dict[str, float] = {}

    def set_regime_adjustments(self, adjustments: dict[str, float]) -> None:
        """Apply regime-based weight adjustments."""
        self._regime_adjustments = adjustments
        logger.info(f"Regime weight adjustments applied: {adjustments}")

    def aggregate(self, all_signals: list[Signal]) -> list[Signal]:
        """Aggregate signals per symbol into weighted consensus signals.

        Multiple strategies may signal on the same symbol. This method
        combines them into a single signal per symbol.
        """
        if not all_signals:
            return []

        # Group signals by symbol
        by_symbol: dict[str, list[Signal]] = defaultdict(list)
        for sig in all_signals:
            by_symbol[sig.symbol].append(sig)

        aggregated = []
        for symbol, signals in by_symbol.items():
            agg = self._aggregate_symbol(symbol, signals)
            if agg is not None:
                aggregated.append(agg)

        # Sort by strength (strongest first)
        aggregated.sort(key=lambda s: s.strength, reverse=True)

        logger.info(
            f"Aggregated {len(all_signals)} raw signals into "
            f"{len(aggregated)} consensus signals"
        )

        return aggregated

    def _aggregate_symbol(self, symbol: str, signals: list[Signal]) -> Signal | None:
        """Combine multiple strategy signals for one symbol."""
        if not signals:
            return None

        # Weighted direction and confidence
        total_weight = 0
        weighted_direction = 0
        weighted_confidence = 0
        best_stop = None
        best_tp = None
        strategies_involved = []
        all_metadata = {}

        for sig in signals:
            base_w = self.weights.get(sig.strategy, 0.1)
            regime_mult = self._regime_adjustments.get(sig.strategy, 1.0)
            w = base_w * regime_mult
            total_weight += w
            weighted_direction += sig.direction * w * sig.confidence
            weighted_confidence += sig.confidence * w
            strategies_involved.append(sig.strategy)

            # Use the most conservative stop loss (highest for longs)
            if sig.stop_loss is not None:
                if best_stop is None:
                    best_stop = sig.stop_loss
                else:
                    best_stop = max(best_stop, sig.stop_loss)  # Tighter stop

            if sig.take_profit is not None:
                if best_tp is None:
                    best_tp = sig.take_profit
                else:
                    best_tp = min(best_tp, sig.take_profit)  # More conservative target

            all_metadata[sig.strategy] = sig.metadata

        if total_weight == 0:
            return None

        direction = weighted_direction / total_weight
        confidence = weighted_confidence / total_weight

        # Agreement bonus: if multiple strategies agree, boost confidence
        n_strategies = len(signals)
        directions = [s.direction for s in signals]
        all_agree = all(d > 0 for d in directions) or all(d < 0 for d in directions)

        if n_strategies >= 2 and all_agree:
            confidence *= 1.0 + 0.1 * (n_strategies - 1)  # 10% boost per agreeing strategy
        elif n_strategies >= 2 and not all_agree:
            # Conflicting signals — reduce confidence
            confidence *= 0.7

        confidence = np.clip(confidence, 0.0, 0.95)
        direction = np.clip(direction, -1.0, 1.0)

        # Require minimum direction strength and multi-strategy agreement
        if abs(direction) < 0.20:
            return None
        if n_strategies < 2:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strategy=f"consensus({','.join(strategies_involved)})",
            stop_loss=best_stop,
            take_profit=best_tp,
            metadata={
                "contributing_strategies": strategies_involved,
                "n_strategies": n_strategies,
                "all_agree": all_agree,
                "raw_signals": all_metadata,
            },
        )
