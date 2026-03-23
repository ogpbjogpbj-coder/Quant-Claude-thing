"""Signal decay tracking module.

Monitors the rolling performance of each strategy's signals and
auto-adjusts weights so that deteriorating strategies are dampened
and strong performers are amplified.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger


@dataclass
class _SignalRecord:
    """Single signal observation."""

    signal_strength: float
    actual_return: float


class SignalDecayTracker:
    """Track rolling signal quality per strategy and adjust weights."""

    # Thresholds
    _DEMOTION_THRESHOLD: float = 0.35
    _DEMOTION_MIN_SIGNALS: int = 20
    _ZERO_OUT_THRESHOLD: float = 0.3
    _HALVE_THRESHOLD: float = 0.4
    _BOOST_THRESHOLD: float = 0.6
    _BOOST_MULTIPLIER: float = 1.3
    _MIN_SIGNALS_FOR_ADJUSTMENT: int = 10

    def __init__(
        self,
        base_weights: dict[str, float],
        window: int = 50,
    ) -> None:
        """Initialise the decay tracker.

        Parameters
        ----------
        base_weights:
            Mapping of strategy name to its base weight,
            e.g. ``{"momentum": 0.25, "mean_reversion": 0.20, ...}``.
        window:
            Rolling window size for performance calculation.
        """
        if not base_weights:
            raise ValueError("base_weights must be a non-empty dict")
        if window < 1:
            raise ValueError("window must be >= 1")

        self._base_weights: dict[str, float] = dict(base_weights)
        self._window: int = window
        self._history: dict[str, deque[_SignalRecord]] = {
            name: deque(maxlen=window) for name in base_weights
        }

        logger.info(
            "SignalDecayTracker initialised with {} strategies, window={}",
            len(base_weights),
            window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal_outcome(
        self,
        strategy: str,
        signal_strength: float,
        actual_return: float,
    ) -> None:
        """Record a signal and its realised outcome.

        Parameters
        ----------
        strategy:
            Name of the strategy that produced the signal.
        signal_strength:
            Predicted signal strength (sign encodes direction).
        actual_return:
            Actual return observed after the signal.
        """
        if strategy not in self._history:
            logger.warning(
                "Unknown strategy '{}' – adding with base weight 0.0",
                strategy,
            )
            self._base_weights[strategy] = 0.0
            self._history[strategy] = deque(maxlen=self._window)

        self._history[strategy].append(
            _SignalRecord(signal_strength=signal_strength, actual_return=actual_return)
        )

    def get_adjusted_weights(self) -> dict[str, float]:
        """Return strategy weights adjusted for recent signal quality.

        If a strategy has fewer than ``_MIN_SIGNALS_FOR_ADJUSTMENT``
        observations its base weight is used (cold-start behaviour).

        Returns
        -------
        dict[str, float]
            Weights normalised to sum to 1.0.
        """
        raw_weights: dict[str, float] = {}

        for strategy, base_w in self._base_weights.items():
            records = self._history.get(strategy)

            if records is None or len(records) < self._MIN_SIGNALS_FOR_ADJUSTMENT:
                raw_weights[strategy] = base_w
                continue

            score = self._compute_score(records)

            if score < self._ZERO_OUT_THRESHOLD:
                adjusted = 0.0
                logger.warning(
                    "Strategy '{}' zeroed out (score={:.3f})", strategy, score
                )
            elif score < self._HALVE_THRESHOLD:
                adjusted = base_w * 0.5
                logger.info(
                    "Strategy '{}' halved (score={:.3f})", strategy, score
                )
            elif score > self._BOOST_THRESHOLD:
                adjusted = base_w * self._BOOST_MULTIPLIER
                logger.info(
                    "Strategy '{}' boosted (score={:.3f})", strategy, score
                )
            else:
                adjusted = base_w

            raw_weights[strategy] = adjusted

        return self._normalise(raw_weights)

    def get_strategy_stats(self) -> dict[str, dict]:
        """Return per-strategy diagnostic statistics.

        Returns
        -------
        dict[str, dict]
            Keys are strategy names.  Each value dict contains:
            ``hit_rate``, ``IC``, ``signal_count``, ``avg_return``,
            ``score``, ``weight_adjustment``.
        """
        adjusted = self.get_adjusted_weights()
        stats: dict[str, dict] = {}

        for strategy in self._base_weights:
            records = self._history.get(strategy)
            count = len(records) if records else 0

            if count < self._MIN_SIGNALS_FOR_ADJUSTMENT:
                stats[strategy] = {
                    "hit_rate": None,
                    "IC": None,
                    "signal_count": count,
                    "avg_return": None,
                    "score": None,
                    "weight_adjustment": "cold_start",
                }
                continue

            hit_rate = self._hit_rate(records)
            ic = self._information_coefficient(records)
            score = 0.5 * hit_rate + 0.5 * (ic + 1.0) / 2.0
            returns = [r.actual_return for r in records]

            if score < self._ZERO_OUT_THRESHOLD:
                adj_label = "zeroed"
            elif score < self._HALVE_THRESHOLD:
                adj_label = "halved"
            elif score > self._BOOST_THRESHOLD:
                adj_label = "boosted"
            else:
                adj_label = "unchanged"

            stats[strategy] = {
                "hit_rate": round(hit_rate, 4),
                "IC": round(ic, 4),
                "signal_count": count,
                "avg_return": round(float(np.mean(returns)), 6),
                "score": round(score, 4),
                "weight_adjustment": adj_label,
            }

        return stats

    def should_demote(self, strategy: str) -> bool:
        """Check whether a strategy should be demoted.

        A strategy is flagged for demotion when it has accumulated at
        least ``_DEMOTION_MIN_SIGNALS`` observations **and** its
        rolling score is below ``_DEMOTION_THRESHOLD``.

        Parameters
        ----------
        strategy:
            Name of the strategy to evaluate.

        Returns
        -------
        bool
        """
        records = self._history.get(strategy)
        if records is None or len(records) < self._DEMOTION_MIN_SIGNALS:
            return False

        score = self._compute_score(records)
        return score < self._DEMOTION_THRESHOLD

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_rate(records: deque[_SignalRecord]) -> float:
        """Fraction of signals whose direction matched actual return."""
        if not records:
            return 0.0
        hits = sum(
            1
            for r in records
            if (r.signal_strength > 0 and r.actual_return > 0)
            or (r.signal_strength < 0 and r.actual_return < 0)
        )
        return hits / len(records)

    @staticmethod
    def _information_coefficient(records: deque[_SignalRecord]) -> float:
        """Pearson correlation between signal strength and actual return."""
        if len(records) < 2:
            return 0.0
        signals = np.array([r.signal_strength for r in records])
        returns = np.array([r.actual_return for r in records])

        std_s = np.std(signals)
        std_r = np.std(returns)
        if std_s == 0 or std_r == 0:
            return 0.0

        corr_matrix = np.corrcoef(signals, returns)
        ic = float(corr_matrix[0, 1])
        # Guard against NaN from degenerate inputs
        if np.isnan(ic):
            return 0.0
        return ic

    def _compute_score(self, records: deque[_SignalRecord]) -> float:
        """Combined score from hit rate and IC."""
        hit = self._hit_rate(records)
        ic = self._information_coefficient(records)
        return 0.5 * hit + 0.5 * (ic + 1.0) / 2.0

    @staticmethod
    def _normalise(weights: dict[str, float]) -> dict[str, float]:
        """Normalise weights to sum to 1.0.

        If all weights are zero, distribute equally.
        """
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}
