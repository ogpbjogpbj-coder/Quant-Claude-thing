"""
Regime detection module.

Analyzes market data across multiple dimensions (volatility, trend, correlation,
crisis indicators) to classify the current market regime and produce recommended
strategy weight adjustments and position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class RegimeType(Enum):
    """Enumeration of detectable market regimes."""

    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    MEAN_REVERTING = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    CRISIS = auto()


@dataclass
class RegimeState:
    """Container for a complete regime detection result."""

    regime: RegimeType
    confidence: float
    volatility_regime: str  # "low", "normal", "high", "extreme"
    trend_strength: float  # -1 to 1
    correlation_level: float  # 0 to 1
    strategy_weights: dict[str, float] = field(default_factory=dict)
    position_scale: float = 1.0


# ---------------------------------------------------------------------------
# Strategy weight presets per regime
# ---------------------------------------------------------------------------

_BASE_STRATEGIES = [
    "momentum",
    "mean_reversion",
    "ml_ensemble",
    "volatility_breakout",
    "trend_following",
    "sector_rotation",
]

_REGIME_WEIGHT_ADJUSTMENTS: dict[RegimeType, dict[str, float]] = {
    RegimeType.TRENDING_UP: {
        "momentum": 1.5,
        "mean_reversion": 0.5,
        "ml_ensemble": 1.0,
        "volatility_breakout": 1.0,
        "trend_following": 1.5,
        "sector_rotation": 1.3,  # Sector rotation works well in trends
    },
    RegimeType.TRENDING_DOWN: {
        "momentum": 0.7,
        "mean_reversion": 1.3,
        "ml_ensemble": 1.0,
        "volatility_breakout": 1.0,
        "trend_following": 1.0,
        "sector_rotation": 1.2,  # Defensive rotation valuable
    },
    RegimeType.MEAN_REVERTING: {
        "momentum": 0.5,
        "mean_reversion": 1.8,
        "ml_ensemble": 1.0,
        "volatility_breakout": 1.0,
        "trend_following": 0.5,
        "sector_rotation": 0.7,  # Less useful in choppy markets
    },
    RegimeType.HIGH_VOLATILITY: {
        "momentum": 0.8,
        "mean_reversion": 0.8,
        "ml_ensemble": 0.9,
        "volatility_breakout": 1.5,
        "trend_following": 0.8,
        "sector_rotation": 0.8,
    },
    RegimeType.LOW_VOLATILITY: {
        "momentum": 1.0,
        "mean_reversion": 1.3,
        "ml_ensemble": 1.0,
        "volatility_breakout": 0.5,
        "trend_following": 1.0,
        "sector_rotation": 1.0,
    },
    RegimeType.CRISIS: {
        "momentum": 0.3,
        "mean_reversion": 0.3,
        "ml_ensemble": 0.4,
        "volatility_breakout": 0.3,
        "trend_following": 0.3,
        "sector_rotation": 0.5,  # Defensive rotation still useful in crisis
    },
}

_REGIME_POSITION_SCALE: dict[RegimeType, float] = {
    RegimeType.TRENDING_UP: 1.0,
    RegimeType.TRENDING_DOWN: 0.8,
    RegimeType.MEAN_REVERTING: 0.9,
    RegimeType.HIGH_VOLATILITY: 0.7,
    RegimeType.LOW_VOLATILITY: 1.0,
    RegimeType.CRISIS: 0.3,
}


def _default_regime_state() -> RegimeState:
    """Return a neutral / default regime when detection fails."""
    return RegimeState(
        regime=RegimeType.MEAN_REVERTING,
        confidence=0.0,
        volatility_regime="normal",
        trend_strength=0.0,
        correlation_level=0.5,
        strategy_weights={s: 1.0 for s in _BASE_STRATEGIES},
        position_scale=0.8,
    )


class RegimeDetector:
    """Detect the prevailing market regime from enriched price data.

    Parameters
    ----------
    vol_lookback : int
        Window (trading days) for realized-volatility calculation.
    vol_history_lookback : int
        Longer window used to compute the historical average of realized vol.
    trend_short_window : int
        Short moving-average window for trend / slope estimation.
    trend_long_window : int
        Long moving-average window (also used for breadth: % above this SMA).
    adx_window : int
        Window for the ADX (Average Directional Index) approximation.
    correlation_lookback : int
        Rolling window for pairwise return correlations.
    crisis_vol_multiplier : float
        Current vol must exceed historical average by this multiple to
        trigger a crisis signal on the volatility axis.
    crisis_corr_threshold : float
        Average pairwise correlation above this level contributes to a
        crisis signal.
    crisis_drawdown_threshold : float
        Drawdown (as a negative fraction, e.g. -0.10) below which the
        drawdown axis contributes to a crisis signal.
    spy_ticker : str
        Ticker used as the VIX proxy (via its realized vol).
    """

    def __init__(
        self,
        vol_lookback: int = 21,
        vol_history_lookback: int = 63,
        trend_short_window: int = 50,
        trend_long_window: int = 200,
        adx_window: int = 14,
        correlation_lookback: int = 63,
        crisis_vol_multiplier: float = 2.0,
        crisis_corr_threshold: float = 0.75,
        crisis_drawdown_threshold: float = -0.10,
        spy_ticker: str = "SPY",
    ) -> None:
        self.vol_lookback = vol_lookback
        self.vol_history_lookback = vol_history_lookback
        self.trend_short_window = trend_short_window
        self.trend_long_window = trend_long_window
        self.adx_window = adx_window
        self.correlation_lookback = correlation_lookback
        self.crisis_vol_multiplier = crisis_vol_multiplier
        self.crisis_corr_threshold = crisis_corr_threshold
        self.crisis_drawdown_threshold = crisis_drawdown_threshold
        self.spy_ticker = spy_ticker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, enriched_data: dict[str, pd.DataFrame]) -> RegimeState:
        """Analyse *enriched_data* and return the current :class:`RegimeState`.

        Parameters
        ----------
        enriched_data:
            Mapping of ``{ticker: DataFrame}`` where each DataFrame has at
            least a ``"close"`` column indexed by date / datetime.

        Returns
        -------
        RegimeState
            The detected regime together with confidence, diagnostics, and
            recommended strategy weights.
        """
        try:
            if not enriched_data:
                logger.warning("Regime detector received empty data; returning default regime.")
                return _default_regime_state()

            # Build a universe-wide close-price matrix and returns matrix.
            close_matrix = self._build_close_matrix(enriched_data)
            if close_matrix.empty or len(close_matrix) < self.vol_history_lookback:
                logger.warning(
                    "Insufficient history for regime detection "
                    f"({len(close_matrix)} bars, need {self.vol_history_lookback}); "
                    "returning default regime."
                )
                return _default_regime_state()

            returns_matrix = close_matrix.pct_change().dropna(how="all")

            # --- Individual signal dimensions ---
            vol_label, vol_z = self._volatility_regime(close_matrix, returns_matrix)
            trend_strength, adx_value, breadth = self._trend_regime(close_matrix)
            corr_level = self._correlation_regime(returns_matrix)
            is_crisis, crisis_confidence = self._crisis_detection(
                close_matrix, returns_matrix, vol_z, corr_level
            )

            # --- Classify ---
            regime, confidence = self._classify(
                vol_label=vol_label,
                vol_z=vol_z,
                trend_strength=trend_strength,
                adx_value=adx_value,
                breadth=breadth,
                corr_level=corr_level,
                is_crisis=is_crisis,
                crisis_confidence=crisis_confidence,
            )

            strategy_weights = dict(_REGIME_WEIGHT_ADJUSTMENTS[regime])
            position_scale = _REGIME_POSITION_SCALE[regime]

            state = RegimeState(
                regime=regime,
                confidence=confidence,
                volatility_regime=vol_label,
                trend_strength=trend_strength,
                correlation_level=corr_level,
                strategy_weights=strategy_weights,
                position_scale=position_scale,
            )

            logger.info(
                "Regime detected: {} (confidence={:.2f}, vol={}, trend={:.2f}, "
                "corr={:.2f}, position_scale={:.2f})",
                regime.name,
                confidence,
                vol_label,
                trend_strength,
                corr_level,
                position_scale,
            )

            return state

        except Exception:
            logger.exception("Regime detection failed; returning default regime.")
            return _default_regime_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_close_matrix(enriched_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract close prices into a single DataFrame (columns = tickers)."""
        frames: dict[str, pd.Series] = {}
        for ticker, df in enriched_data.items():
            if df is None or df.empty:
                continue
            # Accept 'close' or 'Close'
            col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
            if col is None:
                continue
            frames[ticker] = df[col]
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).sort_index().dropna(how="all")

    # --- Volatility --------------------------------------------------

    def _volatility_regime(
        self,
        close_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
    ) -> tuple[str, float]:
        """Classify volatility regime and return (label, z-score).

        Uses SPY (or first available ticker) realized vol as a VIX proxy.
        """
        # Pick a reference series for the vol proxy.
        if self.spy_ticker in returns_matrix.columns:
            ref = returns_matrix[self.spy_ticker].dropna()
        else:
            ref = returns_matrix.iloc[:, 0].dropna()

        current_vol = ref.tail(self.vol_lookback).std() * np.sqrt(252)
        hist_vol_series = ref.rolling(self.vol_lookback).std() * np.sqrt(252)
        hist_vol_series = hist_vol_series.dropna()

        if len(hist_vol_series) < 2:
            return "normal", 0.0

        hist_mean = hist_vol_series.mean()
        hist_std = hist_vol_series.std()

        if hist_std == 0 or np.isnan(hist_std):
            return "normal", 0.0

        vol_z = float((current_vol - hist_mean) / hist_std)

        if vol_z > 2.0:
            label = "extreme"
        elif vol_z > 1.0:
            label = "high"
        elif vol_z < -1.0:
            label = "low"
        else:
            label = "normal"

        return label, vol_z

    # --- Trend -------------------------------------------------------

    def _trend_regime(
        self,
        close_matrix: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """Return (trend_strength, adx_value, breadth).

        * trend_strength: -1..1 composite from MA slopes.
        * adx_value: pseudo-ADX (0..100) averaged across universe.
        * breadth: fraction of universe trading above their 200-SMA.
        """
        slopes: list[float] = []
        adx_values: list[float] = []
        above_long_ma = 0
        total = 0

        for ticker in close_matrix.columns:
            series = close_matrix[ticker].dropna()
            if len(series) < self.trend_long_window:
                continue

            total += 1

            # --- MA slope (normalised) ---
            short_ma = series.rolling(self.trend_short_window).mean()
            long_ma = series.rolling(self.trend_long_window).mean()

            # Slope of the short MA over the last 10 bars, normalised by price.
            recent_short_ma = short_ma.dropna().tail(10)
            if len(recent_short_ma) >= 2:
                x = np.arange(len(recent_short_ma), dtype=float)
                slope, _, _, _, _ = stats.linregress(x, recent_short_ma.values)
                norm_slope = slope / series.iloc[-1]  # normalise
                slopes.append(float(np.clip(norm_slope * 500, -1, 1)))

            # --- Breadth ---
            if not np.isnan(long_ma.iloc[-1]) and series.iloc[-1] > long_ma.iloc[-1]:
                above_long_ma += 1

            # --- Pseudo-ADX ---
            adx_val = self._compute_adx(series)
            if adx_val is not None:
                adx_values.append(adx_val)

        trend_strength = float(np.mean(slopes)) if slopes else 0.0
        adx_value = float(np.mean(adx_values)) if adx_values else 0.0
        breadth = above_long_ma / total if total > 0 else 0.5

        return trend_strength, adx_value, breadth

    def _compute_adx(self, series: pd.Series) -> float | None:
        """Compute a simplified ADX value from a price series.

        This is a lightweight approximation: we compute the directional
        movement from close-to-close changes (no high/low available) and
        smooth with an EMA.
        """
        if len(series) < self.adx_window * 2:
            return None

        diff = series.diff()
        pos_dm = diff.clip(lower=0)
        neg_dm = (-diff).clip(lower=0)

        smoothed_pos = pos_dm.ewm(span=self.adx_window, adjust=False).mean()
        smoothed_neg = neg_dm.ewm(span=self.adx_window, adjust=False).mean()

        atr = series.diff().abs().ewm(span=self.adx_window, adjust=False).mean()
        atr = atr.replace(0, np.nan)

        di_pos = (smoothed_pos / atr) * 100
        di_neg = (smoothed_neg / atr) * 100

        dx_denom = (di_pos + di_neg).replace(0, np.nan)
        dx = ((di_pos - di_neg).abs() / dx_denom) * 100

        adx = dx.ewm(span=self.adx_window, adjust=False).mean()
        last = adx.iloc[-1]
        if np.isnan(last):
            return None
        return float(last)

    # --- Correlation -------------------------------------------------

    def _correlation_regime(self, returns_matrix: pd.DataFrame) -> float:
        """Average pairwise correlation of returns over the correlation lookback."""
        recent = returns_matrix.tail(self.correlation_lookback).dropna(axis=1, how="all")
        if recent.shape[1] < 2:
            return 0.5

        corr = recent.corr()
        # Extract upper triangle (excluding diagonal).
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairwise = corr.values[mask]
        pairwise = pairwise[~np.isnan(pairwise)]

        if len(pairwise) == 0:
            return 0.5

        avg_corr = float(np.mean(pairwise))
        # Clamp to [0, 1] (correlations can be negative, but we want a level).
        return float(np.clip(avg_corr, 0.0, 1.0))

    # --- Crisis detection -------------------------------------------

    def _crisis_detection(
        self,
        close_matrix: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        vol_z: float,
        corr_level: float,
    ) -> tuple[bool, float]:
        """Check for crisis conditions.

        Returns ``(is_crisis, confidence)`` where confidence is 0-1.

        A crisis requires simultaneous:
        - Volatility spike (z-score > crisis_vol_multiplier)
        - Correlation spike (above crisis_corr_threshold)
        - Significant drawdown on the reference index
        """
        signals: list[float] = []

        # 1. Volatility spike
        vol_signal = float(np.clip((vol_z - 1.0) / (self.crisis_vol_multiplier - 1.0), 0, 1))
        signals.append(vol_signal)

        # 2. Correlation spike
        corr_signal = float(
            np.clip(
                (corr_level - 0.5) / (self.crisis_corr_threshold - 0.5),
                0,
                1,
            )
        )
        signals.append(corr_signal)

        # 3. Drawdown on the reference series
        if self.spy_ticker in close_matrix.columns:
            ref = close_matrix[self.spy_ticker].dropna()
        else:
            ref = close_matrix.iloc[:, 0].dropna()

        rolling_max = ref.rolling(self.vol_history_lookback, min_periods=1).max()
        drawdown = (ref.iloc[-1] / rolling_max.iloc[-1]) - 1.0 if rolling_max.iloc[-1] != 0 else 0.0

        dd_signal = float(
            np.clip(
                drawdown / self.crisis_drawdown_threshold,  # positive when drawdown is negative
                0,
                1,
            )
        )
        signals.append(dd_signal)

        # Require all three axes to contribute.
        crisis_confidence = float(np.mean(signals))
        is_crisis = all(s > 0.3 for s in signals) and crisis_confidence > 0.5

        return is_crisis, crisis_confidence

    # --- Final classification ----------------------------------------

    @staticmethod
    def _classify(
        vol_label: str,
        vol_z: float,
        trend_strength: float,
        adx_value: float,
        breadth: float,
        corr_level: float,
        is_crisis: bool,
        crisis_confidence: float,
    ) -> tuple[RegimeType, float]:
        """Combine all signal dimensions into a single regime classification."""

        # Crisis overrides everything.
        if is_crisis:
            return RegimeType.CRISIS, crisis_confidence

        # Score each candidate regime.
        scores: dict[RegimeType, float] = {}

        # --- TRENDING_UP ---
        up_score = 0.0
        if trend_strength > 0.2:
            up_score += min(trend_strength, 1.0) * 0.4
        if adx_value > 25:
            up_score += min(adx_value / 50, 1.0) * 0.3
        if breadth > 0.6:
            up_score += breadth * 0.3
        scores[RegimeType.TRENDING_UP] = up_score

        # --- TRENDING_DOWN ---
        down_score = 0.0
        if trend_strength < -0.2:
            down_score += min(abs(trend_strength), 1.0) * 0.4
        if adx_value > 25:
            down_score += min(adx_value / 50, 1.0) * 0.3
        if breadth < 0.4:
            down_score += (1 - breadth) * 0.3
        scores[RegimeType.TRENDING_DOWN] = down_score

        # --- MEAN_REVERTING ---
        mr_score = 0.0
        if adx_value < 20:
            mr_score += (1 - adx_value / 20) * 0.4
        if abs(trend_strength) < 0.15:
            mr_score += (1 - abs(trend_strength) / 0.15) * 0.3
        if 0.35 < breadth < 0.65:
            # Breadth near 50 % suggests no clear trend.
            mr_score += (1 - abs(breadth - 0.5) / 0.15) * 0.3
        scores[RegimeType.MEAN_REVERTING] = max(mr_score, 0.0)

        # --- HIGH_VOLATILITY ---
        hv_score = 0.0
        if vol_label in ("high", "extreme"):
            hv_score += min(vol_z / 2.5, 1.0) * 0.6
        if corr_level > 0.5:
            hv_score += corr_level * 0.4
        scores[RegimeType.HIGH_VOLATILITY] = hv_score

        # --- LOW_VOLATILITY ---
        lv_score = 0.0
        if vol_label == "low":
            lv_score += min(abs(vol_z) / 2.0, 1.0) * 0.5
        if corr_level < 0.3:
            lv_score += (1 - corr_level / 0.3) * 0.3
        if adx_value < 20:
            lv_score += 0.2
        scores[RegimeType.LOW_VOLATILITY] = lv_score

        # Pick the highest-scoring regime.
        best_regime = max(scores, key=lambda r: scores[r])
        best_score = scores[best_regime]

        # Confidence: raw score normalised so a perfect score maps to ~1.0.
        confidence = float(np.clip(best_score, 0.0, 1.0))

        # If nothing scored meaningfully, default to MEAN_REVERTING.
        if confidence < 0.10:
            return RegimeType.MEAN_REVERTING, 0.1

        return best_regime, confidence
