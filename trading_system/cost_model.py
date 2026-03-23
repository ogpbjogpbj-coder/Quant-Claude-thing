"""Execution cost model for estimating and tracking trading costs."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from loguru import logger

from trading_system.config import ExecutionConfig


@dataclass
class CostEstimate:
    """Estimated execution costs for a trade."""

    spread_cost: float
    impact_cost: float
    slippage_cost: float
    total_cost_pct: float
    total_cost_dollars: float


@dataclass
class _FillRecord:
    expected_price: float
    actual_price: float
    qty: float
    side: str
    slippage_pct: float


class ExecutionCostModel:
    """Model for estimating, evaluating, and tracking execution costs.

    Uses spread estimation, square-root market impact, and volatility-based
    slippage to decide whether a trade's expected alpha justifies its costs.
    """

    # Price thresholds for cap-size spread estimation
    _MEGA_CAP_PRICE = 500.0
    _LARGE_CAP_PRICE = 50.0

    # Spread basis-point assumptions by cap bucket
    _SPREAD_BPS_MEGA = 0.0001   # 0.01%
    _SPREAD_BPS_LARGE = 0.0003  # 0.03%
    _SPREAD_BPS_MID = 0.0005    # 0.05%

    # Minimum cost-to-alpha ratio required to justify a trade
    _ALPHA_COST_MULTIPLE = 2.0

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        # symbol -> list of fill records
        self._fill_history: Dict[str, List[_FillRecord]] = {}
        logger.info(
            "ExecutionCostModel initialised (order_type={}, max_slippage={}%)",
            config.order_type,
            config.max_slippage_pct,
        )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        symbol: str,
        qty: float,
        price: float,
        side: str,
        volatility: float,
        avg_volume: float,
    ) -> CostEstimate:
        """Estimate the all-in execution cost for a proposed trade.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        qty : float
            Number of shares (can be fractional).
        price : float
            Current or reference price per share.
        side : str
            ``"buy"`` or ``"sell"``.
        volatility : float
            Annualised return volatility (e.g. 0.25 for 25%).
        avg_volume : float
            Average daily volume in shares.

        Returns
        -------
        CostEstimate
        """
        if price <= 0:
            logger.warning("Zero/negative price for {} – returning zero cost estimate", symbol)
            return CostEstimate(
                spread_cost=0.0,
                impact_cost=0.0,
                slippage_cost=0.0,
                total_cost_pct=0.0,
                total_cost_dollars=0.0,
            )

        notional = abs(qty) * price

        # --- Spread cost ---
        spread_bps = self._spread_bps_for_price(price)
        # Pay half the spread on each side of a trade
        spread_cost = notional * spread_bps / 2.0

        # --- Market impact (square-root model) ---
        impact_cost = self._market_impact(abs(qty), price, volatility, avg_volume)

        # --- Slippage estimate ---
        slippage_cost = self._slippage_estimate(notional, volatility)

        total_dollars = spread_cost + impact_cost + slippage_cost
        total_pct = (total_dollars / notional * 100.0) if notional > 0 else 0.0

        estimate = CostEstimate(
            spread_cost=round(spread_cost, 4),
            impact_cost=round(impact_cost, 4),
            slippage_cost=round(slippage_cost, 4),
            total_cost_pct=round(total_pct, 6),
            total_cost_dollars=round(total_dollars, 4),
        )

        logger.debug(
            "Cost estimate for {} {} {} @ ${:.2f}: spread=${:.4f} impact=${:.4f} "
            "slip=${:.4f} total={:.4f}% (${:.4f})",
            side,
            qty,
            symbol,
            price,
            estimate.spread_cost,
            estimate.impact_cost,
            estimate.slippage_cost,
            estimate.total_cost_pct,
            estimate.total_cost_dollars,
        )
        return estimate

    # ------------------------------------------------------------------
    # Trade gating
    # ------------------------------------------------------------------

    def should_trade(self, signal_strength: float, cost_estimate: CostEstimate) -> bool:
        """Decide whether the expected alpha justifies the estimated cost.

        The trade is approved only when the signal strength (used as a rough
        proxy for expected return in percent) exceeds twice the total cost
        percentage, providing a buffer for estimation error.

        Parameters
        ----------
        signal_strength : float
            Magnitude of the trading signal, interpreted as expected return
            in the same units as ``cost_estimate.total_cost_pct``.
        cost_estimate : CostEstimate
            Output of :meth:`estimate_cost`.

        Returns
        -------
        bool
        """
        edge = abs(signal_strength)
        threshold = cost_estimate.total_cost_pct * self._ALPHA_COST_MULTIPLE

        if edge > threshold:
            logger.debug(
                "Trade approved: signal {:.4f}% > {:.4f}% threshold",
                edge,
                threshold,
            )
            return True

        logger.debug(
            "Trade rejected: signal {:.4f}% <= {:.4f}% threshold",
            edge,
            threshold,
        )
        return False

    # ------------------------------------------------------------------
    # Fill tracking
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
        qty: float,
        side: str,
    ) -> None:
        """Record a fill to track realised slippage.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        expected_price : float
            Price at which the order was expected to fill.
        actual_price : float
            Actual fill price.
        qty : float
            Filled quantity.
        side : str
            ``"buy"`` or ``"sell"``.
        """
        if expected_price <= 0:
            logger.warning("Cannot record fill for {} with expected_price={}", symbol, expected_price)
            return

        # Slippage: positive means worse fill than expected
        if side.lower() == "buy":
            slippage_pct = (actual_price - expected_price) / expected_price * 100.0
        else:
            slippage_pct = (expected_price - actual_price) / expected_price * 100.0

        record = _FillRecord(
            expected_price=expected_price,
            actual_price=actual_price,
            qty=qty,
            side=side.lower(),
            slippage_pct=slippage_pct,
        )

        self._fill_history.setdefault(symbol, []).append(record)

        logger.info(
            "Fill recorded for {} {}: expected=${:.4f} actual=${:.4f} slippage={:.4f}%",
            side,
            symbol,
            expected_price,
            actual_price,
            slippage_pct,
        )

    def get_slippage_stats(self) -> Dict[str, dict]:
        """Return per-symbol slippage statistics.

        Returns
        -------
        dict
            Mapping of symbol to a dict with keys ``avg_slippage_pct``,
            ``median_slippage_pct``, ``worst_slippage_pct``, and ``n_fills``.
            An empty dict is returned when no fills have been recorded.
        """
        stats: Dict[str, dict] = {}

        for symbol, records in self._fill_history.items():
            slippages = np.array([r.slippage_pct for r in records])
            stats[symbol] = {
                "avg_slippage_pct": float(np.mean(slippages)),
                "median_slippage_pct": float(np.median(slippages)),
                "worst_slippage_pct": float(np.max(slippages)),
                "n_fills": len(records),
            }

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _spread_bps_for_price(cls, price: float) -> float:
        """Return estimated half-spread in decimal (not percent) for a price level."""
        if price >= cls._MEGA_CAP_PRICE:
            return cls._SPREAD_BPS_MEGA
        if price >= cls._LARGE_CAP_PRICE:
            return cls._SPREAD_BPS_LARGE
        return cls._SPREAD_BPS_MID

    @staticmethod
    def _market_impact(qty: float, price: float, volatility: float, avg_volume: float) -> float:
        """Square-root market impact model.

        impact_dollars = price * sigma * sqrt(qty / adv) * qty
        """
        if avg_volume <= 0 or qty <= 0:
            return 0.0

        participation = qty / avg_volume
        # Cap participation at 1.0 to keep the model sensible
        participation = min(participation, 1.0)

        impact_per_share = price * volatility * np.sqrt(participation)
        return float(impact_per_share * qty)

    def _slippage_estimate(self, notional: float, volatility: float) -> float:
        """Estimate expected slippage in dollars.

        For limit orders the slippage risk is lower because the limit offset
        provides a buffer; for market orders we use a volatility-scaled estimate.
        """
        if notional <= 0:
            return 0.0

        if self.config.order_type == "limit":
            # Limit orders have the offset as a cushion; residual slippage is small
            slippage_pct = volatility * 0.001  # very small fraction of vol
        else:
            # Market orders bear full short-term volatility cost
            slippage_pct = volatility * 0.005

        return notional * slippage_pct
