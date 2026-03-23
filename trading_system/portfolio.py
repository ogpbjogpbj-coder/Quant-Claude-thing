"""Portfolio manager — tracks positions, P&L, and manages trade lifecycle."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import TradingConfig
from trading_system.execution import ExecutionEngine
from trading_system.risk_manager import RiskManager
from trading_system.strategies.base import Signal
from trading_system.utils.db import record_snapshot
from trading_system.utils.logger import log_trade


# Sector mappings for the default universe
SECTOR_MAP = {
    "SPY": "ETF", "QQQ": "ETF",
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Technology",
    "TSLA": "Consumer Discretionary", "AMD": "Technology",
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "BAC": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "PG": "Consumer Staples", "COST": "Consumer Staples", "PEP": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "XOM": "Energy", "CVX": "Energy",
    "AVGO": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "NFLX": "Communication Services",
}


@dataclass
class TrackedPosition:
    """In-memory tracking for an open position with metadata."""
    symbol: str
    qty: float
    avg_entry: float
    side: str  # "long" or "short"
    strategy: str = ""
    entry_time: datetime = field(default_factory=datetime.utcnow)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float("inf")

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_entry

    def update_extremes(self, price: float) -> None:
        if price > self.highest_price:
            self.highest_price = price
        if price < self.lowest_price:
            self.lowest_price = price


class PortfolioManager:
    """Orchestrates signal -> order flow with position tracking."""

    def __init__(
        self,
        config: TradingConfig,
        execution: ExecutionEngine,
        risk_manager: RiskManager,
    ):
        self.config = config
        self.execution = execution
        self.risk_manager = risk_manager
        self.tracked: dict[str, TrackedPosition] = {}
        self._equity = 0.0
        self._cash = 0.0

    def sync_positions(self) -> None:
        """Sync tracked positions with broker state."""
        broker_positions = self.execution.get_positions()
        account = self.execution.get_account()

        self._equity = account.get("equity", 0)
        self._cash = account.get("cash", 0)

        # Update existing tracked positions with broker data
        broker_symbols = set(broker_positions.keys())
        tracked_symbols = set(self.tracked.keys())

        # Remove tracked positions that are no longer at broker
        for sym in tracked_symbols - broker_symbols:
            logger.info(f"Position {sym} no longer at broker, removing from tracker")
            del self.tracked[sym]

        # Update or add broker positions
        for sym, pos_data in broker_positions.items():
            if sym in self.tracked:
                self.tracked[sym].qty = pos_data["qty"]
                self.tracked[sym].update_extremes(pos_data["current_price"])
            else:
                self.tracked[sym] = TrackedPosition(
                    symbol=sym,
                    qty=pos_data["qty"],
                    avg_entry=pos_data["avg_entry_price"],
                    side="long" if pos_data["qty"] > 0 else "short",
                )
                self.tracked[sym].highest_price = pos_data["current_price"]
                self.tracked[sym].lowest_price = pos_data["current_price"]

        self.risk_manager.update_peak_equity(self._equity)

    def get_current_positions_qty(self) -> dict[str, float]:
        """Get symbol -> qty mapping for strategy input."""
        return {sym: t.qty for sym, t in self.tracked.items()}

    def get_position_values(self) -> dict[str, float]:
        """Get symbol -> market value from broker."""
        positions = self.execution.get_positions()
        return {sym: abs(p["market_value"]) for sym, p in positions.items()}

    def _get_sector_exposure(self) -> dict[str, float]:
        """Calculate current dollar exposure by sector."""
        position_values = self.get_position_values()
        sector_exposure: dict[str, float] = {}
        for sym, value in position_values.items():
            sector = SECTOR_MAP.get(sym, "Other")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        return sector_exposure

    def _check_sector_limit(self, symbol: str, add_value: float) -> bool:
        """Check if adding this position would breach sector limit."""
        sector = SECTOR_MAP.get(symbol, "Other")
        if sector == "ETF":
            return True  # No sector limit on ETFs
        sector_exposure = self._get_sector_exposure()
        current = sector_exposure.get(sector, 0)
        max_sector = self._equity * self.config.risk.max_sector_exposure_pct / 100
        if current + add_value > max_sector:
            logger.info(
                f"Sector limit: {symbol} ({sector}) would breach "
                f"${current + add_value:,.0f} > ${max_sector:,.0f}"
            )
            return False
        return True

    def _check_correlation(
        self, symbol: str, enriched_data: dict[str, pd.DataFrame] | None
    ) -> float:
        """Return a correlation-based scaling factor (0-1) for a new position.

        If the new symbol is highly correlated with existing positions,
        scale down the position size.
        """
        if not enriched_data or symbol not in enriched_data:
            return 1.0
        if not self.tracked:
            return 1.0

        try:
            new_returns = enriched_data[symbol]["close"].pct_change().dropna().tail(60)
            max_corr = 0.0
            for existing_sym in self.tracked:
                if existing_sym in enriched_data:
                    ex_returns = enriched_data[existing_sym]["close"].pct_change().dropna().tail(60)
                    aligned = pd.concat([new_returns, ex_returns], axis=1).dropna()
                    if len(aligned) > 10:
                        corr = abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                        max_corr = max(max_corr, corr)

            threshold = self.config.risk.max_correlation_threshold
            if max_corr > threshold:
                scale = max(0.2, 1.0 - (max_corr - threshold) / (1.0 - threshold))
                logger.info(
                    f"Correlation scale for {symbol}: {scale:.2f} "
                    f"(max_corr={max_corr:.2f})"
                )
                return scale
        except Exception as e:
            logger.warning(f"Correlation check failed for {symbol}: {e}")

        return 1.0

    def process_signals(
        self,
        signals: list[Signal],
        prices: dict[str, float],
        enriched_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[str]:
        """Process approved signals into orders.

        Returns list of order IDs.
        """
        self.sync_positions()

        # Risk filter
        position_values = self.get_position_values()
        approved = self.risk_manager.filter_signals(
            signals=signals,
            current_positions=self.get_current_positions_qty(),
            portfolio_equity=self._equity,
            position_values=position_values,
        )

        if not approved:
            return []

        order_ids = []

        for signal in approved:
            price = prices.get(signal.symbol, 0)
            if price <= 0:
                continue

            current_value = position_values.get(signal.symbol, 0)
            current_qty = self.get_current_positions_qty().get(signal.symbol, 0)

            if signal.is_buy:
                # Sector limit check
                est_size = self._equity * self.config.risk.max_position_size_pct / 100 * 0.5
                if not self._check_sector_limit(signal.symbol, est_size):
                    continue

                # Calculate position size in dollars
                volatility = signal.metadata.get("volatility_21d", 0.2)
                size_dollars = self.risk_manager.calculate_position_size(
                    signal=signal,
                    price=price,
                    portfolio_equity=self._equity,
                    current_position_value=current_value,
                    volatility=volatility,
                )

                if size_dollars <= 0:
                    continue

                # Apply correlation scaling
                corr_scale = self._check_correlation(signal.symbol, enriched_data)
                size_dollars *= corr_scale

                # Apply regime position scale if present
                regime_scale = signal.metadata.get("regime_scale", 1.0)
                size_dollars *= regime_scale

                if size_dollars <= 0:
                    continue

                qty = size_dollars / price
                if not self.config.execution.enable_fractional:
                    qty = int(qty)
                if qty <= 0:
                    continue

                order_id = self.execution.place_order(
                    symbol=signal.symbol,
                    qty=round(qty, 4),
                    side="buy",
                    price=price,
                    strategy=signal.strategy,
                    signal_strength=signal.strength,
                    stop_loss=signal.stop_loss or 0,
                    take_profit=signal.take_profit or 0,
                )

                if order_id:
                    # Update tracking
                    if signal.symbol in self.tracked:
                        t = self.tracked[signal.symbol]
                        total_qty = t.qty + qty
                        t.avg_entry = (t.avg_entry * t.qty + price * qty) / total_qty
                        t.qty = total_qty
                    else:
                        self.tracked[signal.symbol] = TrackedPosition(
                            symbol=signal.symbol,
                            qty=qty,
                            avg_entry=price,
                            side="long",
                            strategy=signal.strategy,
                            stop_loss=signal.stop_loss or (price * 0.95),
                            take_profit=signal.take_profit or (price * 1.10),
                            highest_price=price,
                        )
                    order_ids.append(order_id)

            elif signal.is_sell and current_qty > 0:
                # Sell: reduce or close position
                sell_fraction = min(abs(signal.direction), 1.0)
                sell_qty = round(current_qty * sell_fraction, 4)

                if sell_qty < 0.001:
                    continue

                if sell_fraction >= 0.9:
                    # Close entire position
                    order_id = self.execution.close_position(
                        signal.symbol, reason=signal.strategy
                    )
                else:
                    order_id = self.execution.place_order(
                        symbol=signal.symbol,
                        qty=sell_qty,
                        side="sell",
                        price=price,
                        strategy=signal.strategy,
                        signal_strength=signal.strength,
                    )

                if order_id:
                    order_ids.append(order_id)

        return order_ids

    def check_stops(self, prices: dict[str, float]) -> list[str]:
        """Check all positions against stop losses and take profits.

        Returns list of symbols where stops were triggered.
        """
        triggered = []

        for sym, tracked in list(self.tracked.items()):
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            tracked.update_extremes(price)

            # Calculate trailing stop
            if tracked.stop_loss > 0 and hasattr(self, '_data_cache'):
                # Use ATR if available, else use fixed stop
                pass

            # Check stop loss
            if tracked.stop_loss > 0 and price <= tracked.stop_loss:
                loss_pct = (tracked.avg_entry - price) / tracked.avg_entry * 100
                logger.warning(
                    f"STOP LOSS triggered for {sym}: "
                    f"price=${price:.2f} <= stop=${tracked.stop_loss:.2f} "
                    f"(loss={loss_pct:.1f}%)"
                )
                self.execution.close_position(sym, reason="stop_loss")
                triggered.append(sym)

            # Check take profit
            elif tracked.take_profit > 0 and price >= tracked.take_profit:
                gain_pct = (price - tracked.avg_entry) / tracked.avg_entry * 100
                logger.info(
                    f"TAKE PROFIT triggered for {sym}: "
                    f"price=${price:.2f} >= target=${tracked.take_profit:.2f} "
                    f"(gain={gain_pct:.1f}%)"
                )
                self.execution.close_position(sym, reason="take_profit")
                triggered.append(sym)

            # Update trailing stop (ratchet up only)
            elif tracked.qty > 0 and tracked.highest_price > tracked.avg_entry:
                # Simple trailing: 2 ATR or 5% from high, whichever is tighter
                pct_trail = tracked.highest_price * 0.95
                new_stop = max(tracked.stop_loss, pct_trail)
                if new_stop > tracked.stop_loss:
                    tracked.stop_loss = new_stop

        return triggered

    def take_snapshot(self) -> None:
        """Record current portfolio state to database."""
        account = self.execution.get_account()
        positions = self.execution.get_positions()

        equity = account.get("equity", 0)
        cash = account.get("cash", 0)
        positions_value = sum(abs(p["market_value"]) for p in positions.values())

        daily_pnl = sum(p["unrealized_pl"] for p in positions.values())
        peak = self.risk_manager._peak_equity or equity
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0

        record_snapshot(
            equity=equity,
            cash=cash,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            num_positions=len(positions),
            drawdown_pct=drawdown,
        )

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def cash(self) -> float:
        return self._cash
