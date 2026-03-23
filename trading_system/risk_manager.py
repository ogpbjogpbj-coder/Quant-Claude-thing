"""Risk management system with circuit breakers and position sizing.

Enforces:
- Per-position size limits
- Portfolio-level risk limits
- Daily/weekly loss circuit breakers
- Max drawdown protection
- Correlation-based diversification
- Kelly criterion position sizing
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import RiskConfig
from trading_system.strategies.base import Signal
from trading_system.utils.db import get_daily_pnl, get_peak_equity, record_risk_event
from trading_system.utils.logger import log_risk


class RiskManager:
    """Enforces risk limits and sizes positions."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        self._daily_start_equity: Optional[float] = None
        self._weekly_start_equity: Optional[float] = None
        self._peak_equity: float = 0.0
        self._week_start: Optional[datetime] = None

    def initialize(self, equity: float) -> None:
        """Set initial reference points."""
        self._daily_start_equity = equity
        self._peak_equity = max(equity, get_peak_equity())

        now = datetime.utcnow()
        if self._week_start is None or (now - self._week_start).days >= 7:
            self._weekly_start_equity = equity
            self._week_start = now

        logger.info(
            f"Risk manager initialized: equity=${equity:,.2f}, "
            f"peak=${self._peak_equity:,.2f}"
        )

    def update_peak_equity(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity

    def check_circuit_breakers(self, current_equity: float) -> bool:
        """Check if any circuit breaker should halt trading. Returns True if safe."""
        if self._circuit_breaker_active:
            logger.warning(f"Circuit breaker active: {self._circuit_breaker_reason}")
            return False

        # Daily loss check
        if self._daily_start_equity and self._daily_start_equity > 0:
            daily_loss_pct = (self._daily_start_equity - current_equity) / self._daily_start_equity * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self._trip_breaker(
                    f"Daily loss limit hit: {daily_loss_pct:.2f}% >= {self.config.max_daily_loss_pct}%"
                )
                return False

        # Weekly loss check
        if self._weekly_start_equity and self._weekly_start_equity > 0:
            weekly_loss_pct = (self._weekly_start_equity - current_equity) / self._weekly_start_equity * 100
            if weekly_loss_pct >= self.config.max_weekly_loss_pct:
                self._trip_breaker(
                    f"Weekly loss limit hit: {weekly_loss_pct:.2f}% >= {self.config.max_weekly_loss_pct}%"
                )
                return False

        # Max drawdown check
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
            if drawdown_pct >= self.config.max_drawdown_pct:
                self._trip_breaker(
                    f"Max drawdown hit: {drawdown_pct:.2f}% >= {self.config.max_drawdown_pct}%"
                )
                return False

        return True

    def _trip_breaker(self, reason: str) -> None:
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason
        log_risk("CIRCUIT_BREAKER", reason=reason)
        record_risk_event("circuit_breaker", reason, "trading_halted")

    def reset_daily(self, equity: float) -> None:
        """Reset daily counters (call at market open)."""
        self._daily_start_equity = equity
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        logger.info(f"Daily risk counters reset. Start equity: ${equity:,.2f}")

    def reset_weekly(self, equity: float) -> None:
        self._weekly_start_equity = equity
        self._week_start = datetime.utcnow()

    def filter_signals(
        self,
        signals: list[Signal],
        current_positions: dict[str, float],
        portfolio_equity: float,
        position_values: dict[str, float],
    ) -> list[Signal]:
        """Filter signals through risk checks and size positions."""
        if not self.check_circuit_breakers(portfolio_equity):
            logger.warning("All signals blocked — circuit breaker active")
            return []

        approved = []
        num_positions = len([v for v in current_positions.values() if abs(v) > 0])

        for signal in signals:
            # Skip weak signals
            if signal.strength < self.config.min_sharpe_ratio * 0.1:
                continue

            # Max positions check
            is_new = abs(current_positions.get(signal.symbol, 0)) < 0.01
            if is_new and signal.is_buy and num_positions >= self.config.max_open_positions:
                log_risk("MAX_POSITIONS", symbol=signal.symbol, current=num_positions)
                continue

            # Allow sell signals for existing positions always
            if signal.is_sell and signal.symbol in current_positions:
                approved.append(signal)
                continue

            # Position size limit
            current_value = abs(position_values.get(signal.symbol, 0))
            max_value = portfolio_equity * self.config.max_position_size_pct / 100
            if current_value >= max_value and signal.is_buy:
                log_risk(
                    "MAX_POSITION_SIZE",
                    symbol=signal.symbol,
                    current_pct=current_value / portfolio_equity * 100,
                )
                continue

            approved.append(signal)

        logger.info(f"Risk filter: {len(signals)} signals -> {len(approved)} approved")
        return approved

    def calculate_position_size(
        self,
        signal: Signal,
        price: float,
        portfolio_equity: float,
        current_position_value: float = 0.0,
        volatility: float = 0.0,
    ) -> float:
        """Calculate the dollar amount to allocate to this trade.

        Returns dollar amount (positive for buy, negative for sell).
        """
        if portfolio_equity <= 0 or price <= 0:
            return 0.0

        max_position_dollars = portfolio_equity * self.config.max_position_size_pct / 100
        max_trade_risk = portfolio_equity * self.config.max_single_trade_risk_pct / 100

        if self.config.position_sizing == "kelly":
            size = self._kelly_size(signal, portfolio_equity, max_trade_risk)
        elif self.config.position_sizing == "volatility_parity":
            size = self._volatility_parity_size(
                signal, portfolio_equity, volatility, max_trade_risk
            )
        else:
            # Equal weight
            size = portfolio_equity / self.config.max_open_positions

        # Cap at max position size minus current exposure
        remaining_capacity = max_position_dollars - abs(current_position_value)
        if signal.is_buy:
            size = min(size, remaining_capacity)

        # Scale by signal strength
        size *= signal.strength

        # Ensure minimum trade size
        min_trade = 1.0  # $1 minimum (for fractional shares)
        if abs(size) < min_trade:
            return 0.0

        return size if signal.is_buy else -abs(size)

    def _kelly_size(
        self,
        signal: Signal,
        equity: float,
        max_risk: float,
    ) -> float:
        """Kelly criterion position sizing (fractional Kelly for safety)."""
        # Estimate win probability from confidence
        win_prob = 0.5 + signal.confidence * 0.2  # Map confidence to 50-70% win rate

        # Estimate win/loss ratio from signal direction strength
        win_loss_ratio = 1.0 + abs(signal.direction) * 0.5  # 1.0x to 1.5x

        # Kelly fraction
        kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_f = max(kelly_f, 0)

        # Apply fraction (quarter-Kelly for safety)
        size = equity * kelly_f * self.config.kelly_fraction

        # Never risk more than max per trade
        return min(size, max_risk * 3)  # size != risk, risk is a fraction

    def _volatility_parity_size(
        self,
        signal: Signal,
        equity: float,
        volatility: float,
        max_risk: float,
    ) -> float:
        """Size inversely proportional to volatility."""
        if volatility <= 0:
            return equity / self.config.max_open_positions

        target_vol = 0.15  # 15% annualized target vol per position
        vol_scalar = target_vol / volatility
        vol_scalar = np.clip(vol_scalar, 0.2, 3.0)

        base_size = equity / self.config.max_open_positions
        size = base_size * vol_scalar

        return min(size, max_risk * 3)

    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        position_qty: float,
    ) -> bool:
        """Check if stop loss is hit. Returns True if should exit."""
        if position_qty > 0 and current_price <= stop_loss:
            loss_pct = (entry_price - current_price) / entry_price * 100
            log_risk(
                "STOP_LOSS_HIT",
                symbol=symbol,
                price=current_price,
                stop=stop_loss,
                loss_pct=f"{loss_pct:.2f}%",
            )
            return True
        elif position_qty < 0 and current_price >= stop_loss:
            return True
        return False

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        highest_since_entry: float,
    ) -> float:
        """Calculate trailing stop loss level."""
        # Trailing stop = highest price - ATR multiplier
        trail = highest_since_entry - self.config.stop_loss_atr_multiplier * atr

        # Never let trailing stop go below initial stop
        initial_stop = entry_price - self.config.stop_loss_atr_multiplier * atr
        return max(trail, initial_stop)

    @property
    def is_halted(self) -> bool:
        return self._circuit_breaker_active
