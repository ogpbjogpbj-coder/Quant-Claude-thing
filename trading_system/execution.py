"""Order execution engine for Alpaca API.

Handles order placement, monitoring, and management with retry logic
and slippage control.
"""

import time
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    GetOrdersRequest,
)
from loguru import logger

from trading_system.config import ExecutionConfig, AlpacaConfig
from trading_system.utils.db import record_trade
from trading_system.utils.logger import log_trade


class ExecutionEngine:
    """Manages order execution through Alpaca API."""

    def __init__(self, alpaca_config: AlpacaConfig, exec_config: ExecutionConfig):
        self.config = exec_config
        self.client = TradingClient(
            api_key=alpaca_config.api_key,
            secret_key=alpaca_config.secret_key,
            paper=alpaca_config.is_paper,
        )

    def get_account(self) -> dict:
        """Get current account status."""
        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}

    def get_positions(self) -> dict[str, dict]:
        """Get all open positions."""
        try:
            positions = self.client.get_all_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    "qty": float(pos.qty),
                    "side": pos.side.value,
                    "market_value": float(pos.market_value),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "cost_basis": float(pos.cost_basis),
                }
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        price: Optional[float] = None,
        strategy: str = "",
        signal_strength: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Optional[str]:
        """Place an order with retry logic.

        Args:
            symbol: Ticker symbol
            qty: Number of shares (fractional OK)
            side: "buy" or "sell"
            price: Limit price (None for market order)
            strategy: Strategy that generated the signal
            signal_strength: Signal strength for logging

        Returns:
            Order ID if successful, None if failed
        """
        if qty <= 0:
            return None

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.order_type == "limit" and price is not None:
                    # Adjust limit price for better fill
                    if side == "buy":
                        limit_price = round(price * (1 + self.config.limit_offset_pct / 100), 2)
                    else:
                        limit_price = round(price * (1 - self.config.limit_offset_pct / 100), 2)

                    order_request = LimitOrderRequest(
                        symbol=symbol,
                        qty=round(qty, 4) if self.config.enable_fractional else int(qty),
                        side=order_side,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce(self.config.time_in_force.lower()),
                        limit_price=limit_price,
                    )
                else:
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=round(qty, 4) if self.config.enable_fractional else int(qty),
                        side=order_side,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce(self.config.time_in_force.lower()),
                    )

                order = self.client.submit_order(order_request)

                fill_price = float(order.filled_avg_price) if order.filled_avg_price else (price or 0)

                log_trade(
                    action=side.upper(),
                    symbol=symbol,
                    qty=qty,
                    price=fill_price,
                    reason=strategy,
                    order_id=order.id,
                    signal=signal_strength,
                    status=order.status.value,
                )

                record_trade(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price=fill_price,
                    order_id=str(order.id),
                    strategy=strategy,
                    signal_strength=signal_strength,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                logger.info(
                    f"Order placed: {side} {qty:.4f} {symbol} @ "
                    f"{'$' + str(fill_price) if fill_price else 'market'} "
                    f"[{strategy}] -> {order.status.value}"
                )

                return str(order.id)

            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)

        logger.error(f"All order attempts failed for {side} {qty} {symbol}")
        return None

    def close_position(self, symbol: str, reason: str = "") -> Optional[str]:
        """Close an entire position."""
        try:
            self.client.close_position(symbol)
            logger.info(f"Position closed: {symbol} | reason={reason}")

            record_trade(
                symbol=symbol,
                side="sell",
                qty=0,  # Full close
                price=0,
                strategy="close",
                notes=reason,
            )

            return symbol
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None

    def close_all_positions(self, reason: str = "") -> list[str]:
        """Emergency: close all positions."""
        logger.warning(f"CLOSING ALL POSITIONS: {reason}")
        try:
            self.client.close_all_positions(cancel_orders=True)
            return ["all"]
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return []

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            self.client.cancel_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    def get_open_orders(self) -> list[dict]:
        """Get all open/pending orders."""
        try:
            request = GetOrdersRequest(status="open")
            orders = self.client.get_orders(request)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.qty),
                    "type": o.type.value,
                    "status": o.status.value,
                    "limit_price": float(o.limit_price) if o.limit_price else None,
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market clock: {e}")
            return False

    def get_clock(self) -> dict:
        """Get market clock details."""
        try:
            clock = self.client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            }
        except Exception as e:
            logger.error(f"Failed to get clock: {e}")
            return {"is_open": False}
