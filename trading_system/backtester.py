"""Backtesting framework for strategy validation.

Uses historical data to simulate trading and compute performance metrics
without risking real capital.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from trading_system.config import TradingConfig, load_config
from trading_system.data_ingestion import DataIngestion
from trading_system.signal_aggregator import SignalAggregator
from trading_system.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    MLEnsembleStrategy,
    VolatilityBreakoutStrategy,
    TrendFollowingStrategy,
)
from trading_system.strategies.base import Signal


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    entry_date: str
    exit_price: float = 0.0
    exit_date: str = ""
    pnl: float = 0.0
    strategy: str = ""


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"BACKTEST RESULTS: {self.start_date} to {self.end_date}\n"
            f"{'='*60}\n"
            f"Initial Capital:    ${self.initial_capital:>12,.2f}\n"
            f"Final Equity:       ${self.final_equity:>12,.2f}\n"
            f"Total Return:       {self.total_return_pct:>12.2f}%\n"
            f"Annual Return:      {self.annual_return_pct:>12.2f}%\n"
            f"Sharpe Ratio:       {self.sharpe_ratio:>12.3f}\n"
            f"Sortino Ratio:      {self.sortino_ratio:>12.3f}\n"
            f"Max Drawdown:       {self.max_drawdown_pct:>12.2f}%\n"
            f"Win Rate:           {self.win_rate:>12.1f}%\n"
            f"Profit Factor:      {self.profit_factor:>12.2f}\n"
            f"Total Trades:       {self.total_trades:>12d}\n"
            f"Avg Trade P&L:      ${self.avg_trade_pnl:>12.2f}\n"
            f"Avg Win:            ${self.avg_win:>12.2f}\n"
            f"Avg Loss:           ${self.avg_loss:>12.2f}\n"
            f"{'='*60}\n"
        )


class Backtester:
    """Simulates strategy execution on historical data."""

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or load_config()

    def run(
        self,
        initial_capital: float = 100_000,
        lookback_days: int = 252,
        max_positions: int = 10,
    ) -> BacktestResult:
        """Run a full backtest."""
        logger.info(f"Starting backtest with ${initial_capital:,.0f} over {lookback_days} days")

        # Fetch data
        data_engine = DataIngestion(self.config)
        raw_data = data_engine.get_bars(
            self.config.universe, lookback_days=lookback_days
        )

        # Compute indicators
        enriched = {}
        for sym, df in raw_data.items():
            enriched[sym] = data_engine.compute_indicators(df.copy())

        if not enriched:
            logger.error("No data for backtest")
            return BacktestResult(
                start_date="", end_date="", initial_capital=initial_capital,
                final_equity=initial_capital, total_return_pct=0, annual_return_pct=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown_pct=0, win_rate=0,
                profit_factor=0, total_trades=0, avg_trade_pnl=0, avg_win=0, avg_loss=0,
            )

        # Initialize strategies
        strategies = []
        sc = self.config.strategies
        if sc.momentum.enabled:
            strategies.append(MomentumStrategy(sc.momentum))
        if sc.mean_reversion.enabled:
            strategies.append(MeanReversionStrategy(sc.mean_reversion))
        if sc.volatility_breakout.enabled:
            strategies.append(VolatilityBreakoutStrategy(sc.volatility_breakout))
        if sc.trend_following.enabled:
            strategies.append(TrendFollowingStrategy(sc.trend_following))
        # Skip ML in backtest for speed (needs separate train/test split)

        aggregator = SignalAggregator(self.config.strategies)

        # Get common date range
        all_dates = set()
        for sym, df in enriched.items():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) < 30:
            logger.error("Insufficient data points for backtest")
            return self._empty_result(initial_capital)

        # Simulation state
        cash = initial_capital
        positions: dict[str, dict] = {}  # sym -> {qty, entry_price}
        equity_curve = []
        trades: list[BacktestTrade] = []

        # Walk forward through dates
        warmup = 60  # Skip first N days for indicator warmup
        for i, date in enumerate(dates[warmup:], start=warmup):
            # Build data slices up to this date
            current_data = {}
            current_prices = {}
            for sym, df in enriched.items():
                mask = df.index <= date
                if mask.sum() >= 30:
                    current_data[sym] = df[mask]
                    current_prices[sym] = df[mask].iloc[-1]["close"]

            if not current_data:
                continue

            # Get current position quantities
            current_pos_qty = {sym: p["qty"] for sym, p in positions.items()}

            # Generate signals
            all_signals = []
            for strategy in strategies:
                try:
                    signals = strategy.generate_signals(current_data, current_pos_qty)
                    all_signals.extend(signals)
                except Exception:
                    pass

            consensus = aggregator.aggregate(all_signals)

            # Execute signals (simplified)
            for signal in consensus[:max_positions]:
                sym = signal.symbol
                price = current_prices.get(sym, 0)
                if price <= 0:
                    continue

                if signal.is_buy and sym not in positions:
                    # Size: equal weight with position limit
                    equity = cash + sum(
                        p["qty"] * current_prices.get(s, p["entry_price"])
                        for s, p in positions.items()
                    )
                    if len(positions) >= max_positions:
                        continue

                    position_size = equity * 0.05 * signal.strength  # 5% max per position
                    position_size = min(position_size, cash * 0.95)
                    if position_size < 100:
                        continue

                    qty = position_size / price
                    cash -= qty * price
                    positions[sym] = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_date": str(date),
                        "stop_loss": signal.stop_loss or price * 0.95,
                        "take_profit": signal.take_profit or price * 1.10,
                        "strategy": signal.strategy,
                    }

                elif signal.is_sell and sym in positions:
                    pos = positions[sym]
                    exit_price = price
                    pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                    cash += pos["qty"] * exit_price

                    trades.append(BacktestTrade(
                        symbol=sym,
                        side="sell",
                        qty=pos["qty"],
                        entry_price=pos["entry_price"],
                        entry_date=pos["entry_date"],
                        exit_price=exit_price,
                        exit_date=str(date),
                        pnl=pnl,
                        strategy=pos["strategy"],
                    ))
                    del positions[sym]

            # Check stops
            for sym in list(positions.keys()):
                price = current_prices.get(sym, 0)
                pos = positions[sym]
                if price <= 0:
                    continue

                if price <= pos["stop_loss"] or price >= pos["take_profit"]:
                    pnl = (price - pos["entry_price"]) * pos["qty"]
                    cash += pos["qty"] * price

                    reason = "stop_loss" if price <= pos["stop_loss"] else "take_profit"
                    trades.append(BacktestTrade(
                        symbol=sym, side="sell", qty=pos["qty"],
                        entry_price=pos["entry_price"], entry_date=pos["entry_date"],
                        exit_price=price, exit_date=str(date), pnl=pnl,
                        strategy=f"{pos['strategy']}_{reason}",
                    ))
                    del positions[sym]

            # Record equity
            portfolio_value = sum(
                p["qty"] * current_prices.get(s, p["entry_price"])
                for s, p in positions.items()
            )
            equity_curve.append(cash + portfolio_value)

        # Close remaining positions at last prices
        for sym, pos in positions.items():
            price = current_prices.get(sym, pos["entry_price"])
            pnl = (price - pos["entry_price"]) * pos["qty"]
            cash += pos["qty"] * price
            trades.append(BacktestTrade(
                symbol=sym, side="sell", qty=pos["qty"],
                entry_price=pos["entry_price"], entry_date=pos["entry_date"],
                exit_price=price, exit_date=str(dates[-1]), pnl=pnl,
                strategy=f"{pos['strategy']}_close",
            ))

        final_equity = cash
        return self._compute_metrics(
            initial_capital, final_equity, equity_curve, trades,
            str(dates[warmup]), str(dates[-1]),
        )

    def _compute_metrics(
        self,
        initial: float,
        final: float,
        equity_curve: list[float],
        trades: list[BacktestTrade],
        start: str,
        end: str,
    ) -> BacktestResult:
        total_return = (final / initial - 1) * 100

        # Annualized return
        n_days = len(equity_curve) or 1
        annual_return = ((final / initial) ** (252 / n_days) - 1) * 100

        # Daily returns for Sharpe/Sortino
        if len(equity_curve) > 1:
            eq = np.array(equity_curve)
            daily_returns = np.diff(eq) / eq[:-1]
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

            downside = daily_returns[daily_returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1
            sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sharpe = sortino = 0

        # Max drawdown
        if equity_curve:
            eq = np.array(equity_curve)
            running_max = np.maximum.accumulate(eq)
            drawdowns = (running_max - eq) / running_max * 100
            max_dd = float(np.max(drawdowns))
        else:
            max_dd = 0

        # Trade stats
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

        return BacktestResult(
            start_date=start,
            end_date=end,
            initial_capital=initial,
            final_equity=final,
            total_return_pct=total_return,
            annual_return_pct=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_pnl=np.mean(pnls) if pnls else 0,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            equity_curve=equity_curve,
            trades=trades,
        )

    def _empty_result(self, capital: float) -> BacktestResult:
        return BacktestResult(
            start_date="", end_date="", initial_capital=capital,
            final_equity=capital, total_return_pct=0, annual_return_pct=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown_pct=0, win_rate=0,
            profit_factor=0, total_trades=0, avg_trade_pnl=0, avg_win=0, avg_loss=0,
        )
