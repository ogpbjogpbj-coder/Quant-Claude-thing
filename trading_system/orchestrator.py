"""Main trading orchestrator — the brain of the system.

Coordinates data ingestion, signal generation, risk management,
and order execution on a scheduled loop.
"""

import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from trading_system.config import TradingConfig, load_config
from trading_system.cost_model import ExecutionCostModel
from trading_system.data_ingestion import DataIngestion
from trading_system.execution import ExecutionEngine
from trading_system.portfolio import PortfolioManager
from trading_system.regime_detector import RegimeDetector
from trading_system.risk_manager import RiskManager
from trading_system.signal_aggregator import SignalAggregator
from trading_system.signal_decay import SignalDecayTracker
from trading_system.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    MLEnsembleStrategy,
    VolatilityBreakoutStrategy,
    TrendFollowingStrategy,
    PairsTradingStrategy,
    SentimentStrategy,
)
from trading_system.utils.db import init_db
from trading_system.utils.logger import setup_logger, log_trade


class TradingOrchestrator:
    """Main trading loop coordinator."""

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or load_config()
        self._running = False
        self._scheduler: Optional[BackgroundScheduler] = None

        # Validate credentials
        if not self.config.alpaca.is_configured:
            raise ValueError(
                "Alpaca API credentials not configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        # Initialize components
        setup_logger()
        init_db()

        logger.info(f"Initializing trading system in {self.config.mode} mode")
        if self.config.alpaca.is_paper:
            logger.info("*** PAPER TRADING MODE ***")
        else:
            logger.warning("*** LIVE TRADING MODE — REAL MONEY ***")

        self.data = DataIngestion(self.config)
        self.execution = ExecutionEngine(self.config.alpaca, self.config.execution)
        self.risk_manager = RiskManager(self.config.risk)
        self.portfolio = PortfolioManager(
            self.config, self.execution, self.risk_manager
        )
        self.aggregator = SignalAggregator(self.config.strategies)

        # New: Regime detection
        self.regime_detector = RegimeDetector()

        # New: Execution cost model
        self.cost_model = ExecutionCostModel(self.config.execution)

        # New: Signal decay tracker
        base_weights = {
            "momentum": self.config.strategies.momentum.weight,
            "mean_reversion": self.config.strategies.mean_reversion.weight,
            "ml_ensemble": self.config.strategies.ml_ensemble.weight,
            "volatility_breakout": self.config.strategies.volatility_breakout.weight,
            "trend_following": self.config.strategies.trend_following.weight,
            "pairs_trading": self.config.strategies.pairs_trading.weight,
            "sentiment": self.config.strategies.sentiment.weight,
        }
        self.signal_decay = SignalDecayTracker(base_weights)

        # Initialize strategies
        self.strategies = []
        sc = self.config.strategies

        if sc.momentum.enabled:
            self.strategies.append(MomentumStrategy(sc.momentum))
        if sc.mean_reversion.enabled:
            self.strategies.append(MeanReversionStrategy(sc.mean_reversion))
        if sc.ml_ensemble.enabled:
            self.strategies.append(MLEnsembleStrategy(sc.ml_ensemble))
        if sc.volatility_breakout.enabled:
            self.strategies.append(VolatilityBreakoutStrategy(sc.volatility_breakout))
        if sc.trend_following.enabled:
            self.strategies.append(TrendFollowingStrategy(sc.trend_following))
        if sc.pairs_trading.enabled:
            self.strategies.append(PairsTradingStrategy(sc.pairs_trading))
        if sc.sentiment.enabled:
            self.strategies.append(
                SentimentStrategy(
                    sc.sentiment,
                    api_key=self.config.alpaca.api_key,
                    secret_key=self.config.alpaca.secret_key,
                )
            )

        logger.info(f"Loaded {len(self.strategies)} strategies: "
                     f"{[s.name for s in self.strategies]}")
        logger.info(f"Trading universe: {len(self.config.universe)} symbols")

    def _initialize_risk(self) -> None:
        """Set risk manager starting reference points."""
        account = self.execution.get_account()
        equity = account.get("equity", 0)
        if equity > 0:
            self.risk_manager.initialize(equity)
        else:
            logger.error("Could not get account equity for risk initialization")

    def run_cycle(self) -> None:
        """Execute one complete trading cycle."""
        cycle_start = datetime.utcnow()
        logger.info("=" * 60)
        logger.info(f"Trading cycle started at {cycle_start.isoformat()}")

        try:
            # 1. Check if market is open
            if not self.execution.is_market_open():
                logger.info("Market is closed, skipping cycle")
                return

            # 2. Sync portfolio state
            self.portfolio.sync_positions()
            logger.info(
                f"Portfolio: equity=${self.portfolio.equity:,.2f}, "
                f"cash=${self.portfolio.cash:,.2f}, "
                f"positions={len(self.portfolio.tracked)}"
            )

            # 3. Check circuit breakers
            if not self.risk_manager.check_circuit_breakers(self.portfolio.equity):
                logger.warning("Circuit breaker active — no new trades")
                # Still check stops on existing positions
                self._check_stops()
                return

            # 4. Fetch enriched market data
            enriched = self.data.get_enriched_data()
            if not enriched:
                logger.warning("No market data available")
                return

            # 5. Regime detection — adjusts strategy weights
            regime_state = self.regime_detector.detect(enriched)
            self.aggregator.set_regime_adjustments(regime_state.strategy_weights)
            logger.info(
                f"Regime: {regime_state.regime.name} "
                f"(conf={regime_state.confidence:.2f}, "
                f"pos_scale={regime_state.position_scale:.2f})"
            )

            # 6. Get latest prices
            prices = self.data.get_latest_quotes(list(enriched.keys()))
            if not prices:
                # Fallback to last close
                prices = {sym: df.iloc[-1]["close"] for sym, df in enriched.items() if not df.empty}

            # 7. Check stops on existing positions
            self._check_stops_with_prices(prices)

            # 8. Generate signals from all strategies
            current_positions = self.portfolio.get_current_positions_qty()
            all_signals = []

            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(enriched, current_positions)
                    all_signals.extend(signals)
                    if signals:
                        logger.info(
                            f"  {strategy.name}: {len(signals)} signals "
                            f"({sum(1 for s in signals if s.is_buy)} buy, "
                            f"{sum(1 for s in signals if s.is_sell)} sell)"
                        )
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed: {e}")

            # 9. Aggregate signals (regime-adjusted weights applied internally)
            consensus = self.aggregator.aggregate(all_signals)
            logger.info(f"Consensus: {len(consensus)} actionable signals")

            # 10. Cost model filter — skip trades where cost > expected alpha
            cost_filtered = []
            for sig in consensus:
                price = prices.get(sig.symbol, 0)
                if price <= 0:
                    continue
                vol = sig.metadata.get("volatility_21d", 0.2)
                avg_vol = sig.metadata.get("avg_volume", 1_000_000)
                est_qty = (self.portfolio.equity * 0.03) / price  # rough estimate
                cost_est = self.cost_model.estimate_cost(
                    symbol=sig.symbol,
                    qty=est_qty,
                    price=price,
                    side="buy" if sig.is_buy else "sell",
                    volatility=vol,
                    avg_volume=avg_vol,
                )
                if self.cost_model.should_trade(sig.strength, cost_est):
                    cost_filtered.append(sig)
                else:
                    logger.info(
                        f"  Skipping {sig.symbol}: cost {cost_est.total_cost_pct:.4f}% "
                        f"> edge {sig.strength:.4f}"
                    )
            consensus = cost_filtered

            # 11. Apply regime position scale to signals
            if regime_state.position_scale < 1.0:
                for sig in consensus:
                    sig.metadata["regime_scale"] = regime_state.position_scale

            for sig in consensus[:5]:  # Log top 5
                logger.info(
                    f"  {sig.symbol}: dir={sig.direction:+.3f} "
                    f"conf={sig.confidence:.3f} str={sig.strength:.3f} "
                    f"[{sig.strategy}]"
                )

            # 12. Execute trades
            if consensus:
                order_ids = self.portfolio.process_signals(consensus, prices, enriched)
                logger.info(f"Placed {len(order_ids)} orders")

            # 13. Take portfolio snapshot
            self.portfolio.take_snapshot()

            # 14. Log signal decay stats periodically
            decay_stats = self.signal_decay.get_strategy_stats()
            for strat, stats in decay_stats.items():
                if stats["signal_count"] > 0:
                    logger.debug(
                        f"  Decay[{strat}]: hit={stats['hit_rate']:.2f} "
                        f"IC={stats['ic']:.3f} score={stats['score']:.3f}"
                    )

            elapsed = (datetime.utcnow() - cycle_start).total_seconds()
            logger.info(f"Cycle completed in {elapsed:.1f}s")

        except Exception as e:
            logger.exception(f"Trading cycle failed: {e}")

    def _check_stops(self) -> None:
        """Check stops using broker prices."""
        try:
            positions = self.execution.get_positions()
            prices = {sym: p["current_price"] for sym, p in positions.items()}
            self._check_stops_with_prices(prices)
        except Exception as e:
            logger.error(f"Stop check failed: {e}")

    def _check_stops_with_prices(self, prices: dict[str, float]) -> None:
        """Check all position stops against given prices."""
        triggered = self.portfolio.check_stops(prices)
        if triggered:
            logger.warning(f"Stops triggered for: {triggered}")

    def start(self) -> None:
        """Start the trading system with scheduled execution."""
        logger.info("Starting QuantClaude Trading System...")

        self._initialize_risk()
        self._running = True

        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        # Scheduler
        self._scheduler = BackgroundScheduler()

        # Main trading cycle
        interval = self.config.schedule.rebalance_interval_minutes
        self._scheduler.add_job(
            self.run_cycle,
            IntervalTrigger(minutes=interval),
            id="trading_cycle",
            name="Main Trading Cycle",
            max_instances=1,
        )

        # Portfolio snapshot every 5 minutes
        self._scheduler.add_job(
            self._safe_snapshot,
            IntervalTrigger(minutes=5),
            id="snapshot",
            name="Portfolio Snapshot",
        )

        # Daily risk reset at 9:30 AM ET (market open)
        self._scheduler.add_job(
            self._daily_reset,
            CronTrigger(hour=13, minute=30),  # UTC = ET + 4/5
            id="daily_reset",
            name="Daily Risk Reset",
        )

        self._scheduler.start()

        # Run first cycle immediately
        logger.info("Running initial trading cycle...")
        self.run_cycle()

        logger.info(
            f"Scheduler running. Trading cycle every {interval} minutes. "
            f"Press Ctrl+C to stop."
        )

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop the trading system."""
        logger.info("Stopping trading system...")
        self._running = False

        if self._scheduler:
            self._scheduler.shutdown(wait=False)

        # Final snapshot
        try:
            self.portfolio.take_snapshot()
        except Exception:
            pass

        # Log final slippage stats
        try:
            slippage = self.cost_model.get_slippage_stats()
            if slippage:
                logger.info(f"Session slippage stats: {slippage}")
        except Exception:
            pass

        logger.info("Trading system stopped")

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def _safe_snapshot(self) -> None:
        try:
            if self.execution.is_market_open():
                self.portfolio.take_snapshot()
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")

    def _daily_reset(self) -> None:
        try:
            account = self.execution.get_account()
            equity = account.get("equity", 0)
            self.risk_manager.reset_daily(equity)
        except Exception as e:
            logger.error(f"Daily reset failed: {e}")

    def emergency_shutdown(self, reason: str = "manual") -> None:
        """Emergency: close all positions and halt."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        self.execution.cancel_all_orders()
        self.execution.close_all_positions(reason=f"emergency: {reason}")
        self.stop()
