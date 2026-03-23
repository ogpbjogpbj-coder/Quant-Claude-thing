"""Main trading orchestrator — the brain of the system.

Coordinates data ingestion, signal generation, risk management,
and order execution on a scheduled loop. Now with adaptive learning:
every trade is journaled, patterns are mined, and the system evolves
its own strategies over time.
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
from trading_system.regime_detector import RegimeDetector, RegimeState
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
    AdaptiveStrategy,
)
from trading_system.strategies.base import Signal
from trading_system.trade_journal import TradeJournal
from trading_system.strategy_evolver import StrategyEvolver
from trading_system.utils.db import init_db
from trading_system.utils.logger import setup_logger, log_trade


class TradingOrchestrator:
    """Main trading loop coordinator with adaptive learning."""

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

        # Regime detection
        self.regime_detector = RegimeDetector()
        self._current_regime: Optional[RegimeState] = None

        # Execution cost model
        self.cost_model = ExecutionCostModel(self.config.execution)

        # Signal decay tracker
        base_weights = {
            "momentum": self.config.strategies.momentum.weight,
            "mean_reversion": self.config.strategies.mean_reversion.weight,
            "ml_ensemble": self.config.strategies.ml_ensemble.weight,
            "volatility_breakout": self.config.strategies.volatility_breakout.weight,
            "trend_following": self.config.strategies.trend_following.weight,
            "pairs_trading": self.config.strategies.pairs_trading.weight,
            "sentiment": self.config.strategies.sentiment.weight,
            "adaptive": self.config.strategies.adaptive.weight,
        }
        self.signal_decay = SignalDecayTracker(base_weights)

        # === NEW: Adaptive learning components ===
        self.trade_journal = TradeJournal()
        self.strategy_evolver = StrategyEvolver()

        # Track which signals led to which trades (for journal entries)
        self._pending_signals: dict[str, dict] = {}
        # Cache enriched data for journal context
        self._last_enriched: dict = {}

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
        if sc.adaptive.enabled:
            self.strategies.append(AdaptiveStrategy(sc.adaptive))

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
                self._check_stops()
                return

            # 4. Fetch enriched market data
            enriched = self.data.get_enriched_data()
            if not enriched:
                logger.warning("No market data available")
                return
            self._last_enriched = enriched

            # 5. Regime detection — adjusts strategy weights
            regime_state = self.regime_detector.detect(enriched)
            self._current_regime = regime_state
            self.aggregator.set_regime_adjustments(regime_state.strategy_weights)
            logger.info(
                f"Regime: {regime_state.regime.name} "
                f"(conf={regime_state.confidence:.2f}, "
                f"pos_scale={regime_state.position_scale:.2f})"
            )

            # 6. Get latest prices
            prices = self.data.get_latest_quotes(list(enriched.keys()))
            if not prices:
                prices = {sym: df.iloc[-1]["close"] for sym, df in enriched.items() if not df.empty}

            # 7. Check stops on existing positions — journal exits
            self._check_stops_with_journal(prices)

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

            # 10. Cost model filter
            cost_filtered = []
            for sig in consensus:
                price = prices.get(sig.symbol, 0)
                if price <= 0:
                    continue
                vol = sig.metadata.get("volatility_21d", 0.2)
                avg_vol = sig.metadata.get("avg_volume", 1_000_000)
                est_qty = (self.portfolio.equity * 0.03) / price
                cost_est = self.cost_model.estimate_cost(
                    symbol=sig.symbol, qty=est_qty, price=price,
                    side="buy" if sig.is_buy else "sell",
                    volatility=vol, avg_volume=avg_vol,
                )
                if self.cost_model.should_trade(sig.strength, cost_est):
                    cost_filtered.append(sig)
                else:
                    logger.info(
                        f"  Skipping {sig.symbol}: cost {cost_est.total_cost_pct:.4f}% "
                        f"> edge {sig.strength:.4f}"
                    )
            consensus = cost_filtered

            # 11. Apply regime position scale
            if regime_state.position_scale < 1.0:
                for sig in consensus:
                    sig.metadata["regime_scale"] = regime_state.position_scale

            for sig in consensus[:5]:
                logger.info(
                    f"  {sig.symbol}: dir={sig.direction:+.3f} "
                    f"conf={sig.confidence:.3f} str={sig.strength:.3f} "
                    f"[{sig.strategy}]"
                )

            # 12. Store signal context for journal BEFORE executing
            for sig in consensus:
                if sig.is_buy:
                    self._pending_signals[sig.symbol] = {
                        "signal": sig,
                        "all_signals": [s for s in all_signals if s.symbol == sig.symbol],
                        "regime_state": regime_state,
                        "enriched": enriched.get(sig.symbol),
                    }

            # 13. Execute trades
            if consensus:
                order_ids = self.portfolio.process_signals(consensus, prices, enriched)
                logger.info(f"Placed {len(order_ids)} orders")

                # 14. Journal entries for new positions
                self._journal_new_entries(consensus, prices)

            # 15. Journal exits for closed positions
            self._journal_closed_positions(prices)

            # 16. Take portfolio snapshot
            self.portfolio.take_snapshot()

            # 17. Feed signal outcomes to decay tracker
            self._update_signal_decay(prices)

            # 18. Log signal decay stats
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

    # =========================================================================
    # Trade Journal Integration
    # =========================================================================

    def _journal_new_entries(
        self,
        consensus: list[Signal],
        prices: dict[str, float],
    ) -> None:
        """Record journal entries for trades that were just placed."""
        for sig in consensus:
            if not sig.is_buy:
                continue
            # Only journal if we actually have a position now
            if sig.symbol not in self.portfolio.tracked:
                continue

            ctx = self._pending_signals.get(sig.symbol, {})
            price = prices.get(sig.symbol, 0)

            try:
                journal_id = self.trade_journal.record_entry(
                    symbol=sig.symbol,
                    qty=self.portfolio.tracked[sig.symbol].qty,
                    price=price or self.portfolio.tracked[sig.symbol].avg_entry,
                    signal=sig,
                    regime_state=self._current_regime,
                    enriched_data=self._last_enriched,
                    all_signals=ctx.get("all_signals", []),
                )
                if journal_id:
                    # Store journal_id on the tracked position for later exit recording
                    self.portfolio.tracked[sig.symbol].strategy = (
                        f"{sig.strategy}|jid:{journal_id}"
                    )
                    logger.debug(f"Journaled entry for {sig.symbol} (id={journal_id})")
            except Exception as e:
                logger.warning(f"Failed to journal entry for {sig.symbol}: {e}")

        # Clear pending signals
        self._pending_signals.clear()

    def _journal_closed_positions(self, prices: dict[str, float]) -> None:
        """Check for positions that were closed this cycle and journal exits."""
        try:
            broker_positions = self.execution.get_positions()
            broker_symbols = set(broker_positions.keys())

            # Find symbols that were tracked but are no longer at broker
            for sym in list(self._recently_closed):
                price = prices.get(sym, 0)
                exit_info = self._recently_closed[sym]
                try:
                    self.trade_journal.record_exit(
                        symbol=sym,
                        exit_price=price or exit_info.get("price", 0),
                        exit_reason=exit_info.get("reason", "unknown"),
                        tracked_position=exit_info.get("tracked"),
                    )
                    logger.debug(f"Journaled exit for {sym}: {exit_info.get('reason')}")
                except Exception as e:
                    logger.warning(f"Failed to journal exit for {sym}: {e}")

            self._recently_closed.clear()
        except Exception as e:
            logger.warning(f"Journal closed positions check failed: {e}")

    def _check_stops_with_journal(self, prices: dict[str, float]) -> None:
        """Check stops and record exits in the journal."""
        # Snapshot tracked positions before stop check
        pre_check = dict(self.portfolio.tracked)

        triggered = self.portfolio.check_stops(prices)

        if triggered:
            logger.warning(f"Stops triggered for: {triggered}")
            for sym in triggered:
                tracked = pre_check.get(sym)
                price = prices.get(sym, 0)
                reason = "stop_loss"
                if tracked and tracked.take_profit > 0 and price >= tracked.take_profit:
                    reason = "take_profit"
                try:
                    self.trade_journal.record_exit(
                        symbol=sym,
                        exit_price=price,
                        exit_reason=reason,
                        tracked_position=tracked,
                    )
                    logger.debug(f"Journaled stop exit for {sym}: {reason}")
                except Exception as e:
                    logger.warning(f"Failed to journal stop exit for {sym}: {e}")

    def _update_signal_decay(self, prices: dict[str, float]) -> None:
        """Feed recent trade outcomes to the signal decay tracker."""
        try:
            for sym, tracked in self.portfolio.tracked.items():
                if sym in prices and tracked.avg_entry > 0:
                    current_return = (prices[sym] - tracked.avg_entry) / tracked.avg_entry
                    # Extract original strategy name
                    strat_name = tracked.strategy.split("|")[0] if tracked.strategy else ""
                    # Try to extract individual strategies from consensus name
                    if strat_name.startswith("consensus("):
                        inner = strat_name[10:].rstrip(")")
                        for s in inner.split(","):
                            s = s.strip()
                            if s:
                                self.signal_decay.record_signal_outcome(
                                    s, 0.5, current_return
                                )
                    elif strat_name:
                        self.signal_decay.record_signal_outcome(
                            strat_name, 0.5, current_return
                        )
        except Exception as e:
            logger.warning(f"Signal decay update failed: {e}")

    # =========================================================================
    # Strategy Evolution (runs periodically)
    # =========================================================================

    def _run_evolution(self) -> None:
        """Run the strategy evolution cycle."""
        try:
            if not self.execution.is_market_open():
                return

            logger.info("Running strategy evolution cycle...")

            # Get latest enriched data
            enriched = self._last_enriched or self.data.get_enriched_data()

            # Run evolution
            report = self.strategy_evolver.evolve_strategies(enriched)

            if report.get("trades_analyzed", 0) > 0:
                logger.info(
                    f"Evolution complete: analyzed {report['trades_analyzed']} trades, "
                    f"promoted {report.get('rules_promoted', 0)} rules, "
                    f"demoted {report.get('rules_demoted', 0)} rules, "
                    f"{report.get('active_rules', 0)} active"
                )
                for insight in report.get("insights", [])[:5]:
                    logger.info(f"  Insight: {insight}")
                for rule in report.get("top_rules", [])[:3]:
                    logger.info(
                        f"  Top rule: {rule['name']} "
                        f"(win={rule['win_rate']:.0%}, pf={rule['profit_factor']:.2f}, "
                        f"n={rule['sample_size']})"
                    )

            # Log trade journal summary
            perf = self.trade_journal.get_strategy_performance()
            if perf.get("total_trades", 0) > 0:
                logger.info(
                    f"Journal stats: {perf['total_trades']} trades, "
                    f"win_rate={perf.get('win_rate', 0):.1%}, "
                    f"avg_pnl={perf.get('avg_pnl_pct', 0):.2%}"
                )

            # Log lesson insights
            lessons = self.trade_journal.get_lessons_summary()
            if lessons:
                for key, val in list(lessons.items())[:3]:
                    if isinstance(val, (int, float)):
                        logger.info(f"  Lesson: {key} = {val:.1%}" if val < 1 else f"  Lesson: {key} = {val}")

        except Exception as e:
            logger.error(f"Strategy evolution failed: {e}")

    # =========================================================================
    # Stop checks
    # =========================================================================

    def _check_stops(self) -> None:
        """Check stops using broker prices."""
        try:
            positions = self.execution.get_positions()
            prices = {sym: p["current_price"] for sym, p in positions.items()}
            self._check_stops_with_journal(prices)
        except Exception as e:
            logger.error(f"Stop check failed: {e}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @property
    def _recently_closed(self) -> dict:
        """Lazy-init dict tracking recently closed positions."""
        if not hasattr(self, "__recently_closed"):
            self.__recently_closed: dict[str, dict] = {}
        return self.__recently_closed

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

        # Strategy evolution every 6 hours during market hours
        self._scheduler.add_job(
            self._run_evolution,
            IntervalTrigger(hours=6),
            id="evolution",
            name="Strategy Evolution",
            max_instances=1,
        )

        self._scheduler.start()

        # Run first cycle immediately
        logger.info("Running initial trading cycle...")
        self.run_cycle()

        logger.info(
            f"Scheduler running. Trading cycle every {interval} minutes. "
            f"Strategy evolution every 6 hours. Press Ctrl+C to stop."
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

        # Log final journal stats
        try:
            perf = self.trade_journal.get_strategy_performance()
            if perf.get("total_trades", 0) > 0:
                logger.info(f"Session journal: {perf}")
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
