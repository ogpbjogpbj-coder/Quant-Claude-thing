"""
Post-trade analysis and journaling system.

Captures full context for every trade — entry conditions, market regime,
technical indicators, exit details, and auto-generated lessons — so the
AI trading system can learn from each trade over time.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Optional

import numpy as np
from loguru import logger

from trading_system.utils.db import DB_PATH, get_connection


# Technical indicators to extract from enriched data for journal entries
_INDICATOR_KEYS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
    "adx",
    "atr_pct",
    "volume_ratio",
    "volatility_21d",
    "returns_5d",
    "returns_21d",
    "price_vs_sma50",
    "price_vs_sma200",
    "stoch_k",
    "obv",
]


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if not possible."""
    if value is None:
        return None
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


class TradeJournal:
    """Records, analyses, and extracts lessons from every completed trade."""

    def __init__(self) -> None:
        self._pending: dict[str, dict] = {}  # symbol -> partial journal row
        self._init_tables()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        """Create the trade_journal and evolved_rules tables if missing."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- Identity
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    -- Entry context
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_qty REAL NOT NULL,
                    entry_signal_direction REAL,
                    entry_signal_confidence REAL,
                    entry_signal_strength REAL,
                    -- Market context at entry
                    entry_regime TEXT,
                    entry_regime_confidence REAL,
                    entry_volatility_regime TEXT,
                    entry_trend_strength REAL,
                    entry_correlation_level REAL,
                    -- Technical indicators at entry (JSON)
                    entry_indicators TEXT,
                    -- Exit context
                    exit_time TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    -- Outcome
                    pnl_dollars REAL,
                    pnl_pct REAL,
                    holding_period_hours REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    -- Strategy agreement
                    agreeing_strategies TEXT,
                    disagreeing_strategies TEXT,
                    -- Post-trade analysis
                    outcome_label TEXT,
                    lessons TEXT,
                    -- Timestamps
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evolved_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL UNIQUE,
                    rule_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    performance_score REAL,
                    times_triggered INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_updated TEXT,
                    active INTEGER DEFAULT 1
                );
            """)
            conn.commit()
            conn.close()
            logger.info("Trade journal tables initialised")
        except Exception as e:
            logger.error(f"Failed to initialise trade journal tables: {e}")

    # ------------------------------------------------------------------
    # Entry recording
    # ------------------------------------------------------------------

    def record_entry(
        self,
        symbol: str,
        qty: float,
        price: float,
        signal: Any,
        regime_state: Any,
        enriched_data: Any,
        all_signals: list,
    ) -> Optional[int]:
        """Record a new trade entry with full market context.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares / units.
            price: Fill price.
            signal: A Signal dataclass instance for this trade.
            regime_state: A RegimeState dataclass with current regime info.
            enriched_data: Dict of symbol -> DataFrame with technical indicators,
                           or a single DataFrame for this symbol.
            all_signals: List of all Signal objects generated in this cycle.

        Returns:
            The trade_journal row id, or None on failure.
        """
        try:
            now = datetime.utcnow().isoformat()

            # --- Signal fields ---
            sig_direction = _safe_float(getattr(signal, "direction", None))
            sig_confidence = _safe_float(getattr(signal, "confidence", None))
            sig_strength = _safe_float(getattr(signal, "strength", None))
            strategy = getattr(signal, "strategy", "") or ""

            # --- Regime fields ---
            regime_name = None
            regime_conf = None
            vol_regime = None
            trend_str = None
            corr_level = None
            if regime_state is not None:
                regime_obj = getattr(regime_state, "regime", None)
                regime_name = regime_obj.name if hasattr(regime_obj, "name") else str(regime_obj)
                regime_conf = _safe_float(getattr(regime_state, "confidence", None))
                vol_regime = getattr(regime_state, "volatility_regime", None)
                trend_str = _safe_float(getattr(regime_state, "trend_strength", None))
                corr_level = _safe_float(getattr(regime_state, "correlation_level", None))

            # --- Technical indicators ---
            indicators = self._extract_indicators(symbol, enriched_data)
            indicators_json = json.dumps(indicators)

            # --- Strategy agreement ---
            agreeing, disagreeing = self._classify_agreement(signal, symbol, all_signals)

            # --- Insert into DB ---
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO trade_journal
                   (symbol, strategy, entry_time, entry_price, entry_qty,
                    entry_signal_direction, entry_signal_confidence, entry_signal_strength,
                    entry_regime, entry_regime_confidence, entry_volatility_regime,
                    entry_trend_strength, entry_correlation_level,
                    entry_indicators,
                    agreeing_strategies, disagreeing_strategies,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol, strategy, now, price, qty,
                    sig_direction, sig_confidence, sig_strength,
                    regime_name, regime_conf, vol_regime,
                    trend_str, corr_level,
                    indicators_json,
                    json.dumps(agreeing), json.dumps(disagreeing),
                    now,
                ),
            )
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # Keep in-memory for exit matching
            self._pending[symbol] = {
                "id": trade_id,
                "entry_time": now,
                "entry_price": price,
                "entry_qty": qty,
                "strategy": strategy,
                "signal_direction": sig_direction,
                "indicators": indicators,
            }

            logger.info(
                f"Trade journal entry recorded: {symbol} id={trade_id} "
                f"price={price} qty={qty} strategy={strategy}"
            )
            return trade_id

        except Exception as e:
            logger.error(f"Failed to record trade journal entry for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Exit recording
    # ------------------------------------------------------------------

    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        tracked_position: Any = None,
    ) -> bool:
        """Record a trade exit and compute post-trade analysis.

        Args:
            symbol: Ticker symbol.
            exit_price: Fill price on exit.
            exit_reason: One of stop_loss, take_profit, signal_reversal,
                         regime_change, manual.
            tracked_position: Optional TrackedPosition for MFE/MAE calculation.

        Returns:
            True on success, False on failure.
        """
        try:
            pending = self._pending.pop(symbol, None)
            if pending is None:
                logger.warning(f"No pending trade journal entry for {symbol}, attempting DB lookup")
                pending = self._lookup_pending_from_db(symbol)
                if pending is None:
                    logger.error(f"Cannot record exit for {symbol}: no pending entry found")
                    return False

            trade_id = pending["id"]
            entry_price = pending["entry_price"]
            entry_qty = pending["entry_qty"]
            entry_time_str = pending["entry_time"]
            signal_direction = pending.get("signal_direction")
            indicators = pending.get("indicators", {})

            now = datetime.utcnow()
            exit_time = now.isoformat()

            # --- P&L ---
            if signal_direction is not None and signal_direction < 0:
                # Short trade
                pnl_dollars = (entry_price - exit_price) * entry_qty
            else:
                pnl_dollars = (exit_price - entry_price) * entry_qty

            pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0.0
            if signal_direction is not None and signal_direction < 0:
                pnl_pct = -pnl_pct

            # --- Holding period ---
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                holding_hours = (now - entry_dt).total_seconds() / 3600.0
            except Exception:
                holding_hours = None

            # --- MFE / MAE ---
            mfe, mae = self._compute_excursions(
                entry_price, signal_direction, tracked_position
            )

            # --- Outcome classification ---
            outcome_label = self._classify_outcome(pnl_pct)

            # --- Auto-generated lessons ---
            lessons = self._generate_lessons(
                pnl_pct=pnl_pct,
                mfe=mfe,
                mae=mae,
                holding_hours=holding_hours,
                exit_reason=exit_reason,
                signal_direction=signal_direction,
                indicators=indicators,
            )

            # --- Update DB ---
            conn = get_connection()
            conn.execute(
                """UPDATE trade_journal
                   SET exit_time = ?,
                       exit_price = ?,
                       exit_reason = ?,
                       pnl_dollars = ?,
                       pnl_pct = ?,
                       holding_period_hours = ?,
                       max_favorable_excursion = ?,
                       max_adverse_excursion = ?,
                       outcome_label = ?,
                       lessons = ?
                   WHERE id = ?""",
                (
                    exit_time, exit_price, exit_reason,
                    pnl_dollars, pnl_pct, holding_hours,
                    mfe, mae,
                    outcome_label, json.dumps(lessons),
                    trade_id,
                ),
            )
            conn.commit()
            conn.close()

            logger.info(
                f"Trade journal exit recorded: {symbol} id={trade_id} "
                f"pnl={pnl_pct:+.2f}% outcome={outcome_label} reason={exit_reason}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to record trade journal exit for {symbol}: {e}")
            return False

    # ------------------------------------------------------------------
    # Analysis queries
    # ------------------------------------------------------------------

    def get_strategy_performance(
        self, strategy: Optional[str] = None, last_n: int = 100
    ) -> dict:
        """Aggregate performance stats, optionally filtered by strategy.

        Returns dict with keys: overall, by_strategy, by_regime.
        """
        try:
            conn = get_connection()
            query = """
                SELECT * FROM trade_journal
                WHERE exit_time IS NOT NULL
                ORDER BY exit_time DESC
                LIMIT ?
            """
            params: list[Any] = [last_n]
            if strategy:
                query = """
                    SELECT * FROM trade_journal
                    WHERE exit_time IS NOT NULL AND strategy = ?
                    ORDER BY exit_time DESC
                    LIMIT ?
                """
                params = [strategy, last_n]

            rows = conn.execute(query, params).fetchall()
            conn.close()

            if not rows:
                return {"overall": {}, "by_strategy": {}, "by_regime": {}}

            trades = [dict(r) for r in rows]
            result: dict[str, Any] = {
                "overall": self._calc_stats(trades),
                "by_strategy": {},
                "by_regime": {},
            }

            # Group by strategy
            strat_groups: dict[str, list] = {}
            for t in trades:
                s = t.get("strategy") or "unknown"
                strat_groups.setdefault(s, []).append(t)
            for s, group in strat_groups.items():
                result["by_strategy"][s] = self._calc_stats(group)

            # Group by regime
            regime_groups: dict[str, list] = {}
            for t in trades:
                r = t.get("entry_regime") or "unknown"
                regime_groups.setdefault(r, []).append(t)
            for r, group in regime_groups.items():
                result["by_regime"][r] = self._calc_stats(group)

            return result

        except Exception as e:
            logger.error(f"get_strategy_performance error: {e}")
            return {"overall": {}, "by_strategy": {}, "by_regime": {}}

    def get_indicator_edge(
        self, indicator_name: str, threshold: float, direction: str = "below"
    ) -> dict:
        """Check win rate when an indicator is above/below a threshold at entry.

        Args:
            indicator_name: Key inside entry_indicators JSON (e.g. "rsi_14").
            threshold: Numeric threshold.
            direction: "below" or "above".

        Returns:
            Dict with total, wins, losses, win_rate, avg_pnl_pct.
        """
        try:
            conn = get_connection()
            rows = conn.execute(
                """SELECT entry_indicators, pnl_pct FROM trade_journal
                   WHERE exit_time IS NOT NULL AND entry_indicators IS NOT NULL"""
            ).fetchall()
            conn.close()

            matching: list[float] = []
            for row in rows:
                try:
                    indicators = json.loads(row["entry_indicators"])
                except (json.JSONDecodeError, TypeError):
                    continue
                val = indicators.get(indicator_name)
                if val is None:
                    continue
                val = float(val)
                if direction == "below" and val < threshold:
                    matching.append(float(row["pnl_pct"] or 0))
                elif direction == "above" and val >= threshold:
                    matching.append(float(row["pnl_pct"] or 0))

            if not matching:
                return {
                    "indicator": indicator_name,
                    "threshold": threshold,
                    "direction": direction,
                    "total": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "avg_pnl_pct": 0.0,
                }

            wins = sum(1 for p in matching if p > 0)
            losses = sum(1 for p in matching if p <= 0)
            return {
                "indicator": indicator_name,
                "threshold": threshold,
                "direction": direction,
                "total": len(matching),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(matching) if matching else 0.0,
                "avg_pnl_pct": float(np.mean(matching)),
            }

        except Exception as e:
            logger.error(f"get_indicator_edge error: {e}")
            return {
                "indicator": indicator_name,
                "threshold": threshold,
                "direction": direction,
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
            }

    def get_regime_performance(self) -> dict:
        """Performance breakdown by entry regime type."""
        try:
            conn = get_connection()
            rows = conn.execute(
                """SELECT * FROM trade_journal
                   WHERE exit_time IS NOT NULL"""
            ).fetchall()
            conn.close()

            if not rows:
                return {}

            trades = [dict(r) for r in rows]
            regime_groups: dict[str, list] = {}
            for t in trades:
                r = t.get("entry_regime") or "unknown"
                regime_groups.setdefault(r, []).append(t)

            result = {}
            for regime, group in regime_groups.items():
                result[regime] = self._calc_stats(group)

            return result

        except Exception as e:
            logger.error(f"get_regime_performance error: {e}")
            return {}

    def get_lessons_summary(self, last_n: int = 50) -> dict:
        """Aggregate lesson flags across recent completed trades.

        Returns dict mapping lesson keys to aggregate stats like:
        {"volume_confirmed": {"total": 40, "true_count": 28, "rate": 0.70,
                              "avg_pnl_when_true": 1.2, "avg_pnl_when_false": -0.8}}
        """
        try:
            conn = get_connection()
            rows = conn.execute(
                """SELECT lessons, pnl_pct FROM trade_journal
                   WHERE exit_time IS NOT NULL AND lessons IS NOT NULL
                   ORDER BY exit_time DESC
                   LIMIT ?""",
                (last_n,),
            ).fetchall()
            conn.close()

            if not rows:
                return {}

            # Collect per-lesson-key stats
            lesson_data: dict[str, dict] = {}
            for row in rows:
                try:
                    lessons = json.loads(row["lessons"])
                except (json.JSONDecodeError, TypeError):
                    continue
                pnl = float(row["pnl_pct"] or 0)
                for key, value in lessons.items():
                    if key not in lesson_data:
                        lesson_data[key] = {
                            "total": 0,
                            "true_count": 0,
                            "pnl_when_true": [],
                            "pnl_when_false": [],
                        }
                    lesson_data[key]["total"] += 1
                    if value:
                        lesson_data[key]["true_count"] += 1
                        lesson_data[key]["pnl_when_true"].append(pnl)
                    else:
                        lesson_data[key]["pnl_when_false"].append(pnl)

            result = {}
            for key, data in lesson_data.items():
                total = data["total"]
                true_count = data["true_count"]
                result[key] = {
                    "total": total,
                    "true_count": true_count,
                    "rate": true_count / total if total else 0.0,
                    "avg_pnl_when_true": (
                        float(np.mean(data["pnl_when_true"]))
                        if data["pnl_when_true"]
                        else 0.0
                    ),
                    "avg_pnl_when_false": (
                        float(np.mean(data["pnl_when_false"]))
                        if data["pnl_when_false"]
                        else 0.0
                    ),
                }

            return result

        except Exception as e:
            logger.error(f"get_lessons_summary error: {e}")
            return {}

    def get_pattern_stats(self) -> list[dict]:
        """Find indicator thresholds that best separate winners from losers.

        For each indicator, tests a range of percentile-based thresholds and
        returns the split with the greatest win-rate difference.

        Returns:
            List of dicts with: indicator, threshold, win_rate_above,
            win_rate_below, sample_size.
        """
        try:
            conn = get_connection()
            rows = conn.execute(
                """SELECT entry_indicators, pnl_pct FROM trade_journal
                   WHERE exit_time IS NOT NULL AND entry_indicators IS NOT NULL"""
            ).fetchall()
            conn.close()

            if not rows:
                return []

            # Build arrays per indicator
            indicator_values: dict[str, list[tuple[float, float]]] = {}
            for row in rows:
                try:
                    indicators = json.loads(row["entry_indicators"])
                except (json.JSONDecodeError, TypeError):
                    continue
                pnl = float(row["pnl_pct"] or 0)
                for key, val in indicators.items():
                    if val is None:
                        continue
                    try:
                        val_f = float(val)
                    except (TypeError, ValueError):
                        continue
                    indicator_values.setdefault(key, []).append((val_f, pnl))

            results = []
            for indicator, pairs in indicator_values.items():
                if len(pairs) < 10:
                    continue
                vals = np.array([p[0] for p in pairs])
                pnls = np.array([p[1] for p in pairs])

                best_diff = 0.0
                best_threshold = 0.0
                best_wr_above = 0.0
                best_wr_below = 0.0

                for pctile in [20, 30, 40, 50, 60, 70, 80]:
                    threshold = float(np.percentile(vals, pctile))
                    above_mask = vals >= threshold
                    below_mask = ~above_mask

                    n_above = int(above_mask.sum())
                    n_below = int(below_mask.sum())
                    if n_above < 3 or n_below < 3:
                        continue

                    wr_above = float(np.mean(pnls[above_mask] > 0))
                    wr_below = float(np.mean(pnls[below_mask] > 0))
                    diff = abs(wr_above - wr_below)

                    if diff > best_diff:
                        best_diff = diff
                        best_threshold = threshold
                        best_wr_above = wr_above
                        best_wr_below = wr_below

                if best_diff > 0:
                    results.append({
                        "indicator": indicator,
                        "threshold": round(best_threshold, 4),
                        "win_rate_above": round(best_wr_above, 4),
                        "win_rate_below": round(best_wr_below, 4),
                        "sample_size": len(pairs),
                    })

            # Sort by biggest separation
            results.sort(key=lambda x: abs(x["win_rate_above"] - x["win_rate_below"]), reverse=True)
            return results

        except Exception as e:
            logger.error(f"get_pattern_stats error: {e}")
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_indicators(self, symbol: str, enriched_data: Any) -> dict:
        """Pull key indicator values from enriched data for a symbol."""
        indicators: dict[str, Optional[float]] = {}
        try:
            # enriched_data may be dict[symbol -> DataFrame] or a DataFrame directly
            df = None
            if isinstance(enriched_data, dict):
                df = enriched_data.get(symbol)
            elif hasattr(enriched_data, "iloc"):
                df = enriched_data

            if df is None or len(df) == 0:
                return indicators

            # Use the last row (most recent bar)
            last = df.iloc[-1]
            for key in _INDICATOR_KEYS:
                val = last.get(key) if hasattr(last, "get") else getattr(last, key, None)
                indicators[key] = _safe_float(val)

        except Exception as e:
            logger.debug(f"Indicator extraction for {symbol}: {e}")

        return indicators

    def _classify_agreement(
        self, signal: Any, symbol: str, all_signals: list
    ) -> tuple[list[str], list[str]]:
        """Determine which other strategies agreed or disagreed with this signal."""
        agreeing: list[str] = []
        disagreeing: list[str] = []
        try:
            my_direction = getattr(signal, "direction", 0)
            my_strategy = getattr(signal, "strategy", "")
            for sig in all_signals:
                if getattr(sig, "symbol", None) != symbol:
                    continue
                other_strategy = getattr(sig, "strategy", "")
                if other_strategy == my_strategy:
                    continue
                other_direction = getattr(sig, "direction", 0)
                if other_direction == 0:
                    continue
                if (my_direction > 0 and other_direction > 0) or (
                    my_direction < 0 and other_direction < 0
                ):
                    agreeing.append(other_strategy)
                else:
                    disagreeing.append(other_strategy)
        except Exception as e:
            logger.debug(f"Agreement classification: {e}")
        return agreeing, disagreeing

    def _compute_excursions(
        self,
        entry_price: float,
        signal_direction: Optional[float],
        tracked_position: Any,
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute MFE and MAE from TrackedPosition extremes."""
        if tracked_position is None or entry_price == 0:
            return None, None
        try:
            highest = getattr(tracked_position, "highest_price", None)
            lowest = getattr(tracked_position, "lowest_price", None)
            if highest is None or lowest is None:
                return None, None

            is_long = signal_direction is None or signal_direction >= 0
            if is_long:
                mfe = (highest - entry_price) / entry_price * 100  # best % gain
                mae = (lowest - entry_price) / entry_price * 100   # worst % drawdown
            else:
                mfe = (entry_price - lowest) / entry_price * 100
                mae = (entry_price - highest) / entry_price * 100

            return _safe_float(mfe), _safe_float(mae)
        except Exception as e:
            logger.debug(f"Excursion computation: {e}")
            return None, None

    @staticmethod
    def _classify_outcome(pnl_pct: float) -> str:
        """Map P&L percentage to a human-readable outcome label."""
        if pnl_pct > 3.0:
            return "big_win"
        elif pnl_pct > 0.0:
            return "small_win"
        elif pnl_pct >= -0.5:
            return "breakeven"
        elif pnl_pct >= -3.0:
            return "small_loss"
        else:
            return "big_loss"

    @staticmethod
    def _generate_lessons(
        pnl_pct: float,
        mfe: Optional[float],
        mae: Optional[float],
        holding_hours: Optional[float],
        exit_reason: str,
        signal_direction: Optional[float],
        indicators: dict,
    ) -> dict:
        """Auto-generate a dict of boolean lesson flags for this trade."""
        lessons: dict[str, bool] = {}

        # --- regime_match: defer to caller context, default True ---
        lessons["regime_match"] = True  # placeholder; enriched by strategy analysis

        # --- signal_strength_adequate ---
        # We consider strength adequate if entry_signal_strength was recorded;
        # since we don't have the population median here, flag as True if > 0.5
        strength = indicators.get("entry_signal_strength")
        lessons["signal_strength_adequate"] = (
            strength is not None and float(strength) > 0.5
        ) if strength is not None else True

        # --- volume_confirmed ---
        vol_ratio = indicators.get("volume_ratio")
        lessons["volume_confirmed"] = (
            vol_ratio is not None and float(vol_ratio) > 1.0
        )

        # --- trend_aligned ---
        sma50 = indicators.get("price_vs_sma50")
        if sma50 is not None and signal_direction is not None:
            lessons["trend_aligned"] = (
                (signal_direction > 0 and float(sma50) > 0)
                or (signal_direction < 0 and float(sma50) < 0)
            )
        else:
            lessons["trend_aligned"] = True  # unknown defaults to True

        # --- rsi_extreme ---
        rsi = indicators.get("rsi_14")
        lessons["rsi_extreme"] = rsi is not None and (float(rsi) < 30 or float(rsi) > 70)

        # --- stopped_out ---
        lessons["stopped_out"] = exit_reason == "stop_loss"

        # --- held_too_long ---
        lessons["held_too_long"] = (
            holding_hours is not None
            and holding_hours > 5 * 24
            and pnl_pct < 0
        )

        # --- should_have_held (left money on the table) ---
        if mfe is not None and pnl_pct != 0:
            lessons["should_have_held"] = mfe > 2.0 * abs(pnl_pct) and pnl_pct > 0
        else:
            lessons["should_have_held"] = False

        # --- good_risk_reward ---
        if mfe is not None and mae is not None and mae != 0:
            actual_risk = abs(mae)
            actual_reward = abs(pnl_pct)
            lessons["good_risk_reward"] = actual_reward > 1.5 * actual_risk if actual_risk > 0 else False
        else:
            lessons["good_risk_reward"] = pnl_pct > 0

        return lessons

    def _lookup_pending_from_db(self, symbol: str) -> Optional[dict]:
        """Fallback: find the most recent un-exited journal entry for a symbol."""
        try:
            conn = get_connection()
            row = conn.execute(
                """SELECT id, entry_time, entry_price, entry_qty, strategy,
                          entry_signal_direction, entry_indicators
                   FROM trade_journal
                   WHERE symbol = ? AND exit_time IS NULL
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (symbol,),
            ).fetchone()
            conn.close()

            if row is None:
                return None

            indicators = {}
            if row["entry_indicators"]:
                try:
                    indicators = json.loads(row["entry_indicators"])
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "id": row["id"],
                "entry_time": row["entry_time"],
                "entry_price": float(row["entry_price"]),
                "entry_qty": float(row["entry_qty"]),
                "strategy": row["strategy"],
                "signal_direction": _safe_float(row["entry_signal_direction"]),
                "indicators": indicators,
            }
        except Exception as e:
            logger.error(f"DB lookup for pending {symbol}: {e}")
            return None

    @staticmethod
    def _calc_stats(trades: list[dict]) -> dict:
        """Calculate aggregate stats for a list of completed trade dicts."""
        if not trades:
            return {}

        pnls = [float(t.get("pnl_pct") or 0) for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        holding_hours = [
            float(t["holding_period_hours"])
            for t in trades
            if t.get("holding_period_hours") is not None
        ]

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0

        return {
            "total_trades": len(trades),
            "win_rate": len(winners) / len(pnls) if pnls else 0.0,
            "avg_win_pct": float(np.mean(winners)) if winners else 0.0,
            "avg_loss_pct": float(np.mean(losers)) if losers else 0.0,
            "profit_factor": (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            ),
            "avg_pnl_pct": float(np.mean(pnls)) if pnls else 0.0,
            "total_pnl_pct": float(np.sum(pnls)),
            "avg_holding_hours": (
                float(np.mean(holding_hours)) if holding_hours else 0.0
            ),
            "best_trade_pct": max(pnls) if pnls else 0.0,
            "worst_trade_pct": min(pnls) if pnls else 0.0,
        }
