"""Evolution engine that creates, backtests, and promotes new candidate strategies.

Mines the trade journal for indicator patterns that separate winners from losers,
builds composite multi-condition rules, validates them with walk-forward testing
and price-data backtesting, and promotes successful rules to the evolved_rules table.

Designed to run periodically (e.g. daily) as a standalone process, separate from
the real-time AdaptiveStrategy.
"""

import json
import sqlite3
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from trading_system.utils.db import DB_PATH, get_connection


# Indicators expected in entry_indicators JSON and enriched DataFrames
KNOWN_INDICATORS = [
    "rsi_14", "macd", "macd_signal", "bb_position", "adx", "atr_pct",
    "volume_ratio", "volatility_21d", "returns_5d", "returns_21d",
    "price_vs_sma50", "price_vs_sma200", "stoch_k", "price_zscore",
]

WINNER_LABELS = {"big_win", "small_win"}
LOSER_LABELS = {"big_loss", "small_loss"}


class StrategyEvolver:
    """Mines trade history, discovers rules, validates them, and promotes the best."""

    def __init__(self, min_trades: int = 30, min_backtest_trades: int = 15) -> None:
        self.min_trades = min_trades
        self.min_backtest_trades = min_backtest_trades
        self.last_evolution_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve_strategies(self, enriched_data: dict[str, pd.DataFrame]) -> dict:
        """Main entry point. Run the full evolution pipeline and return a report.

        Parameters
        ----------
        enriched_data : dict mapping symbol -> DataFrame with indicator columns.

        Returns
        -------
        dict  Summary report of the evolution run.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        empty_report = self._empty_report(timestamp)

        # Load completed trades from the journal
        trades = self._load_trades()
        if len(trades) < self.min_trades:
            logger.info(
                "StrategyEvolver: only {} trades available (need {}), skipping",
                len(trades), self.min_trades,
            )
            return empty_report

        logger.info("StrategyEvolver: analysing {} completed trades", len(trades))

        # Step 1 – mine single-indicator rules
        single_rules = self._mine_single_indicator_rules(trades)
        logger.info("Step 1: found {} single-indicator candidate rules", len(single_rules))

        # Step 2 – build composite strategies
        combos = self._build_combos(single_rules, trades)
        logger.info("Step 2: built {} composite rule combos", len(combos))

        # Step 3 – walk-forward validation
        validated: list[dict] = []
        for rule in combos:
            if self._walk_forward_validate(rule, trades):
                validated.append(rule)
        logger.info("Step 3: {} rules passed walk-forward validation", len(validated))

        # Step 4 – backtest on price data
        backtested: list[dict] = []
        for rule in validated:
            bt = self._backtest_rule(rule, enriched_data)
            if bt.get("passed"):
                rule["backtest"] = bt
                backtested.append(rule)
        logger.info("Step 4: {} rules passed price backtest", len(backtested))

        # Step 5 – promote / demote
        promoted, demoted = self._promote_demote(backtested, trades)

        # Step 6 – build report
        active_rules = self.get_active_rules_summary()
        insights = self._generate_insights(single_rules, trades)

        top_rules = []
        for r in active_rules[:10]:
            top_rules.append({
                "name": r["rule_name"],
                "win_rate": r["win_rate"],
                "profit_factor": r.get("profit_factor", 0.0),
                "sample_size": r["times_triggered"],
            })

        report = {
            "timestamp": timestamp,
            "trades_analyzed": len(trades),
            "candidates_generated": len(single_rules) + len(combos),
            "candidates_validated": len(validated),
            "rules_promoted": promoted,
            "rules_demoted": demoted,
            "active_rules": len(active_rules),
            "top_rules": top_rules,
            "insights": insights,
        }

        self.last_evolution_time = datetime.now(timezone.utc)
        logger.info(
            "StrategyEvolver complete: promoted={}, demoted={}, active={}",
            promoted, demoted, len(active_rules),
        )
        return report

    def get_active_rules_summary(self) -> list[dict]:
        """Return all active evolved rules with performance stats."""
        try:
            conn = get_connection()
            rows = conn.execute(
                "SELECT * FROM evolved_rules WHERE active = 1 ORDER BY performance_score DESC"
            ).fetchall()
            conn.close()
            results = []
            for row in rows:
                d = dict(row)
                triggered = d.get("times_triggered", 0)
                correct = d.get("times_correct", 0)
                d["win_rate"] = correct / triggered if triggered > 0 else 0.0
                results.append(d)
            return results
        except Exception as exc:
            logger.error("Failed to load active rules: {}", exc)
            return []

    def record_rule_outcome(self, rule_name: str, was_correct: bool) -> None:
        """Update trigger and correctness counters for a rule after a trade."""
        try:
            conn = get_connection()
            if was_correct:
                conn.execute(
                    "UPDATE evolved_rules SET times_triggered = times_triggered + 1, "
                    "times_correct = times_correct + 1, last_updated = ? WHERE rule_name = ?",
                    (datetime.now(timezone.utc).isoformat(), rule_name),
                )
            else:
                conn.execute(
                    "UPDATE evolved_rules SET times_triggered = times_triggered + 1, "
                    "last_updated = ? WHERE rule_name = ?",
                    (datetime.now(timezone.utc).isoformat(), rule_name),
                )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("Failed to record rule outcome for {}: {}", rule_name, exc)

    # ------------------------------------------------------------------
    # Internal: Step 1 – mine single-indicator rules
    # ------------------------------------------------------------------

    def _mine_single_indicator_rules(self, trades: list[dict]) -> list[dict]:
        """Find single-indicator rules that separate winners from losers."""
        rules: list[dict] = []

        for indicator in KNOWN_INDICATORS:
            winner_vals: list[float] = []
            loser_vals: list[float] = []

            for t in trades:
                indicators = t.get("entry_indicators")
                if indicators is None:
                    continue
                if isinstance(indicators, str):
                    try:
                        indicators = json.loads(indicators)
                    except (json.JSONDecodeError, TypeError):
                        continue
                val = indicators.get(indicator)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue

                label = t.get("outcome_label", "")
                if label in WINNER_LABELS:
                    winner_vals.append(val)
                elif label in LOSER_LABELS:
                    loser_vals.append(val)

            if len(winner_vals) < 5 or len(loser_vals) < 5:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(winner_vals, loser_vals)
            if p_value >= 0.05:
                continue

            # Find optimal threshold via accuracy maximization
            all_vals = winner_vals + loser_vals
            all_labels = [1] * len(winner_vals) + [0] * len(loser_vals)
            arr = np.array(all_vals)
            lab = np.array(all_labels)

            best_acc = 0.0
            best_thresh = float(np.median(arr))
            best_op = "<"

            percentiles = np.percentile(arr, np.arange(10, 95, 5))
            for thresh in percentiles:
                for op in ["<", ">"]:
                    if op == "<":
                        preds = arr < thresh
                    else:
                        preds = arr > thresh
                    acc = np.mean(preds == lab)
                    if acc > best_acc:
                        best_acc = acc
                        best_thresh = float(thresh)
                        best_op = op

            if best_acc < 0.52:
                continue

            # Compute win rate and avg PnL for this rule
            wr, avg_pnl, pf, n = self._score_single_rule(
                trades, indicator, best_op, best_thresh,
            )

            rule = {
                "indicator": indicator,
                "operator": best_op,
                "threshold": round(best_thresh, 6),
                "accuracy": round(best_acc, 4),
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "win_rate": round(wr, 4),
                "avg_pnl": round(avg_pnl, 6),
                "profit_factor": round(pf, 4),
                "sample_size": n,
            }
            rules.append(rule)

        # Sort by accuracy descending
        rules.sort(key=lambda r: r["accuracy"], reverse=True)
        return rules

    # ------------------------------------------------------------------
    # Internal: Step 2 – build combos
    # ------------------------------------------------------------------

    def _build_combos(self, single_rules: list[dict], trades: list[dict]) -> list[dict]:
        """Try AND combinations of the top single rules."""
        combos: list[dict] = []
        top_rules = single_rules[:15]  # limit to avoid combinatorial explosion

        for r1, r2 in combinations(top_rules, 2):
            # Skip if same indicator
            if r1["indicator"] == r2["indicator"]:
                continue

            conditions = [
                {"indicator": r1["indicator"], "operator": r1["operator"], "threshold": r1["threshold"]},
                {"indicator": r2["indicator"], "operator": r2["operator"], "threshold": r2["threshold"]},
            ]

            wr, avg_pnl, pf, n = self._score_combo(trades, conditions)

            if n < 10:
                continue
            if wr <= 0.55:
                continue
            if pf <= 1.2:
                continue

            name = self._make_rule_name(conditions)
            combos.append({
                "rule_name": name,
                "rule_type": "buy_when",
                "conditions": conditions,
                "win_rate": round(wr, 4),
                "avg_pnl": round(avg_pnl, 6),
                "profit_factor": round(pf, 4),
                "sample_size": n,
            })

        # Sort by profit factor
        combos.sort(key=lambda c: c["profit_factor"], reverse=True)
        return combos

    # ------------------------------------------------------------------
    # Internal: Step 3 – walk-forward validation
    # ------------------------------------------------------------------

    def _walk_forward_validate(self, rule: dict, trades: list[dict]) -> bool:
        """Train/test split validation. Returns True if the rule passes."""
        sorted_trades = sorted(trades, key=lambda t: t.get("entry_time", ""))
        split = int(len(sorted_trades) * 0.7)
        train = sorted_trades[:split]
        test = sorted_trades[split:]

        if len(train) < 10 or len(test) < 5:
            return False

        conditions = rule["conditions"]

        # Train performance
        train_wr, train_pnl, train_pf, train_n = self._score_combo(train, conditions)
        if train_n < 5 or train_wr < 0.50:
            return False

        # Test performance
        test_wr, test_pnl, test_pf, test_n = self._score_combo(test, conditions)
        if test_n < 3 or test_wr < 0.50:
            return False

        # Out-of-sample Sharpe approximation
        test_returns = self._get_matching_returns(test, conditions)
        if len(test_returns) < 3:
            return False

        avg_holding = self._median_holding_days(trades)
        if avg_holding <= 0:
            avg_holding = 1.0

        mean_ret = np.mean(test_returns)
        std_ret = np.std(test_returns, ddof=1)
        if std_ret == 0:
            return False

        oos_sharpe = (mean_ret / std_ret) * np.sqrt(252.0 / avg_holding)

        rule["oos_sharpe"] = round(float(oos_sharpe), 4)
        rule["oos_win_rate"] = round(test_wr, 4)
        return True

    # ------------------------------------------------------------------
    # Internal: Step 4 – backtest on price data
    # ------------------------------------------------------------------

    def _backtest_rule(self, rule: dict, enriched_data: dict[str, pd.DataFrame]) -> dict:
        """Simple price-based backtest of a rule across enriched data."""
        conditions = rule["conditions"]
        holding_days = max(1, int(self._median_holding_days_from_db()))

        all_returns: list[float] = []
        wins = 0
        total = 0

        for symbol, df in enriched_data.items():
            if df is None or df.empty:
                continue

            # Check required indicators are present
            required = {c["indicator"] for c in conditions}
            if not required.issubset(set(df.columns)):
                continue

            df = df.copy().reset_index(drop=True)

            for i in range(len(df) - holding_days):
                row = df.iloc[i]
                if self._row_matches_conditions(row, conditions):
                    entry_price = row.get("close", row.get("Close", None))
                    exit_row = df.iloc[i + holding_days]
                    exit_price = exit_row.get("close", exit_row.get("Close", None))

                    if entry_price is None or exit_price is None:
                        continue
                    if entry_price == 0:
                        continue

                    ret = (exit_price - entry_price) / entry_price
                    all_returns.append(float(ret))
                    total += 1
                    if ret > 0:
                        wins += 1

        result: dict = {"passed": False}

        if total < self.min_backtest_trades:
            return result

        returns_arr = np.array(all_returns)
        mean_ret = float(np.mean(returns_arr))
        std_ret = float(np.std(returns_arr, ddof=1)) if total > 1 else 0.0
        win_rate = wins / total

        if std_ret > 0 and holding_days > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252.0 / holding_days)
        else:
            sharpe = 0.0

        result.update({
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "mean_return": round(mean_ret, 6),
            "sharpe": round(sharpe, 4),
        })

        if sharpe > 0.5 and win_rate > 0.50:
            result["passed"] = True

        return result

    # ------------------------------------------------------------------
    # Internal: Step 5 – promote / demote
    # ------------------------------------------------------------------

    def _promote_demote(self, backtested_rules: list[dict], trades: list[dict]) -> tuple[int, int]:
        """Save passing rules, deactivate failing ones. Returns (promoted, demoted)."""
        promoted = 0
        demoted = 0

        try:
            conn = get_connection()

            # Promote rules that passed all tests
            for rule in backtested_rules:
                bt = rule.get("backtest", {})
                perf_score = (
                    rule.get("win_rate", 0) * 0.4
                    + rule.get("profit_factor", 0) * 0.3
                    + rule.get("oos_sharpe", 0) * 0.2
                    + bt.get("sharpe", 0) * 0.1
                )
                conditions_json = json.dumps(rule["conditions"])
                now = datetime.now(timezone.utc).isoformat()

                existing = conn.execute(
                    "SELECT id FROM evolved_rules WHERE rule_name = ?",
                    (rule["rule_name"],),
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE evolved_rules SET conditions = ?, performance_score = ?, "
                        "active = 1, last_updated = ? WHERE rule_name = ?",
                        (conditions_json, round(perf_score, 4), now, rule["rule_name"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO evolved_rules "
                        "(rule_name, rule_type, conditions, performance_score, "
                        "times_triggered, times_correct, created_at, last_updated, active) "
                        "VALUES (?, ?, ?, ?, 0, 0, ?, ?, 1)",
                        (
                            rule["rule_name"],
                            rule.get("rule_type", "buy_when"),
                            conditions_json,
                            round(perf_score, 4),
                            now,
                            now,
                        ),
                    )
                promoted += 1
                logger.info(
                    "Promoted rule: {} (score={:.3f}, wr={:.1%})",
                    rule["rule_name"], perf_score, rule.get("win_rate", 0),
                )

            # Demote decayed rules: triggered > 50 and win_rate < 45%
            decayed = conn.execute(
                "SELECT id, rule_name, times_triggered, times_correct "
                "FROM evolved_rules WHERE active = 1 AND times_triggered > 50"
            ).fetchall()

            for row in decayed:
                d = dict(row)
                triggered = d["times_triggered"]
                correct = d["times_correct"]
                wr = correct / triggered if triggered > 0 else 0
                if wr < 0.45:
                    conn.execute(
                        "UPDATE evolved_rules SET active = 0, last_updated = ? WHERE id = ?",
                        (datetime.now(timezone.utc).isoformat(), d["id"]),
                    )
                    demoted += 1
                    logger.info(
                        "Demoted decayed rule: {} (wr={:.1%}, triggers={})",
                        d["rule_name"], wr, triggered,
                    )

            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("Error during promote/demote: {}", exc)

        return promoted, demoted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_trades(self) -> list[dict]:
        """Load all completed trades from the trade journal."""
        try:
            conn = get_connection()
            rows = conn.execute(
                "SELECT * FROM trade_journal WHERE exit_time IS NOT NULL ORDER BY entry_time"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("Failed to load trade journal: {}", exc)
            return []

    def _get_indicator_value(self, trade: dict, indicator: str) -> Optional[float]:
        """Extract an indicator value from a trade's entry_indicators JSON."""
        raw = trade.get("entry_indicators")
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        val = raw.get(indicator)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a single condition."""
        if operator == "<":
            return value < threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == ">=":
            return value >= threshold
        return False

    def _trade_matches_conditions(self, trade: dict, conditions: list[dict]) -> bool:
        """Check if a trade's entry indicators match ALL conditions."""
        for cond in conditions:
            val = self._get_indicator_value(trade, cond["indicator"])
            if val is None:
                return False
            if not self._evaluate_condition(val, cond["operator"], cond["threshold"]):
                return False
        return True

    def _row_matches_conditions(self, row: pd.Series, conditions: list[dict]) -> bool:
        """Check if a DataFrame row matches ALL conditions."""
        for cond in conditions:
            val = row.get(cond["indicator"])
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return False
            if not self._evaluate_condition(float(val), cond["operator"], cond["threshold"]):
                return False
        return True

    def _score_single_rule(
        self, trades: list[dict], indicator: str, operator: str, threshold: float,
    ) -> tuple[float, float, float, int]:
        """Score a single rule on trades. Returns (win_rate, avg_pnl, profit_factor, n)."""
        conditions = [{"indicator": indicator, "operator": operator, "threshold": threshold}]
        return self._score_combo(trades, conditions)

    def _score_combo(
        self, trades: list[dict], conditions: list[dict],
    ) -> tuple[float, float, float, int]:
        """Score a rule combo. Returns (win_rate, avg_pnl, profit_factor, n)."""
        wins = 0
        total = 0
        gross_profit = 0.0
        gross_loss = 0.0
        pnl_sum = 0.0

        for t in trades:
            if not self._trade_matches_conditions(t, conditions):
                continue
            total += 1
            pnl = t.get("pnl_pct", 0.0) or 0.0
            try:
                pnl = float(pnl)
            except (ValueError, TypeError):
                pnl = 0.0

            pnl_sum += pnl
            label = t.get("outcome_label", "")
            if label in WINNER_LABELS:
                wins += 1
                gross_profit += abs(pnl)
            elif label in LOSER_LABELS:
                gross_loss += abs(pnl)

        if total == 0:
            return 0.0, 0.0, 0.0, 0

        win_rate = wins / total
        avg_pnl = pnl_sum / total
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            10.0 if gross_profit > 0 else 0.0
        )

        return win_rate, avg_pnl, profit_factor, total

    def _get_matching_returns(self, trades: list[dict], conditions: list[dict]) -> list[float]:
        """Get PnL returns for trades matching conditions."""
        returns: list[float] = []
        for t in trades:
            if not self._trade_matches_conditions(t, conditions):
                continue
            pnl = t.get("pnl_pct", 0.0) or 0.0
            try:
                returns.append(float(pnl))
            except (ValueError, TypeError):
                pass
        return returns

    def _median_holding_days(self, trades: list[dict]) -> float:
        """Compute median holding period in days from trade list."""
        periods: list[float] = []
        for t in trades:
            hp = t.get("holding_period_hours")
            if hp is not None:
                try:
                    periods.append(float(hp) / 24.0)
                except (ValueError, TypeError):
                    pass
        if not periods:
            return 1.0
        return float(np.median(periods))

    def _median_holding_days_from_db(self) -> float:
        """Get median holding period from the full trade journal."""
        try:
            conn = get_connection()
            rows = conn.execute(
                "SELECT holding_period_hours FROM trade_journal "
                "WHERE holding_period_hours IS NOT NULL"
            ).fetchall()
            conn.close()
            if not rows:
                return 1.0
            hours = [float(dict(r)["holding_period_hours"]) for r in rows]
            return float(np.median(hours)) / 24.0
        except Exception:
            return 1.0

    def _make_rule_name(self, conditions: list[dict]) -> str:
        """Generate a human-readable rule name from conditions."""
        parts: list[str] = []
        for c in conditions:
            parts.append(f"{c['indicator']}{c['operator']}{c['threshold']}")
        return "combo_" + "_AND_".join(parts)

    def _generate_insights(self, single_rules: list[dict], trades: list[dict]) -> list[str]:
        """Generate human-readable insight strings."""
        insights: list[str] = []

        for rule in single_rules[:5]:
            ind = rule["indicator"]
            op = rule["operator"]
            thresh = rule["threshold"]
            wr = rule["win_rate"]
            n = rule["sample_size"]
            insights.append(
                f"{ind} {op} {thresh} entries have {wr:.0%} win rate (n={n})"
            )

        total = len(trades)
        winners = sum(1 for t in trades if t.get("outcome_label") in WINNER_LABELS)
        if total > 0:
            overall_wr = winners / total
            insights.append(f"Overall baseline win rate: {overall_wr:.0%} ({total} trades)")

        return insights

    def _empty_report(self, timestamp: str) -> dict:
        """Return a skeleton report when there is insufficient data."""
        return {
            "timestamp": timestamp,
            "trades_analyzed": 0,
            "candidates_generated": 0,
            "candidates_validated": 0,
            "rules_promoted": 0,
            "rules_demoted": 0,
            "active_rules": 0,
            "top_rules": [],
            "insights": [],
        }
