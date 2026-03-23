"""Adaptive AI-driven strategy that learns from trade history and evolves rules.

Analyzes completed trades from the journal to discover indicator patterns that
separate winners from losers, builds multi-condition rules, and adapts them
over time. Works alongside the existing pre-made strategies.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import log
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from trading_system.strategies.base import BaseStrategy, Signal
from trading_system.utils.db import DB_PATH, get_connection


class StrategyAdaptiveConfig(BaseModel):
    """Configuration for the adaptive strategy."""

    enabled: bool = True
    weight: float = 0.15
    min_trades_to_learn: int = 20
    learning_lookback: int = 200
    min_rule_confidence: float = 0.55
    max_rules: int = 30
    evolution_interval_hours: int = 6


@dataclass
class TradingRule:
    """A single evolved trading rule."""

    rule_name: str
    rule_type: str  # "buy_when", "sell_when", "avoid_when"
    conditions: list[dict]  # each: {"indicator": str, "operator": str, "threshold": float}
    win_rate: float
    sample_size: int
    avg_return: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Indicators that can appear in entry_indicators JSON and enriched DataFrames
LEARNABLE_INDICATORS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "adx",
    "atr_pct",
    "volume_ratio",
    "volatility_10d",
    "volatility_21d",
    "returns_1d",
    "returns_5d",
    "returns_10d",
    "returns_21d",
    "price_vs_sma50",
    "price_vs_sma200",
    "stoch_k",
    "stoch_d",
    "price_zscore",
]


class AdaptiveStrategy(BaseStrategy):
    """Strategy that evolves its own trading rules from historical trade outcomes."""

    name = "adaptive"

    def __init__(
        self,
        config: StrategyAdaptiveConfig,
        db_path: Optional[str] = None,
    ):
        self.config = config
        self.db_path = db_path or str(DB_PATH)
        self.rules: list[TradingRule] = []
        self.last_evolution_time: Optional[datetime] = None
        self._ensure_evolved_rules_table()
        self.rules = self._load_rules_from_db()
        if self.rules:
            logger.info(f"Adaptive strategy loaded {len(self.rules)} existing rules")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        """Generate signals by evaluating evolved rules against current indicators."""
        if not self.config.enabled:
            return []

        trade_count = self._count_completed_trades()
        if trade_count < self.config.min_trades_to_learn:
            return []

        # Evolve if interval has elapsed
        now = datetime.now(timezone.utc)
        if self.last_evolution_time is None or (
            now - self.last_evolution_time
            > timedelta(hours=self.config.evolution_interval_hours)
        ):
            self.evolve()

        if not self.rules:
            return []

        buy_rules = [r for r in self.rules if r.rule_type == "buy_when"]
        sell_rules = [r for r in self.rules if r.rule_type == "sell_when"]
        avoid_rules = [r for r in self.rules if r.rule_type == "avoid_when"]

        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < 2:
                continue

            try:
                indicators = self._latest_indicators(df)
            except Exception as e:
                logger.debug(f"Adaptive: failed to read indicators for {symbol}: {e}")
                continue

            # Check avoid rules first
            vetoed = any(self._evaluate_rule(r, indicators) for r in avoid_rules)

            # Evaluate buy rules
            firing_buy = [r for r in buy_rules if self._evaluate_rule(r, indicators)]
            if firing_buy and not vetoed:
                signal = self._build_signal(symbol, firing_buy, indicators, direction_sign=1.0)
                if signal is not None:
                    signals.append(signal)

            # Evaluate sell rules
            firing_sell = [r for r in sell_rules if self._evaluate_rule(r, indicators)]
            if firing_sell and not vetoed:
                signal = self._build_signal(symbol, firing_sell, indicators, direction_sign=-1.0)
                if signal is not None:
                    signals.append(signal)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    # ------------------------------------------------------------------
    # Evolution engine
    # ------------------------------------------------------------------

    def evolve(self) -> None:
        """Core learning method: mine trade history and evolve rules."""
        logger.info("Adaptive strategy: beginning rule evolution ...")
        self.last_evolution_time = datetime.now(timezone.utc)

        trades = self._load_completed_trades()
        if len(trades) < self.config.min_trades_to_learn:
            logger.info(
                f"Adaptive: only {len(trades)} trades, need {self.config.min_trades_to_learn}"
            )
            return

        # Parse indicator snapshots
        parsed = self._parse_trade_indicators(trades)
        if not parsed:
            logger.warning("Adaptive: could not parse any trade indicators")
            return

        winners = [t for t in parsed if t["pnl_pct"] > 0]
        losers = [t for t in parsed if t["pnl_pct"] <= 0]

        if len(winners) < 5 or len(losers) < 5:
            logger.info("Adaptive: not enough winners/losers to learn from")
            return

        # Step 1: Single-indicator rules
        candidate_rules = self._discover_single_rules(parsed, winners, losers)

        # Step 2: Multi-condition rules (combine top singles)
        candidate_rules.extend(self._discover_combo_rules(candidate_rules, parsed))

        # Step 3: Regime-conditional rules
        candidate_rules.extend(self._discover_regime_rules(parsed, winners, losers))

        # Step 4: Lesson-based rules
        candidate_rules.extend(self._learn_from_lessons(trades, parsed))

        # De-duplicate by rule_name, keep highest score
        best_by_name: dict[str, TradingRule] = {}
        for rule in candidate_rules:
            score = self._score_rule(rule)
            existing = best_by_name.get(rule.rule_name)
            if existing is None or score > self._score_rule(existing):
                best_by_name[rule.rule_name] = rule

        # Rank and keep top N
        ranked = sorted(best_by_name.values(), key=self._score_rule, reverse=True)
        self.rules = ranked[: self.config.max_rules]

        # Save to DB
        self._save_rules_to_db(self.rules)

        if self.rules:
            top = self.rules[0]
            logger.info(
                f"Evolved {len(self.rules)} rules, top rule: {top.rule_name} "
                f"(win_rate={top.win_rate * 100:.1f}%, n={top.sample_size})"
            )
        else:
            logger.info("Evolved 0 rules — no patterns met confidence threshold")

    # ------------------------------------------------------------------
    # Rule discovery helpers
    # ------------------------------------------------------------------

    def _discover_single_rules(
        self,
        parsed: list[dict],
        winners: list[dict],
        losers: list[dict],
    ) -> list[TradingRule]:
        """Find single-indicator thresholds separating winners from losers."""
        rules: list[TradingRule] = []
        min_conf = self.config.min_rule_confidence
        min_n = 10

        for indicator in LEARNABLE_INDICATORS:
            values = [t["indicators"].get(indicator) for t in parsed]
            values_clean = [v for v in values if v is not None and np.isfinite(v)]
            if len(values_clean) < min_n:
                continue

            arr = np.array(values_clean)
            percentiles = np.percentile(arr, [25, 50, 75])

            for pct_label, threshold in zip(["p25", "p50", "p75"], percentiles):
                for op, op_label in [(">", "above"), ("<", "below")]:
                    above = op == ">"
                    matching = [
                        t
                        for t in parsed
                        if t["indicators"].get(indicator) is not None
                        and (
                            (t["indicators"][indicator] > threshold)
                            if above
                            else (t["indicators"][indicator] < threshold)
                        )
                    ]

                    if len(matching) < min_n:
                        continue

                    wins = [t for t in matching if t["pnl_pct"] > 0]
                    wr = len(wins) / len(matching)
                    avg_ret = float(np.mean([t["pnl_pct"] for t in matching]))

                    if wr >= min_conf:
                        rule_type = "buy_when" if avg_ret > 0 else "sell_when"
                        rule_name = f"{indicator}_{op_label}_{pct_label}"
                        rules.append(
                            TradingRule(
                                rule_name=rule_name,
                                rule_type=rule_type,
                                conditions=[
                                    {
                                        "indicator": indicator,
                                        "operator": op,
                                        "threshold": float(threshold),
                                    }
                                ],
                                win_rate=wr,
                                sample_size=len(matching),
                                avg_return=avg_ret,
                            )
                        )

                    # Also check the complement for "avoid_when"
                    non_matching = [t for t in parsed if t not in matching]
                    if len(non_matching) >= min_n:
                        non_wins = [t for t in non_matching if t["pnl_pct"] > 0]
                        non_wr = len(non_wins) / len(non_matching)
                        if non_wr < (1 - min_conf):
                            # This region is bad — create avoid rule
                            avoid_op = "<" if above else ">"
                            rules.append(
                                TradingRule(
                                    rule_name=f"avoid_{indicator}_{op_label}_{pct_label}",
                                    rule_type="avoid_when",
                                    conditions=[
                                        {
                                            "indicator": indicator,
                                            "operator": avoid_op,
                                            "threshold": float(threshold),
                                        }
                                    ],
                                    win_rate=1 - non_wr,
                                    sample_size=len(non_matching),
                                    avg_return=float(
                                        np.mean([t["pnl_pct"] for t in non_matching])
                                    ),
                                )
                            )

        return rules

    def _discover_combo_rules(
        self,
        single_rules: list[TradingRule],
        parsed: list[dict],
    ) -> list[TradingRule]:
        """Combine top single-indicator rules into 2-condition AND rules."""
        rules: list[TradingRule] = []
        min_conf = self.config.min_rule_confidence
        min_n = 10

        # Take top 10 single rules by score
        top_singles = sorted(single_rules, key=self._score_rule, reverse=True)[:10]

        for i, r1 in enumerate(top_singles):
            for r2 in top_singles[i + 1 :]:
                # Skip if same indicator
                ind1 = r1.conditions[0]["indicator"]
                ind2 = r2.conditions[0]["indicator"]
                if ind1 == ind2:
                    continue
                # Skip if different rule types (don't combine buy + sell)
                if r1.rule_type != r2.rule_type:
                    continue

                combined_conditions = r1.conditions + r2.conditions
                matching = [
                    t
                    for t in parsed
                    if self._evaluate_rule_on_trade(combined_conditions, t["indicators"])
                ]
                if len(matching) < min_n:
                    continue

                wins = [t for t in matching if t["pnl_pct"] > 0]
                wr = len(wins) / len(matching)
                avg_ret = float(np.mean([t["pnl_pct"] for t in matching]))

                if wr >= min_conf:
                    rule_name = f"combo_{ind1}_AND_{ind2}"
                    rules.append(
                        TradingRule(
                            rule_name=rule_name,
                            rule_type=r1.rule_type,
                            conditions=combined_conditions,
                            win_rate=wr,
                            sample_size=len(matching),
                            avg_return=avg_ret,
                        )
                    )

        return rules

    def _discover_regime_rules(
        self,
        parsed: list[dict],
        winners: list[dict],
        losers: list[dict],
    ) -> list[TradingRule]:
        """Find rules that work specifically within certain market regimes."""
        rules: list[TradingRule] = []
        min_conf = self.config.min_rule_confidence
        min_n = 10

        regimes = set(t.get("regime") for t in parsed if t.get("regime"))
        if not regimes:
            return rules

        for regime in regimes:
            regime_trades = [t for t in parsed if t.get("regime") == regime]
            if len(regime_trades) < min_n:
                continue

            regime_winners = [t for t in regime_trades if t["pnl_pct"] > 0]
            regime_losers = [t for t in regime_trades if t["pnl_pct"] <= 0]
            if len(regime_winners) < 3 or len(regime_losers) < 3:
                continue

            for indicator in LEARNABLE_INDICATORS:
                values = [
                    t["indicators"].get(indicator)
                    for t in regime_trades
                    if t["indicators"].get(indicator) is not None
                ]
                if len(values) < min_n:
                    continue

                arr = np.array([v for v in values if np.isfinite(v)])
                if len(arr) < min_n:
                    continue

                median = float(np.median(arr))
                for op, label in [(">", "above"), ("<", "below")]:
                    above = op == ">"
                    matching = [
                        t
                        for t in regime_trades
                        if t["indicators"].get(indicator) is not None
                        and (
                            (t["indicators"][indicator] > median)
                            if above
                            else (t["indicators"][indicator] < median)
                        )
                    ]
                    if len(matching) < min_n:
                        continue

                    wins = [t for t in matching if t["pnl_pct"] > 0]
                    wr = len(wins) / len(matching)
                    avg_ret = float(np.mean([t["pnl_pct"] for t in matching]))

                    if wr >= min_conf:
                        rule_type = "buy_when" if avg_ret > 0 else "sell_when"
                        rule_name = f"regime_{regime}_{indicator}_{label}_median"
                        conditions = [
                            {
                                "indicator": indicator,
                                "operator": op,
                                "threshold": median,
                            },
                            {
                                "indicator": "_regime",
                                "operator": "==",
                                "threshold": regime,
                            },
                        ]
                        rules.append(
                            TradingRule(
                                rule_name=rule_name,
                                rule_type=rule_type,
                                conditions=conditions,
                                win_rate=wr,
                                sample_size=len(matching),
                                avg_return=avg_ret,
                            )
                        )

        return rules

    def _learn_from_lessons(
        self, trades: list[dict], parsed: list[dict]
    ) -> list[TradingRule]:
        """Extract rules from the lessons JSON column."""
        rules: list[TradingRule] = []
        min_n = 10

        # Collect lesson flags
        volume_confirmed_wins = 0
        volume_confirmed_losses = 0
        no_volume_wins = 0
        no_volume_losses = 0
        stopped_out_count = 0
        held_too_long_count = 0
        total = 0

        for trade in trades:
            lessons_raw = trade.get("lessons")
            if not lessons_raw:
                continue

            try:
                lessons = json.loads(lessons_raw) if isinstance(lessons_raw, str) else lessons_raw
            except (json.JSONDecodeError, TypeError):
                continue

            is_winner = (trade.get("pnl_pct") or 0) > 0
            total += 1

            if lessons.get("volume_confirmed"):
                if is_winner:
                    volume_confirmed_wins += 1
                else:
                    volume_confirmed_losses += 1
            else:
                if is_winner:
                    no_volume_wins += 1
                else:
                    no_volume_losses += 1

            if lessons.get("stopped_out"):
                stopped_out_count += 1
            if lessons.get("held_too_long"):
                held_too_long_count += 1

        # Volume confirmation rule
        vol_confirmed_total = volume_confirmed_wins + volume_confirmed_losses
        no_vol_total = no_volume_wins + no_volume_losses
        if vol_confirmed_total >= min_n and no_vol_total >= min_n:
            vol_wr = volume_confirmed_wins / vol_confirmed_total
            no_vol_wr = no_volume_wins / no_vol_total
            if vol_wr > no_vol_wr and vol_wr >= self.config.min_rule_confidence:
                rules.append(
                    TradingRule(
                        rule_name="lesson_require_volume",
                        rule_type="avoid_when",
                        conditions=[
                            {
                                "indicator": "volume_ratio",
                                "operator": "<",
                                "threshold": 1.0,
                            }
                        ],
                        win_rate=vol_wr,
                        sample_size=vol_confirmed_total,
                        avg_return=0.0,
                    )
                )

        # Stopped out → avoid high ATR entries
        if total >= min_n and stopped_out_count / total > 0.3:
            # Find median atr_pct among stopped-out trades
            stopped_atrs = []
            for t in parsed:
                atr = t["indicators"].get("atr_pct")
                if atr is not None:
                    stopped_atrs.append(atr)
            if stopped_atrs:
                high_atr = float(np.percentile(stopped_atrs, 75))
                rules.append(
                    TradingRule(
                        rule_name="lesson_avoid_high_atr",
                        rule_type="avoid_when",
                        conditions=[
                            {
                                "indicator": "atr_pct",
                                "operator": ">",
                                "threshold": high_atr,
                            }
                        ],
                        win_rate=0.0,
                        sample_size=stopped_out_count,
                        avg_return=0.0,
                    )
                )

        # Held too long → earlier exit signals (sell when momentum fading)
        if total >= min_n and held_too_long_count / total > 0.25:
            rules.append(
                TradingRule(
                    rule_name="lesson_exit_early_macd",
                    rule_type="sell_when",
                    conditions=[
                        {
                            "indicator": "macd_hist",
                            "operator": "<",
                            "threshold": 0.0,
                        },
                        {
                            "indicator": "rsi_14",
                            "operator": ">",
                            "threshold": 60.0,
                        },
                    ],
                    win_rate=0.6,
                    sample_size=held_too_long_count,
                    avg_return=0.0,
                )
            )

        return rules

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule(self, rule: TradingRule, indicators: dict) -> bool:
        """Check whether all conditions in a rule are met by current indicators."""
        return self._evaluate_rule_on_trade(rule.conditions, indicators)

    @staticmethod
    def _evaluate_rule_on_trade(conditions: list[dict], indicators: dict) -> bool:
        """Evaluate a list of conditions against an indicator dict (AND logic)."""
        for cond in conditions:
            ind_name = cond["indicator"]
            op = cond["operator"]
            threshold = cond["threshold"]

            value = indicators.get(ind_name)
            if value is None:
                return False

            # Regime equality check (special case)
            if op == "==":
                if value != threshold:
                    return False
                continue

            try:
                value = float(value)
            except (ValueError, TypeError):
                return False

            if op == ">":
                if not (value > threshold):
                    return False
            elif op == "<":
                if not (value < threshold):
                    return False
            elif op == ">=":
                if not (value >= threshold):
                    return False
            elif op == "<=":
                if not (value <= threshold):
                    return False
            elif op == "between":
                # threshold should be [low, high]
                if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                    if not (threshold[0] <= value <= threshold[1]):
                        return False
                else:
                    return False
            else:
                return False

        return True

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        firing_rules: list[TradingRule],
        indicators: dict,
        direction_sign: float,
    ) -> Optional[Signal]:
        """Build a Signal from a set of firing rules."""
        avg_wr = float(np.mean([r.win_rate for r in firing_rules]))
        if avg_wr < self.config.min_rule_confidence:
            return None

        avg_return = float(np.mean([r.avg_return for r in firing_rules]))
        direction = float(np.clip(avg_return * direction_sign * 10, -1.0, 1.0))
        # Ensure direction has the correct sign
        if direction_sign > 0:
            direction = max(direction, 0.1)
        else:
            direction = min(direction, -0.1)

        close = indicators.get("close", 0.0)
        atr = indicators.get("atr_14", 0.0)

        stop_loss = (close - 2 * atr) if close and atr else None
        take_profit = (close + 3 * atr) if close and atr else None
        if direction_sign < 0 and stop_loss is not None and take_profit is not None:
            stop_loss, take_profit = (close + 2 * atr), (close - 3 * atr)

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=avg_wr,
            strategy=self.name,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "rules_fired": [
                    {
                        "name": r.rule_name,
                        "win_rate": round(r.win_rate, 4),
                        "sample_size": r.sample_size,
                    }
                    for r in firing_rules
                ],
                "avg_win_rate": round(avg_wr, 4),
                "num_rules": len(firing_rules),
            },
        )

    @staticmethod
    def _latest_indicators(df: pd.DataFrame) -> dict:
        """Extract the latest row of a DataFrame as a plain dict."""
        last = df.iloc[-1]
        return {col: last[col] for col in df.columns if pd.notna(last[col])}

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_evolved_rules_table(self) -> None:
        """Create the evolved_rules table if it does not exist."""
        try:
            conn = self._get_connection()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evolved_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT UNIQUE NOT NULL,
                    rule_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    performance_score REAL DEFAULT 0,
                    times_triggered INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    active INTEGER DEFAULT 1
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Adaptive: could not ensure evolved_rules table: {e}")

    def _load_rules_from_db(self) -> list[TradingRule]:
        """Load active rules from the evolved_rules table."""
        rules: list[TradingRule] = []
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM evolved_rules WHERE active = 1"
            ).fetchall()
            conn.close()

            for row in rows:
                try:
                    conditions = json.loads(row["conditions"])
                    rules.append(
                        TradingRule(
                            rule_name=row["rule_name"],
                            rule_type=row["rule_type"],
                            conditions=conditions,
                            win_rate=(
                                row["times_correct"] / row["times_triggered"]
                                if row["times_triggered"] > 0
                                else 0.0
                            ),
                            sample_size=row["times_triggered"],
                            avg_return=row["performance_score"],
                            last_updated=datetime.fromisoformat(
                                row["last_updated"]
                            ) if row["last_updated"] else datetime.now(timezone.utc),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug(f"Adaptive: skipping malformed rule: {e}")
        except Exception as e:
            logger.debug(f"Adaptive: could not load rules from DB: {e}")

        return rules

    def _save_rules_to_db(self, rules: list[TradingRule]) -> None:
        """INSERT OR REPLACE rules into evolved_rules table."""
        try:
            conn = self._get_connection()
            now = datetime.now(timezone.utc).isoformat()

            # Mark all existing rules inactive
            conn.execute("UPDATE evolved_rules SET active = 0")

            for rule in rules:
                score = self._score_rule(rule)
                active = 1 if rule.win_rate >= self.config.min_rule_confidence else 0
                times_correct = int(rule.win_rate * rule.sample_size)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO evolved_rules
                        (rule_name, rule_type, conditions, performance_score,
                         times_triggered, times_correct, last_updated, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule.rule_name,
                        rule.rule_type,
                        json.dumps(rule.conditions),
                        score,
                        rule.sample_size,
                        times_correct,
                        now,
                        active,
                    ),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Adaptive: failed to save rules to DB: {e}")

    def _count_completed_trades(self) -> int:
        """Return the number of completed trades in trade_journal."""
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trade_journal "
                "WHERE exit_time IS NOT NULL AND pnl_pct IS NOT NULL"
            ).fetchone()
            conn.close()
            return row["cnt"] if row else 0
        except Exception:
            return 0

    def _load_completed_trades(self) -> list[dict]:
        """Load recent completed trades from trade_journal."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM trade_journal "
                "WHERE exit_time IS NOT NULL AND pnl_pct IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT ?",
                (self.config.learning_lookback,),
            ).fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.debug(f"Adaptive: could not load trades: {e}")
            return []

    def _parse_trade_indicators(self, trades: list[dict]) -> list[dict]:
        """Parse entry_indicators JSON and attach to each trade record."""
        parsed: list[dict] = []
        for trade in trades:
            raw = trade.get("entry_indicators")
            if not raw:
                continue
            try:
                indicators = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue

            parsed.append(
                {
                    "pnl_pct": trade.get("pnl_pct", 0),
                    "regime": trade.get("entry_regime"),
                    "indicators": indicators,
                }
            )
        return parsed

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_rule(rule: TradingRule) -> float:
        """Score a rule: win_rate * log(sample_size) * avg_return_if_win."""
        if rule.sample_size <= 0:
            return 0.0
        avg_ret = max(abs(rule.avg_return), 0.001)
        return rule.win_rate * log(max(rule.sample_size, 1)) * avg_ret
