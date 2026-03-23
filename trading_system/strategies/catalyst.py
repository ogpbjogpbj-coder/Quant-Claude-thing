"""Event-driven catalyst strategy for medium/long-term trades (1-4 weeks).

Combines earnings calendar awareness, news event scoring, and price
confirmation to take positions around significant corporate and macro
catalysts. Designed for longer holding periods than the intraday/swing
strategies, with wider stops and higher conviction thresholds.

Data sources:
- Alpaca news API (earnings reports, M&A, FDA approvals, macro events)
- Price action confirmation (post-event momentum, volume surge)
- Earnings surprise detection from news headlines
"""

import math
import re
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger
from pydantic import BaseModel

from trading_system.strategies.base import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class StrategyCatalystConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.15
    # News scanning
    news_lookback_days: int = 7
    min_catalyst_score: float = 0.4
    cache_hours: int = 2
    # Position sizing / holding
    wider_stop_multiplier: float = 4.0   # ATR multiplier for stops (wider than swing)
    target_multiplier: float = 6.0       # ATR multiplier for take profit
    min_volume_surge: float = 1.5        # Volume must be 1.5x 20d avg for confirmation
    # Earnings
    earnings_boost: float = 0.3          # Extra score for earnings-related catalysts
    # Confidence
    max_confidence: float = 0.80
    min_articles: int = 2


# ---------------------------------------------------------------------------
# Catalyst keyword categories with weights
# ---------------------------------------------------------------------------

EARNINGS_KEYWORDS: dict[str, float] = {
    "earnings beat": 0.8, "earnings miss": -0.8,
    "revenue beat": 0.7, "revenue miss": -0.7,
    "eps beat": 0.8, "eps miss": -0.8,
    "raised guidance": 0.9, "guidance raise": 0.9,
    "lowered guidance": -0.9, "guidance cut": -0.9,
    "record revenue": 0.7, "record earnings": 0.7,
    "record profit": 0.7,
    "profit warning": -0.8, "earnings warning": -0.8,
    "strong quarter": 0.6, "weak quarter": -0.6,
    "blowout quarter": 0.8,
    "top line beat": 0.6, "bottom line beat": 0.7,
    "exceeded expectations": 0.7, "missed expectations": -0.7,
    "above consensus": 0.6, "below consensus": -0.6,
}

MACRO_KEYWORDS: dict[str, float] = {
    "rate cut": 0.5, "rate hike": -0.3,
    "fed pivot": 0.5, "dovish": 0.4, "hawkish": -0.3,
    "stimulus": 0.4, "tariff": -0.4,
    "trade deal": 0.5, "trade war": -0.5,
    "sanctions": -0.3, "inflation easing": 0.4,
    "inflation rising": -0.3, "jobs report strong": 0.3,
    "jobs report weak": -0.3, "soft landing": 0.4,
    "recession risk": -0.5, "debt ceiling": -0.3,
}

CORPORATE_KEYWORDS: dict[str, float] = {
    "acquisition": 0.4, "merger": 0.4, "buyout": 0.5,
    "takeover": 0.5, "buyback": 0.4, "share repurchase": 0.4,
    "dividend increase": 0.5, "dividend cut": -0.6,
    "special dividend": 0.5,
    "stock split": 0.3, "spin-off": 0.3, "spinoff": 0.3,
    "fda approval": 0.8, "fda approved": 0.8,
    "fda rejection": -0.8, "fda reject": -0.8,
    "patent": 0.3, "lawsuit settled": 0.3,
    "class action": -0.4, "sec investigation": -0.5,
    "ceo resign": -0.4, "ceo fired": -0.5,
    "insider buying": 0.4, "insider selling": -0.3,
    "upgrade": 0.4, "downgrade": -0.4,
    "price target raised": 0.3, "price target cut": -0.3,
    "analyst upgrade": 0.4, "analyst downgrade": -0.4,
    "contract win": 0.5, "contract awarded": 0.5,
    "partnership": 0.3, "strategic alliance": 0.3,
}

ALL_CATALYSTS = {**EARNINGS_KEYWORDS, **MACRO_KEYWORDS, **CORPORATE_KEYWORDS}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class CatalystStrategy(BaseStrategy):
    """Event-driven strategy for medium-term catalyst trades."""

    name = "catalyst"

    _NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(
        self,
        config: StrategyCatalystConfig,
        api_key: str,
        secret_key: str,
    ):
        self.config = config
        self._api_key = api_key
        self._secret_key = secret_key
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        signals: list[Signal] = []

        for symbol, df in data.items():
            if df.empty or len(df) < 30:
                continue

            try:
                sig = self._evaluate(symbol, df, current_positions.get(symbol, 0))
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"Catalyst: failed for {symbol}: {e}")

        return signals

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self, symbol: str, df: pd.DataFrame, position: float
    ) -> Optional[Signal]:
        # Fetch recent news
        articles = self._fetch_news(symbol)
        if len(articles) < self.config.min_articles:
            return None

        # Score catalysts
        catalyst_score, catalyst_type, is_earnings = self._score_catalysts(articles)

        if abs(catalyst_score) < self.config.min_catalyst_score:
            return None

        # Earnings events get a conviction boost
        if is_earnings:
            catalyst_score *= (1.0 + self.config.earnings_boost)

        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(last.get("atr_14", close * 0.02))
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02

        # --- Price confirmation checks ---
        # Require volume surge (event should move volume)
        volume_ratio = float(last.get("volume_ratio", 1.0))
        if pd.isna(volume_ratio):
            volume_ratio = 1.0

        has_volume = volume_ratio >= self.config.min_volume_surge

        # Require price moving in catalyst direction (confirmation)
        returns_5d = float(last.get("returns_5d", 0))
        if pd.isna(returns_5d):
            returns_5d = 0
        price_confirms = (catalyst_score > 0 and returns_5d > 0) or \
                         (catalyst_score < 0 and returns_5d < 0)

        # Both volume and price must confirm for full signal
        if not has_volume and not price_confirms:
            return None

        # --- Build signal ---
        direction = float(np.clip(catalyst_score, -1.0, 1.0))

        # Confidence based on: catalyst strength, volume, price confirmation, article count
        conf = 0.4
        if has_volume:
            conf += 0.15
        if price_confirms:
            conf += 0.15
        conf += min(len(articles) / 30, 0.1)  # Small boost for more coverage
        conf = float(np.clip(conf, 0.3, self.config.max_confidence))

        # Wider stops for longer holding period
        if direction > 0:
            stop_loss = close - self.config.wider_stop_multiplier * atr
            take_profit = close + self.config.target_multiplier * atr
        else:
            stop_loss = close + self.config.wider_stop_multiplier * atr
            take_profit = close - self.config.target_multiplier * atr

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            strategy=self.name,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "catalyst_score": round(catalyst_score, 4),
                "catalyst_type": catalyst_type,
                "is_earnings": is_earnings,
                "volume_ratio": round(volume_ratio, 2),
                "returns_5d": round(returns_5d, 4),
                "price_confirms": price_confirms,
                "volume_confirms": has_volume,
                "n_articles": len(articles),
                "holding_period": "medium_term",
                "max_holding_days": self.config.max_holding_days,
                "volatility_21d": float(last.get("volatility_21d", 0.2)),
            },
        )

    # ------------------------------------------------------------------
    # Catalyst scoring
    # ------------------------------------------------------------------

    def _score_catalysts(
        self, articles: list[dict]
    ) -> tuple[float, str, bool]:
        """Score articles for catalyst events.

        Returns (score, dominant_catalyst_type, is_earnings_related).
        """
        now_utc = datetime.now(timezone.utc)
        total_score = 0.0
        total_weight = 0.0
        type_scores: dict[str, float] = {
            "earnings": 0.0,
            "macro": 0.0,
            "corporate": 0.0,
        }
        is_earnings = False

        for article in articles:
            headline = (article.get("headline", "") or "").lower()
            summary = (article.get("summary", "") or "").lower()
            text = f"{headline} {headline} {summary}"  # headline weighted 2x

            # Time decay — more recent events matter more
            created = article.get("created_at", "")
            hours_ago = self._hours_since(created, now_utc)
            # Longer decay for catalyst strategy (half-life ~48 hours)
            time_weight = math.exp(-0.015 * hours_ago)

            # Score against all catalyst keywords
            article_score = 0.0
            matched_type = None

            for phrase, value in ALL_CATALYSTS.items():
                if phrase in text:
                    article_score += value
                    # Track which category matched
                    if phrase in EARNINGS_KEYWORDS:
                        type_scores["earnings"] += abs(value) * time_weight
                        is_earnings = True
                    elif phrase in MACRO_KEYWORDS:
                        type_scores["macro"] += abs(value) * time_weight
                    else:
                        type_scores["corporate"] += abs(value) * time_weight

            if article_score != 0:
                total_score += article_score * time_weight
                total_weight += time_weight

        if total_weight == 0:
            return 0.0, "none", False

        avg_score = total_score / total_weight

        # Determine dominant catalyst type
        dominant = max(type_scores, key=type_scores.get)

        return avg_score, dominant, is_earnings

    # ------------------------------------------------------------------
    # News fetching (with caching)
    # ------------------------------------------------------------------

    def _fetch_news(self, symbol: str) -> list[dict]:
        """Fetch recent news for symbol from Alpaca."""
        now = time.time()
        cache_ttl = self.config.cache_hours * 3600

        if symbol in self._cache:
            cached_at, cached_articles = self._cache[symbol]
            if now - cached_at < cache_ttl:
                return cached_articles

        try:
            headers = {
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
            }
            params = {
                "symbols": symbol,
                "limit": 50,
                "sort": "desc",
            }

            resp = requests.get(
                self._NEWS_URL,
                headers=headers,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()

            articles: list[dict] = resp.json().get("news", [])
            self._cache[symbol] = (now, articles)
            return articles

        except Exception as e:
            logger.warning(f"Catalyst: news fetch failed for {symbol}: {e}")
            # Return cached data if available, even if stale
            if symbol in self._cache:
                return self._cache[symbol][1]
            return []

    @staticmethod
    def _hours_since(iso_str: str, now: datetime) -> float:
        if not iso_str:
            return 168.0  # default to 1 week ago
        try:
            created = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            delta = now - created
            return max(delta.total_seconds() / 3600, 0.0)
        except (ValueError, TypeError):
            return 168.0
