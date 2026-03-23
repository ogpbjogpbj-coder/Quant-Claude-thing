"""NLP sentiment analysis strategy.

Fetches news articles from Alpaca's free news API, scores sentiment using
a keyword-based approach with weighted word lists, and generates trading
signals based on aggregate sentiment per symbol. Designed as a low-weight
overlay strategy since keyword-based NLP has limited accuracy.
"""

import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from loguru import logger
from pydantic import BaseModel

from trading_system.strategies.base import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class StrategySentimentConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.10
    min_articles: int = 3
    sentiment_threshold: float = 0.3
    cache_hours: int = 4


# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

POSITIVE_WORDS: set[str] = {
    "beat", "beats", "exceeds", "exceeded", "growth", "profit", "profits",
    "upgrade", "upgraded", "bullish", "surge", "surges", "surged", "rally",
    "rallies", "rallied", "strong", "record", "outperform", "outperforms",
    "boost", "boosted", "gain", "gains", "gained", "positive", "raises",
    "raised", "upside", "optimistic", "innovation", "breakthrough",
    "expansion", "revenue", "dividend", "approval", "approved", "soar",
    "soared", "momentum", "recovery", "recover", "recovered", "upbeat",
    "jumps", "jumped", "rises", "risen", "climbs", "climbed", "tops",
    "topped", "higher", "accelerate", "accelerated",
}

NEGATIVE_WORDS: set[str] = {
    "miss", "misses", "missed", "decline", "declines", "declined", "loss",
    "losses", "downgrade", "downgraded", "bearish", "crash", "crashed",
    "weak", "warning", "risk", "risks", "lawsuit", "fraud", "recall",
    "recalled", "cut", "cuts", "layoff", "layoffs", "debt", "default",
    "defaults", "negative", "concern", "concerns", "slowdown", "investigation",
    "investigated", "plunge", "plunged", "tumble", "tumbled", "drops",
    "dropped", "falls", "fallen", "lower", "penalty", "fine", "fined",
    "bankruptcy", "inflation", "recession", "sell-off", "selloff",
    "disappoints", "disappointed", "slump", "slumped",
}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class SentimentStrategy(BaseStrategy):
    """Keyword-based news sentiment strategy using Alpaca news API."""

    name = "sentiment"

    # Alpaca news endpoint
    _NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(
        self,
        config: StrategySentimentConfig,
        api_key: str,
        secret_key: str,
    ):
        self.config = config
        self._api_key = api_key
        self._secret_key = secret_key

        # Cache: symbol -> (timestamp, list[article_dict])
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
            if df.empty:
                continue

            try:
                articles = self._fetch_news(symbol)
            except Exception as e:
                logger.warning(f"Sentiment: news fetch failed for {symbol}: {e}")
                continue

            if len(articles) < self.config.min_articles:
                logger.debug(
                    f"Sentiment: only {len(articles)} articles for {symbol}, "
                    f"need {self.config.min_articles}"
                )
                continue

            try:
                sentiment, agreement = self._aggregate_sentiment(articles)
            except Exception as e:
                logger.debug(f"Sentiment: scoring failed for {symbol}: {e}")
                continue

            if abs(sentiment) < self.config.sentiment_threshold:
                continue

            # Build signal ------------------------------------------------
            last = df.iloc[-1]
            close = last["close"]
            atr = last.get("atr_14", 0)
            if pd.isna(atr):
                atr = 0

            # Direction proportional to sentiment, clamped to [-1, 1]
            direction = float(np.clip(sentiment, -1.0, 1.0))

            # Confidence: base range 0.3-0.6 scaled by article count & agreement
            n_articles = len(articles)
            article_factor = min(n_articles / 20, 1.0)  # diminishing returns >20
            confidence = 0.3 + 0.3 * article_factor * agreement
            confidence = float(np.clip(confidence, 0.3, 0.6))

            stop_loss = None
            take_profit = None
            if atr > 0:
                if direction > 0:
                    stop_loss = close - 2.5 * atr
                    take_profit = close + 3.0 * atr
                else:
                    stop_loss = close + 2.5 * atr
                    take_profit = close - 3.0 * atr

            signals.append(
                Signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    strategy=self.name,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "sentiment_score": round(sentiment, 4),
                        "agreement": round(agreement, 4),
                        "n_articles": n_articles,
                    },
                )
            )

        return signals

    # ------------------------------------------------------------------
    # News fetching
    # ------------------------------------------------------------------

    def _fetch_news(self, symbol: str) -> list[dict]:
        """Fetch news articles for *symbol* from Alpaca, with caching."""
        now = time.time()
        cache_ttl = self.config.cache_hours * 3600

        if symbol in self._cache:
            cached_at, cached_articles = self._cache[symbol]
            if now - cached_at < cache_ttl:
                return cached_articles

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }
        params = {
            "symbols": symbol,
            "limit": 50,
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

    # ------------------------------------------------------------------
    # Sentiment scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_text(text: str) -> tuple[int, int, int]:
        """Return (positive_count, negative_count, total_words) for *text*."""
        words = text.lower().split()
        total = len(words)
        pos = sum(1 for w in words if w.strip(".,;:!?\"'()-") in POSITIVE_WORDS)
        neg = sum(1 for w in words if w.strip(".,;:!?\"'()-") in NEGATIVE_WORDS)
        return pos, neg, total

    def _score_article(self, article: dict) -> float:
        """Score a single article (headline weighted 2x, summary 1x)."""
        headline = article.get("headline", "")
        summary = article.get("summary", "")

        h_pos, h_neg, h_total = self._score_text(headline)
        s_pos, s_neg, s_total = self._score_text(summary)

        # Headline counts double
        pos = 2 * h_pos + s_pos
        neg = 2 * h_neg + s_neg
        total = 2 * h_total + s_total

        if total == 0:
            return 0.0

        return (pos - neg) / total

    def _aggregate_sentiment(
        self, articles: list[dict]
    ) -> tuple[float, float]:
        """Compute time-weighted aggregate sentiment and agreement score.

        Returns
        -------
        sentiment : float
            Weighted average sentiment in roughly [-1, 1].
        agreement : float
            Fraction of articles that agree on direction (0-1).
        """
        now_utc = datetime.now(timezone.utc)
        scores: list[float] = []
        weights: list[float] = []

        for article in articles:
            score = self._score_article(article)

            # Exponential decay by hours since publication
            created = article.get("created_at", "")
            hours_ago = self._hours_since(created, now_utc)
            weight = math.exp(-0.05 * hours_ago)  # half-life ~14 hours

            scores.append(score)
            weights.append(weight)

        if not weights or sum(weights) == 0:
            return 0.0, 0.0

        total_weight = sum(weights)
        sentiment = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Agreement: fraction of non-zero articles sharing majority direction
        nonzero = [(s > 0) for s in scores if s != 0]
        if nonzero:
            majority = sum(nonzero) / len(nonzero)
            agreement = max(majority, 1 - majority)  # 0.5 – 1.0
        else:
            agreement = 0.5

        return sentiment, agreement

    @staticmethod
    def _hours_since(iso_str: str, now: datetime) -> float:
        """Parse ISO-8601 timestamp and return hours elapsed."""
        if not iso_str:
            return 48.0  # default to 2 days ago if missing

        try:
            created = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            delta = now - created
            return max(delta.total_seconds() / 3600, 0.0)
        except (ValueError, TypeError):
            return 48.0
