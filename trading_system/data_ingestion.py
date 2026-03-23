"""Market data ingestion from Alpaca API with caching and indicator computation."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger

from trading_system.config import TradingConfig


class DataIngestion:
    """Fetches and caches market data from Alpaca, computes technical indicators."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.client = StockHistoricalDataClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = timedelta(minutes=config.data.cache_ttl_minutes)

    def get_bars(
        self,
        symbols: list[str],
        lookback_days: Optional[int] = None,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for multiple symbols."""
        if lookback_days is None:
            lookback_days = self.config.data.lookback_days

        tf_map = {
            "1Day": TimeFrame.Day,
            "1Hour": TimeFrame.Hour,
            "15Min": TimeFrame(15, "Min"),
            "5Min": TimeFrame(5, "Min"),
            "1Min": TimeFrame.Minute,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        # Check cache
        now = datetime.utcnow()
        uncached = []
        result = {}
        for sym in symbols:
            cache_key = f"{sym}_{timeframe}_{lookback_days}"
            if cache_key in self._cache:
                cached_time, cached_df = self._cache[cache_key]
                if now - cached_time < self._cache_ttl:
                    result[sym] = cached_df
                    continue
            uncached.append(sym)

        if uncached:
            start = datetime.utcnow() - timedelta(days=lookback_days)
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=uncached,
                    timeframe=tf,
                    start=start,
                )
                bars = self.client.get_stock_bars(request)
                bars_df = bars.df

                if bars_df.empty:
                    logger.warning(f"No data returned for {uncached}")
                    return result

                # bars_df has multi-index (symbol, timestamp)
                for sym in uncached:
                    try:
                        if sym in bars_df.index.get_level_values(0):
                            sym_df = bars_df.loc[sym].copy()
                            sym_df.index = pd.to_datetime(sym_df.index)
                            sym_df = sym_df.sort_index()
                            cache_key = f"{sym}_{timeframe}_{lookback_days}"
                            self._cache[cache_key] = (now, sym_df)
                            result[sym] = sym_df
                        else:
                            logger.warning(f"No data for {sym}")
                    except Exception as e:
                        logger.warning(f"Error processing {sym}: {e}")

            except Exception as e:
                logger.error(f"Failed to fetch bars: {e}")

        return result

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Get latest bid/ask midpoint prices."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.client.get_stock_latest_quote(request)
            prices = {}
            for sym, quote in quotes.items():
                mid = (quote.ask_price + quote.bid_price) / 2
                if mid > 0:
                    prices[sym] = mid
                else:
                    prices[sym] = quote.ask_price or quote.bid_price
            return prices
        except Exception as e:
            logger.error(f"Failed to get latest quotes: {e}")
            return {}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a rich set of technical indicators on OHLCV data."""
        if df.empty or len(df) < 20:
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Returns
        df["returns_1d"] = close.pct_change(1)
        df["returns_5d"] = close.pct_change(5)
        df["returns_10d"] = close.pct_change(10)
        df["returns_21d"] = close.pct_change(21)

        # Volatility
        df["volatility_10d"] = df["returns_1d"].rolling(10).std() * np.sqrt(252)
        df["volatility_21d"] = df["returns_1d"].rolling(21).std() * np.sqrt(252)

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = close.rolling(period).mean()
            if period <= 50:
                df[f"ema_{period}"] = close.ewm(span=period).mean()

        # Price relative to MAs
        if "sma_50" in df.columns:
            df["price_vs_sma50"] = (close / df["sma_50"]) - 1
        if "sma_200" in df.columns:
            df["price_vs_sma200"] = (close / df["sma_200"]) - 1

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_mid"] = sma20
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr_14"] / close

        # Volume indicators
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        denom = (high14 - low14).replace(0, np.nan)
        df["stoch_k"] = 100 * (close - low14) / denom
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ADX (simplified)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr14 = df["atr_14"]
        plus_di = 100 * (plus_dm.ewm(span=14).mean() / atr14.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(span=14).mean() / atr14.replace(0, np.nan))
        dx = (100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)))
        df["adx"] = dx.ewm(span=14).mean()

        # On Balance Volume
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_sma_20"] = obv.rolling(20).mean()

        # Z-score of price (for mean reversion)
        df["price_zscore"] = (close - sma20) / std20.replace(0, np.nan)

        return df

    def get_enriched_data(
        self,
        symbols: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars and compute all indicators for the universe."""
        if symbols is None:
            symbols = self.config.universe

        bars = self.get_bars(symbols, lookback_days)
        enriched = {}
        for sym, df in bars.items():
            try:
                enriched[sym] = self.compute_indicators(df.copy())
            except Exception as e:
                logger.warning(f"Failed to compute indicators for {sym}: {e}")
                enriched[sym] = df

        logger.info(f"Enriched data ready for {len(enriched)} symbols")
        return enriched
