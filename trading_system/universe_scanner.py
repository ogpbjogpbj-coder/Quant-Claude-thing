"""Dynamic universe scanner — pulls tradeable stocks from Alpaca and filters to ~500.

Includes large-caps, mid-caps, small-caps, and penny stocks with volume filters
to avoid illiquid junk. Refreshes daily.
"""

from datetime import datetime, timedelta
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest
from loguru import logger

from trading_system.config import TradingConfig


class UniverseScanner:
    """Dynamically builds a ~500-stock trading universe from Alpaca's full asset list."""

    # Core holdings that are always included
    CORE_SYMBOLS = [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "AMD", "JPM", "V",
    ]

    def __init__(self, config: TradingConfig):
        self.config = config
        base_url = config.alpaca.base_url
        self.trading_client = TradingClient(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            paper="paper" in base_url,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
        )
        self._cached_universe: list[str] = []
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=12)

        # Filters
        self.target_size = 500
        self.min_avg_volume = 200_000       # shares/day
        self.penny_min_volume = 1_000_000   # higher bar for penny stocks
        self.min_price = 0.10               # absolute floor
        self.penny_threshold = 5.0          # price < $5 = penny stock

    def get_universe(self, force_refresh: bool = False) -> list[str]:
        """Return the current universe, refreshing if stale."""
        now = datetime.utcnow()
        if (
            not force_refresh
            and self._cached_universe
            and self._cache_time
            and (now - self._cache_time) < self._cache_ttl
        ):
            return self._cached_universe

        try:
            universe = self._scan()
            self._cached_universe = universe
            self._cache_time = now
            logger.info(f"Universe refreshed: {len(universe)} symbols")
            return universe
        except Exception as e:
            logger.error(f"Universe scan failed: {e}")
            if self._cached_universe:
                logger.info("Using cached universe as fallback")
                return self._cached_universe
            logger.info("Falling back to config universe")
            return self.config.universe

    def _scan(self) -> list[str]:
        """Pull all tradeable assets, fetch volume/price data, filter to ~500."""
        # Step 1: Get all tradeable US equities from Alpaca
        req = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        all_assets = self.trading_client.get_all_assets(req)
        tradable = [
            a for a in all_assets
            if a.tradable and a.symbol.isalpha() and len(a.symbol) <= 5
        ]
        logger.info(f"Alpaca has {len(tradable)} tradeable US equities")

        # Step 2: Fetch recent bars in batches to get volume + price
        symbols = [a.symbol for a in tradable]
        scored = self._score_symbols(symbols)

        # Step 3: Always include core symbols
        selected = set(self.CORE_SYMBOLS)

        # Step 4: Separate into tiers and pick from each
        large_caps = []   # price >= $50
        mid_caps = []     # $10 <= price < $50
        small_caps = []   # $5 <= price < $10
        penny = []        # price < $5

        for sym, info in scored.items():
            price = info["price"]
            vol = info["avg_volume"]

            if price < self.min_price:
                continue

            if price >= 50 and vol >= self.min_avg_volume:
                large_caps.append((sym, info["score"]))
            elif 10 <= price < 50 and vol >= self.min_avg_volume:
                mid_caps.append((sym, info["score"]))
            elif self.penny_threshold <= price < 10 and vol >= self.min_avg_volume:
                small_caps.append((sym, info["score"]))
            elif price < self.penny_threshold and vol >= self.penny_min_volume:
                penny.append((sym, info["score"]))

        # Sort each tier by score (volume * volatility — we want active movers)
        large_caps.sort(key=lambda x: x[1], reverse=True)
        mid_caps.sort(key=lambda x: x[1], reverse=True)
        small_caps.sort(key=lambda x: x[1], reverse=True)
        penny.sort(key=lambda x: x[1], reverse=True)

        # Allocate slots: 200 large, 150 mid, 80 small, 70 penny
        for sym, _ in large_caps[:200]:
            selected.add(sym)
        for sym, _ in mid_caps[:150]:
            selected.add(sym)
        for sym, _ in small_caps[:80]:
            selected.add(sym)
        for sym, _ in penny[:70]:
            selected.add(sym)

        result = sorted(selected)

        logger.info(
            f"Universe breakdown: {len(large_caps)} large-cap candidates, "
            f"{len(mid_caps)} mid-cap, {len(small_caps)} small-cap, "
            f"{len(penny)} penny stocks (vol>{self.penny_min_volume:,})"
        )
        logger.info(
            f"Selected {len(result)} symbols "
            f"(large={min(len(large_caps), 200)}, mid={min(len(mid_caps), 150)}, "
            f"small={min(len(small_caps), 80)}, penny={min(len(penny), 70)})"
        )

        return result

    def _score_symbols(self, symbols: list[str]) -> dict:
        """Fetch 5-day bars in batches, compute avg volume and price for scoring."""
        scored = {}
        batch_size = 1000  # Alpaca handles large symbol lists
        start = datetime.utcnow() - timedelta(days=7)

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start,
                )
                bars = self.data_client.get_stock_bars(request)
                bars_df = bars.df

                if bars_df.empty:
                    continue

                for sym in batch:
                    try:
                        if sym not in bars_df.index.get_level_values(0):
                            continue
                        sym_df = bars_df.loc[sym]
                        if len(sym_df) < 2:
                            continue

                        avg_vol = sym_df["volume"].mean()
                        last_price = sym_df["close"].iloc[-1]
                        # Volatility = std of daily returns
                        returns = sym_df["close"].pct_change().dropna()
                        volatility = returns.std() if len(returns) > 1 else 0

                        # Score = dollar volume * volatility (we want liquid movers)
                        dollar_vol = avg_vol * last_price
                        score = dollar_vol * max(volatility, 0.001)

                        scored[sym] = {
                            "price": last_price,
                            "avg_volume": avg_vol,
                            "volatility": volatility,
                            "dollar_volume": dollar_vol,
                            "score": score,
                        }
                    except Exception:
                        continue

                logger.info(
                    f"Scored batch {i // batch_size + 1}: "
                    f"{len(scored)} symbols so far"
                )

            except Exception as e:
                logger.error(f"Failed to fetch batch {i // batch_size + 1}: {e}")
                continue

        return scored
