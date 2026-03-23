"""Sector rotation strategy.

Measures relative momentum across sectors, overweights the strongest
sectors and underweights (or shorts) the weakest. Uses the universe's
sector mapping to group symbols and compare sector-level performance.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from trading_system.strategies.base import BaseStrategy, Signal


# Sector mappings — extended to cover dynamic universe symbols
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology", "AMD": "Technology",
    "AVGO": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "PANW": "Technology", "QCOM": "Technology", "IDCC": "Technology",
    "INTC": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "TXN": "Technology", "MU": "Technology", "AMAT": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "SNPS": "Technology",
    "CDNS": "Technology", "MRVL": "Technology", "ON": "Technology",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "PFE": "Healthcare", "ABT": "Healthcare", "AMGN": "Healthcare",
    "BMY": "Healthcare", "GILD": "Healthcare", "ISRG": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "MDT": "Healthcare",
    # Financials
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "C": "Financials", "AXP": "Financials", "FIG": "Financials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "COST": "Consumer Staples",
    "PEP": "Consumer Staples", "KO": "Consumer Staples",
    "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "OXY": "Energy",
    "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    # Communication Services
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    # Industrials
    "BA": "Industrials", "HON": "Industrials", "CAT": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "DE": "Industrials",
    "GE": "Industrials", "LMT": "Industrials", "MMM": "Industrials",
    # Materials
    "GLD": "Materials", "SLV": "Materials", "NEM": "Materials",
    "FCX": "Materials", "APD": "Materials", "ECL": "Materials",
    "SHW": "Materials", "LIN": "Materials", "MOS": "Materials",
    "AEM": "Materials", "KGC": "Materials", "WPM": "Materials",
    "SCCO": "Materials", "COPX": "Materials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLF": "ETF", "XLK": "ETF", "XLE": "ETF", "XLV": "ETF",
    "FXI": "EM ETF", "KWEB": "EM ETF", "EEM": "EM ETF",
}


class StrategySectorRotationConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.10
    rotation_lookback: int = 21
    top_n_sectors: int = 3
    bottom_n_sectors: int = 2
    min_sector_stocks: int = 2


class SectorRotationStrategy(BaseStrategy):
    """Rotate into strongest sectors, underweight weakest."""

    name = "sector_rotation"

    def __init__(self, config: StrategySectorRotationConfig):
        self.config = config

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        signals: list[Signal] = []

        # Group symbols by sector
        sector_symbols: dict[str, list[str]] = defaultdict(list)
        for symbol in data:
            sector = SECTOR_MAP.get(symbol, "Other")
            if sector in ("ETF", "EM ETF", "Other"):
                continue
            sector_symbols[sector].append(symbol)

        # Filter sectors with enough representation
        valid_sectors = {
            s: syms for s, syms in sector_symbols.items()
            if len(syms) >= self.config.min_sector_stocks
        }

        if len(valid_sectors) < 3:
            return signals

        # Calculate sector momentum (avg return over lookback)
        sector_momentum: dict[str, float] = {}
        sector_vol: dict[str, float] = {}

        for sector, syms in valid_sectors.items():
            returns = []
            vols = []
            for sym in syms:
                df = data[sym]
                if df.empty or len(df) < self.config.rotation_lookback:
                    continue
                try:
                    close = df["close"]
                    ret = (close.iloc[-1] / close.iloc[-self.config.rotation_lookback] - 1.0)
                    vol = df.get("volatility_21d")
                    if vol is not None and not vol.empty:
                        v = float(vol.iloc[-1])
                        if not pd.isna(v):
                            vols.append(v)
                    if not pd.isna(ret):
                        returns.append(float(ret))
                except Exception:
                    continue

            if returns:
                sector_momentum[sector] = np.mean(returns)
                sector_vol[sector] = np.mean(vols) if vols else 0.2

        if len(sector_momentum) < 3:
            return signals

        # Rank sectors by momentum
        ranked = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)

        top_sectors = set(s for s, _ in ranked[:self.config.top_n_sectors])
        bottom_sectors = set(s for s, _ in ranked[-self.config.bottom_n_sectors:])

        logger.debug(
            f"Sector rotation: top={list(top_sectors)}, "
            f"bottom={list(bottom_sectors)}"
        )

        # Generate buy signals for best stocks in top sectors
        for sector in top_sectors:
            mom = sector_momentum[sector]
            if mom <= 0:
                continue  # Only buy sectors with positive momentum

            syms = valid_sectors[sector]
            # Pick best stocks in sector by individual momentum
            stock_returns = []
            for sym in syms:
                df = data[sym]
                if df.empty or len(df) < self.config.rotation_lookback:
                    continue
                try:
                    ret = float(df["close"].iloc[-1] / df["close"].iloc[-self.config.rotation_lookback] - 1.0)
                    last = df.iloc[-1]
                    stock_returns.append((sym, ret, last))
                except Exception:
                    continue

            # Sort by return, take top half
            stock_returns.sort(key=lambda x: x[1], reverse=True)
            top_stocks = stock_returns[:max(1, len(stock_returns) // 2)]

            for sym, ret, last in top_stocks:
                close = float(last["close"])
                atr = float(last.get("atr_14", close * 0.02))
                if pd.isna(atr) or atr <= 0:
                    atr = close * 0.02

                direction = min(mom * 3, 0.8)  # Scale sector momentum to direction
                confidence = 0.4 + min(mom * 2, 0.3)  # Base + momentum bonus

                signals.append(Signal(
                    symbol=sym,
                    direction=float(np.clip(direction, 0.1, 0.8)),
                    confidence=float(np.clip(confidence, 0.3, 0.7)),
                    strategy=self.name,
                    stop_loss=close - 2.5 * atr,
                    take_profit=close + 4.0 * atr,
                    metadata={
                        "sector": sector,
                        "sector_momentum": round(mom, 4),
                        "stock_return": round(ret, 4),
                        "sector_rank": "top",
                        "volatility_21d": float(last.get("volatility_21d", 0.2)),
                    },
                ))

        # Generate sell signals for worst stocks in bottom sectors
        for sector in bottom_sectors:
            mom = sector_momentum[sector]
            if mom >= 0:
                continue  # Only short sectors with negative momentum

            syms = valid_sectors[sector]
            stock_returns = []
            for sym in syms:
                df = data[sym]
                if df.empty or len(df) < self.config.rotation_lookback:
                    continue
                try:
                    ret = float(df["close"].iloc[-1] / df["close"].iloc[-self.config.rotation_lookback] - 1.0)
                    last = df.iloc[-1]
                    stock_returns.append((sym, ret, last))
                except Exception:
                    continue

            # Sort by return ascending (worst first)
            stock_returns.sort(key=lambda x: x[1])
            worst_stocks = stock_returns[:max(1, len(stock_returns) // 2)]

            for sym, ret, last in worst_stocks:
                close = float(last["close"])
                atr = float(last.get("atr_14", close * 0.02))
                if pd.isna(atr) or atr <= 0:
                    atr = close * 0.02

                direction = max(mom * 3, -0.8)
                confidence = 0.4 + min(abs(mom) * 2, 0.3)

                signals.append(Signal(
                    symbol=sym,
                    direction=float(np.clip(direction, -0.8, -0.1)),
                    confidence=float(np.clip(confidence, 0.3, 0.7)),
                    strategy=self.name,
                    stop_loss=close + 2.5 * atr,
                    take_profit=close - 4.0 * atr,
                    metadata={
                        "sector": sector,
                        "sector_momentum": round(mom, 4),
                        "stock_return": round(ret, 4),
                        "sector_rank": "bottom",
                        "volatility_21d": float(last.get("volatility_21d", 0.2)),
                    },
                ))

        return signals
