"""Base strategy interface and signal dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    symbol: str
    direction: float        # -1.0 (strong sell) to +1.0 (strong buy), 0 = no signal
    confidence: float       # 0.0 to 1.0
    strategy: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.direction > 0

    @property
    def is_sell(self) -> bool:
        return self.direction < 0

    @property
    def strength(self) -> float:
        return abs(self.direction) * self.confidence


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        """Generate trading signals from enriched market data.

        Args:
            data: Symbol -> DataFrame with OHLCV + indicators
            current_positions: Symbol -> current position qty (+ long, - short)

        Returns:
            List of Signal objects
        """
        ...
