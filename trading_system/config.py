"""Configuration management for the trading system."""

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


class AlpacaConfig(BaseSettings):
    api_key: str = Field(default="", alias="ALPACA_API_KEY")
    secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    base_url: str = Field(
        default="https://api.alpaca.markets", alias="ALPACA_BASE_URL"
    )

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)


class RiskConfig(BaseModel):
    max_portfolio_risk_pct: float = 4.0
    max_position_size_pct: float = 10.0
    max_single_trade_risk_pct: float = 2.0
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 8.0
    max_drawdown_pct: float = 15.0
    max_open_positions: int = 12
    max_sector_exposure_pct: float = 25.0
    max_correlation_threshold: float = 0.70
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0
    min_sharpe_ratio: float = 0.5
    position_sizing: str = "kelly"
    kelly_fraction: float = 0.5
    max_holding_days: int = 30  # Default max hold: auto-exit after this many days


class ExecutionConfig(BaseModel):
    order_type: str = "limit"
    limit_offset_pct: float = 0.01
    max_slippage_pct: float = 0.1
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    time_in_force: str = "day"
    enable_fractional: bool = True


class ScheduleConfig(BaseModel):
    market_open_offset_minutes: int = 15
    market_close_offset_minutes: int = 15
    rebalance_interval_minutes: int = 30
    data_refresh_interval_minutes: int = 5


class DataConfig(BaseModel):
    lookback_days: int = 120
    bar_timeframe: str = "1Day"
    intraday_timeframe: str = "15Min"
    cache_ttl_minutes: int = 5


class StrategyMomentumConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.25
    lookback_periods: list[int] = [5, 10, 21, 63]


class StrategyMeanReversionConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.20
    z_score_entry: float = 2.0
    z_score_exit: float = 0.5
    lookback: int = 20


class StrategyMLEnsembleConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.30
    retrain_interval_hours: int = 24
    features: list[str] = [
        "rsi_14", "macd_signal", "bb_position", "volume_ratio",
        "atr_pct", "returns_5d", "returns_21d", "volatility_21d",
        "price_vs_sma50", "price_vs_sma200",
    ]


class StrategyVolBreakoutConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.15
    atr_multiplier: float = 1.5
    lookback: int = 20


class StrategyTrendConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.10
    fast_ma: int = 10
    slow_ma: int = 50
    signal_ma: int = 200


class StrategyPairsConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.15
    lookback: int = 60
    z_score_entry: float = 2.0
    z_score_exit: float = 0.5
    min_half_life: int = 5
    max_half_life: int = 60
    max_pairs: int = 10


class StrategySentimentConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.10
    min_articles: int = 3
    sentiment_threshold: float = 0.3
    cache_hours: int = 4


class StrategyAdaptiveConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.15
    min_trades_to_learn: int = 20
    learning_lookback: int = 200
    min_rule_confidence: float = 0.55
    max_rules: int = 30
    evolution_interval_hours: int = 6


class StrategyCatalystConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.15
    news_lookback_days: int = 7
    min_catalyst_score: float = 0.4
    cache_hours: int = 2
    wider_stop_multiplier: float = 4.0
    target_multiplier: float = 6.0
    min_volume_surge: float = 1.5
    earnings_boost: float = 0.3
    max_confidence: float = 0.80
    min_articles: int = 2
    max_holding_days: int = 21  # Auto-exit catalyst trades after 3 weeks


class StrategiesConfig(BaseModel):
    momentum: StrategyMomentumConfig = StrategyMomentumConfig()
    mean_reversion: StrategyMeanReversionConfig = StrategyMeanReversionConfig()
    ml_ensemble: StrategyMLEnsembleConfig = StrategyMLEnsembleConfig()
    volatility_breakout: StrategyVolBreakoutConfig = StrategyVolBreakoutConfig()
    trend_following: StrategyTrendConfig = StrategyTrendConfig()
    pairs_trading: StrategyPairsConfig = StrategyPairsConfig()
    sentiment: StrategySentimentConfig = StrategySentimentConfig()
    adaptive: StrategyAdaptiveConfig = StrategyAdaptiveConfig()
    catalyst: StrategyCatalystConfig = StrategyCatalystConfig()


class UniverseConfig(BaseModel):
    dynamic: bool = True
    target_size: int = 500
    min_avg_volume: int = 200_000
    penny_min_volume: int = 1_000_000
    min_price: float = 0.10
    penny_threshold: float = 5.0
    refresh_hours: int = 12


class TradingConfig(BaseModel):
    alpaca: AlpacaConfig = AlpacaConfig()
    universe: list[str] = [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "AMD", "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "BAC",
        "XOM", "CVX", "AVGO", "LLY", "COST", "ABBV", "MRK", "PEP", "TMO",
        "CRM", "ADBE", "NFLX",
    ]
    universe_config: UniverseConfig = UniverseConfig()
    strategies: StrategiesConfig = StrategiesConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    data: DataConfig = DataConfig()
    mode: str = "live"

    model_config = {"arbitrary_types_allowed": True}


def load_config(config_path: Optional[str] = None) -> TradingConfig:
    """Load configuration from YAML file, with env var overrides."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "trading_config.yaml")

    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if raw:
            cfg = raw

    alpaca = AlpacaConfig()

    universe_env = os.getenv("TRADING_UNIVERSE")
    universe = (
        [s.strip() for s in universe_env.split(",")]
        if universe_env
        else cfg.get("universe", {}).get("symbols", TradingConfig().universe)
    )

    risk_cfg = cfg.get("risk", {})
    risk_cfg["max_portfolio_risk_pct"] = float(
        os.getenv("MAX_PORTFOLIO_RISK_PCT", risk_cfg.get("max_portfolio_risk_pct", 2.0))
    )
    risk_cfg["max_position_size_pct"] = float(
        os.getenv("MAX_POSITION_SIZE_PCT", risk_cfg.get("max_position_size_pct", 5.0))
    )
    risk_cfg["max_daily_loss_pct"] = float(
        os.getenv("MAX_DAILY_LOSS_PCT", risk_cfg.get("max_daily_loss_pct", 3.0))
    )
    risk_cfg["max_open_positions"] = int(
        os.getenv("MAX_OPEN_POSITIONS", risk_cfg.get("max_open_positions", 20))
    )

    return TradingConfig(
        alpaca=alpaca,
        universe=universe,
        strategies=StrategiesConfig(**cfg.get("strategies", {})),
        risk=RiskConfig(**risk_cfg),
        execution=ExecutionConfig(**cfg.get("execution", {})),
        schedule=ScheduleConfig(**cfg.get("schedule", {})),
        data=DataConfig(**cfg.get("data", {})),
        mode=cfg.get("system", {}).get("mode", "live"),
    )
