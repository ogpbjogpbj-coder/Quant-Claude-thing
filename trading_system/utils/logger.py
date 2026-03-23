"""Centralized logging with loguru and optional alerting."""

import os
import sys
from pathlib import Path

from loguru import logger


LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(level: str = "INFO") -> None:
    """Configure loguru logger with console + file sinks."""
    logger.remove()

    log_level = os.getenv("LOG_LEVEL", level).upper()

    # Console with rich formatting
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Main log file (rotated daily)
    logger.add(
        str(LOG_DIR / "trading_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # Separate file for trades only
    logger.add(
        str(LOG_DIR / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        rotation="1 day",
        retention="90 days",
        filter=lambda record: "TRADE" in record["message"] or record["extra"].get("trade"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
    )

    # Error file
    logger.add(
        str(LOG_DIR / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation="1 day",
        retention="90 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}\n{exception}",
    )

    logger.info(f"Logger initialized at level {log_level}")


def log_trade(action: str, symbol: str, qty: float, price: float, reason: str, **kwargs) -> None:
    """Log a trade execution with structured data."""
    extra = {"trade": True}
    details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"TRADE | {action} | {symbol} | qty={qty} | price={price:.2f} | reason={reason}"
    if details:
        msg += f" | {details}"
    logger.bind(**extra).info(msg)


def log_signal(strategy: str, symbol: str, signal: float, confidence: float, **kwargs) -> None:
    """Log a trading signal."""
    details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"SIGNAL | {strategy} | {symbol} | signal={signal:.4f} | confidence={confidence:.4f}"
    if details:
        msg += f" | {details}"
    logger.debug(msg)


def log_risk(event: str, **kwargs) -> None:
    """Log a risk management event."""
    details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"RISK | {event}"
    if details:
        msg += f" | {details}"
    logger.warning(msg)
