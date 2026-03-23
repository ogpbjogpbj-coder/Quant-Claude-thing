"""SQLite database for trade history and performance tracking."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            order_id TEXT,
            strategy TEXT,
            signal_strength REAL,
            stop_loss REAL,
            take_profit REAL,
            status TEXT DEFAULT 'filled',
            pnl REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            daily_pnl REAL,
            total_pnl REAL,
            num_positions INTEGER,
            drawdown_pct REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal REAL NOT NULL,
            confidence REAL NOT NULL,
            acted_on INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS risk_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT,
            action_taken TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def record_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    order_id: str = "",
    strategy: str = "",
    signal_strength: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    pnl: Optional[float] = None,
    notes: str = "",
) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO trades
           (timestamp, symbol, side, qty, price, order_id, strategy,
            signal_strength, stop_loss, take_profit, pnl, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            symbol, side, qty, price, order_id, strategy,
            signal_strength, stop_loss, take_profit, pnl, notes,
        ),
    )
    conn.commit()
    conn.close()


def record_snapshot(
    equity: float,
    cash: float,
    positions_value: float,
    daily_pnl: float = 0.0,
    total_pnl: float = 0.0,
    num_positions: int = 0,
    drawdown_pct: float = 0.0,
) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO portfolio_snapshots
           (timestamp, equity, cash, positions_value, daily_pnl,
            total_pnl, num_positions, drawdown_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            equity, cash, positions_value, daily_pnl,
            total_pnl, num_positions, drawdown_pct,
        ),
    )
    conn.commit()
    conn.close()


def record_risk_event(event_type: str, details: str, action_taken: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO risk_events (timestamp, event_type, details, action_taken) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), event_type, details, action_taken),
    )
    conn.commit()
    conn.close()


def get_recent_trades(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_daily_pnl(date: Optional[str] = None) -> float:
    """Get total realized PnL for a given date."""
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    conn = get_connection()
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE timestamp LIKE ? AND pnl IS NOT NULL",
        (f"{date}%",),
    ).fetchone()
    conn.close()
    return float(row["total"])


def get_peak_equity() -> float:
    """Get historical peak equity from snapshots."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COALESCE(MAX(equity), 0) as peak FROM portfolio_snapshots"
    ).fetchone()
    conn.close()
    return float(row["peak"])
