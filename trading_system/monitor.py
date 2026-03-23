"""Real-time monitoring and dashboard for the trading system."""

from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from loguru import logger

from trading_system.execution import ExecutionEngine
from trading_system.portfolio import PortfolioManager
from trading_system.risk_manager import RiskManager
from trading_system.utils.db import get_recent_trades


class Monitor:
    """Terminal-based monitoring dashboard."""

    def __init__(
        self,
        execution: ExecutionEngine,
        portfolio: PortfolioManager,
        risk_manager: RiskManager,
    ):
        self.execution = execution
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.console = Console()

    def print_status(self) -> None:
        """Print a comprehensive status report to the terminal."""
        self.console.clear()
        self.console.print(self._build_status_panel())

    def _build_status_panel(self) -> Panel:
        layout = Layout()

        # Account info
        account = self.execution.get_account()
        account_table = Table(title="Account", show_header=False)
        account_table.add_column("Metric", style="cyan")
        account_table.add_column("Value", style="green")

        equity = account.get("equity", 0)
        cash = account.get("cash", 0)
        buying_power = account.get("buying_power", 0)

        account_table.add_row("Equity", f"${equity:,.2f}")
        account_table.add_row("Cash", f"${cash:,.2f}")
        account_table.add_row("Buying Power", f"${buying_power:,.2f}")
        account_table.add_row("PDT Count", str(account.get("day_trade_count", 0)))

        # Risk status
        risk_table = Table(title="Risk Status", show_header=False)
        risk_table.add_column("Check", style="cyan")
        risk_table.add_column("Status", style="green")

        peak = self.risk_manager._peak_equity
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0

        risk_table.add_row(
            "Circuit Breaker",
            "[red]ACTIVE[/red]" if self.risk_manager.is_halted else "[green]OK[/green]",
        )
        risk_table.add_row("Peak Equity", f"${peak:,.2f}")
        risk_table.add_row(
            "Drawdown",
            f"[{'red' if drawdown > 5 else 'yellow' if drawdown > 2 else 'green'}]"
            f"{drawdown:.2f}%[/]",
        )
        risk_table.add_row("Max Drawdown Limit", f"{self.risk_manager.config.max_drawdown_pct}%")

        # Positions
        positions = self.execution.get_positions()
        pos_table = Table(title=f"Positions ({len(positions)})")
        pos_table.add_column("Symbol", style="cyan")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P&L", justify="right")
        pos_table.add_column("P&L %", justify="right")

        total_unrealized = 0
        for sym, pos in sorted(positions.items()):
            pnl = pos["unrealized_pl"]
            pnl_pct = pos["unrealized_plpc"] * 100
            total_unrealized += pnl

            color = "green" if pnl >= 0 else "red"
            pos_table.add_row(
                sym,
                f"{pos['qty']:.2f}",
                f"${pos['avg_entry_price']:.2f}",
                f"${pos['current_price']:.2f}",
                f"[{color}]${pnl:,.2f}[/{color}]",
                f"[{color}]{pnl_pct:+.2f}%[/{color}]",
            )

        pos_table.add_row(
            "[bold]TOTAL[/bold]", "", "", "",
            f"[{'green' if total_unrealized >= 0 else 'red'}]"
            f"${total_unrealized:,.2f}[/]",
            "",
        )

        # Recent trades
        recent = get_recent_trades(10)
        trades_table = Table(title="Recent Trades")
        trades_table.add_column("Time", style="dim")
        trades_table.add_column("Symbol", style="cyan")
        trades_table.add_column("Side")
        trades_table.add_column("Qty", justify="right")
        trades_table.add_column("Price", justify="right")
        trades_table.add_column("Strategy")

        for trade in recent:
            side_color = "green" if trade["side"] == "buy" else "red"
            trades_table.add_row(
                trade["timestamp"][:19],
                trade["symbol"],
                f"[{side_color}]{trade['side'].upper()}[/{side_color}]",
                f"{trade['qty']:.2f}",
                f"${trade['price']:.2f}",
                trade.get("strategy", ""),
            )

        # Market clock
        clock = self.execution.get_clock()
        clock_str = (
            "[green]MARKET OPEN[/green]" if clock.get("is_open")
            else f"[red]MARKET CLOSED[/red] | Next open: {clock.get('next_open', 'N/A')}"
        )

        header = (
            f"[bold]QuantClaude AI Trading System[/bold] | "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | "
            f"{clock_str}"
        )

        # Compose output
        output = Table.grid(padding=1)
        output.add_row(account_table, risk_table)
        output.add_row(pos_table)
        output.add_row(trades_table)

        return Panel(output, title=header, border_style="blue")

    def get_summary(self) -> str:
        """Get a text summary of current state."""
        account = self.execution.get_account()
        positions = self.execution.get_positions()

        equity = account.get("equity", 0)
        cash = account.get("cash", 0)
        n_pos = len(positions)
        total_pnl = sum(p["unrealized_pl"] for p in positions.values())

        return (
            f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | "
            f"Positions: {n_pos} | Unrealized P&L: ${total_pnl:,.2f} | "
            f"Halted: {self.risk_manager.is_halted}"
        )
