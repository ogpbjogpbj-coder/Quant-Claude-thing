#!/usr/bin/env python3
"""QuantClaude AI Trading System — CLI entry point.

Usage:
    python main.py run          # Start live/paper trading
    python main.py backtest     # Run backtest
    python main.py status       # Show current portfolio status
    python main.py emergency    # Emergency: close all positions
"""

import argparse
import sys

from loguru import logger


def cmd_run(args):
    """Start the live trading system."""
    from trading_system.config import load_config
    from trading_system.orchestrator import TradingOrchestrator

    config = load_config(args.config)

    if args.paper:
        config.alpaca.base_url = "https://paper-api.alpaca.markets"
        config.mode = "paper"

    orchestrator = TradingOrchestrator(config)

    if args.once:
        orchestrator.run_cycle()
    else:
        orchestrator.start()


def cmd_backtest(args):
    """Run a backtest."""
    from trading_system.config import load_config
    from trading_system.backtester import Backtester
    from trading_system.utils.logger import setup_logger

    setup_logger()
    config = load_config(args.config)
    bt = Backtester(config)
    result = bt.run(
        initial_capital=args.capital,
        lookback_days=args.days,
        max_positions=args.max_positions,
    )
    print(result.summary())


def cmd_status(args):
    """Show current portfolio status."""
    from trading_system.config import load_config
    from trading_system.execution import ExecutionEngine
    from trading_system.portfolio import PortfolioManager
    from trading_system.risk_manager import RiskManager
    from trading_system.monitor import Monitor
    from trading_system.utils.logger import setup_logger

    setup_logger()
    config = load_config(args.config)
    execution = ExecutionEngine(config.alpaca, config.execution)
    risk_manager = RiskManager(config.risk)
    portfolio = PortfolioManager(config, execution, risk_manager)

    account = execution.get_account()
    risk_manager.initialize(account.get("equity", 0))

    monitor = Monitor(execution, portfolio, risk_manager)
    monitor.print_status()


def cmd_emergency(args):
    """Emergency shutdown — close all positions."""
    from trading_system.config import load_config
    from trading_system.execution import ExecutionEngine
    from trading_system.utils.logger import setup_logger

    setup_logger()
    config = load_config(args.config)
    execution = ExecutionEngine(config.alpaca, config.execution)

    print("!!! EMERGENCY SHUTDOWN !!!")
    print("This will close ALL positions and cancel ALL orders.")
    confirm = input("Type 'CONFIRM' to proceed: ")

    if confirm == "CONFIRM":
        execution.cancel_all_orders()
        execution.close_all_positions(reason="manual emergency shutdown")
        print("All positions closed, all orders cancelled.")
    else:
        print("Aborted.")


def main():
    parser = argparse.ArgumentParser(
        description="QuantClaude AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to trading_config.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = subparsers.add_parser("run", help="Start live trading")
    run_parser.add_argument("--paper", action="store_true", help="Use paper trading")
    run_parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    run_parser.set_defaults(func=cmd_run)

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    bt_parser.add_argument("--days", type=int, default=252, help="Lookback days")
    bt_parser.add_argument("--max-positions", type=int, default=10, help="Max positions")
    bt_parser.set_defaults(func=cmd_backtest)

    # status
    status_parser = subparsers.add_parser("status", help="Show portfolio status")
    status_parser.set_defaults(func=cmd_status)

    # emergency
    emerg_parser = subparsers.add_parser("emergency", help="Emergency close all")
    emerg_parser.set_defaults(func=cmd_emergency)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
