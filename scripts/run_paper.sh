#!/bin/bash
# Start the trading system in paper trading mode
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Starting QuantClaude AI Trading System (PAPER MODE)"
echo "================================================="

python main.py run --paper
