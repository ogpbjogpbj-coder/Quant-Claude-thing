#!/bin/bash
# Start the trading system in LIVE mode
set -euo pipefail

cd "$(dirname "$0")/.."

echo "!!! WARNING: LIVE TRADING MODE !!!"
echo "This will trade REAL MONEY on your Alpaca account."
echo ""
read -p "Type 'LIVE' to confirm: " confirm

if [ "$confirm" != "LIVE" ]; then
    echo "Aborted."
    exit 1
fi

python main.py run
