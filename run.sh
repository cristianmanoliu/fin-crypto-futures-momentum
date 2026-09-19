#!/usr/bin/env bash
# Download futures data and run the sweep.
# Usage: nohup bash run.sh > pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

echo "=== $(date) === FUTURES MOMENTUM PIPELINE START ==="

echo ""
echo "--- DOWNLOAD ---"
uv run python -m fin_crypto_futures_momentum.download
echo "--- DOWNLOAD DONE $(date) ---"

echo ""
echo "--- SWEEP ---"
uv run python -m fin_crypto_futures_momentum.run_sweep
EXIT=$?
echo "--- SWEEP EXIT: $EXIT $(date) ---"

echo ""
echo "=== $(date) === PIPELINE COMPLETE ==="
echo "EXIT: $EXIT (0=PASS, 1=FAIL, 2=KILL)"
ls -la results/futures_momentum_*/verdict.md 2>/dev/null || echo "(no verdict files found)"
