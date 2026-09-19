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
echo "--- SWEEP (grid) ---"
uv run python -m fin_crypto_futures_momentum.run_sweep
SWEEP_EXIT=$?
echo "--- SWEEP EXIT: $SWEEP_EXIT $(date) ---"

echo ""
echo "--- PREREG (365d top20) ---"
uv run python -m fin_crypto_futures_momentum.run_prereg
PREREG_EXIT=$?
echo "--- PREREG EXIT: $PREREG_EXIT $(date) ---"

echo ""
echo "=== $(date) === PIPELINE COMPLETE ==="
echo "SWEEP EXIT: $SWEEP_EXIT  PREREG EXIT: $PREREG_EXIT  (0=PASS, 1=FAIL, 2=KILL)"
ls -la results/*/verdict.md 2>/dev/null || echo "(no verdict files found)"
