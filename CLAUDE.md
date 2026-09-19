# CLAUDE.md

## What this project is

**fin-crypto-futures-momentum**: cross-sectional momentum on Kraken perpetual
futures. Tests the hypothesis that the momentum signal from `fin-crypto-lab`
works when costs drop from 80bp/side (spot) to 5bp/side (futures).

Sibling to `fin-crypto-lab`. Same honesty battery (DSR, PBO, kill conditions).
`metrics_overfit.py` is a **verbatim copy** from fin-equity-lab. Do not modify.

## Stack

- Python 3.12+, polars, numpy, httpx, pytest. uv-managed.
- Data source: Kraken Futures REST API (public, no auth).

## Key invariants

- **24/7 calendar.** Every calendar day is a session.
- **Formation dates:** every Sunday at 00:00 UTC.
- **Fill rule:** signal on close of formation day, fill at open of next session.
- **Cost model:** 5bp/side taker.
- **Train/test split:** 2022-07-03 to 2024-06-30 / 2024-07-01 to 2026-09-14.
- **Grid:** 9 configs (lookback 90/180/365 x top 10/20/30). skip=7.

## Commands

Install: `uv sync`

Tests: `uv run pytest`

Download data: `uv run python -m fin_crypto_futures_momentum.download`

Run sweep: `uv run python -m fin_crypto_futures_momentum.run_sweep`

Full pipeline: `bash run.sh`
