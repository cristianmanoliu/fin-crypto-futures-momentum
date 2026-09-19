# fin-crypto-futures-momentum

Cross-sectional momentum on Kraken perpetual futures. The same signal family
as `fin-crypto-lab`, but on a cheaper venue (5bp/side vs 80bp/side spot) with
a train/test split that matches instrument availability.

## Origin

Research idea #1 from `fin-crypto-lab` (2026-09-19). Spot momentum FAIL
(test Sharpe 0.20, DSR 0.899). The hypothesis: the signal works but spot
costs (80bp/side) eat the edge. Futures cost 16x less.

## What we know

- Kraken perpetual futures (PF_ prefix) launched 2022-03-22.
- The candle endpoint (`futures.kraken.com/api/charts/v1/trade/{symbol}/1d`)
  returns full history since inception. No auth required.
- BTC perpetual has ~1642 daily candles. 99 pairs have signal at 90-day
  lookback; 89 pairs at 180-day.
- Cost: 5bp/side taker.
- The `fin-crypto-lab` spot data pipeline uses `api.kraken.com`. Futures use
  a separate API at `futures.kraken.com`.

## Proposed train/test split

The current `fin-crypto-lab` split (2017-2022 / 2022-2026) was designed for
spot. It leaves the futures train window empty.

Proposed: 2022-07 to 2024-06 train, 2024-07 to 2026-09 test. Short history,
so overfitting risk is high. The honesty battery matters more here than usual.

## What to do next

### Phase 0: Data pipeline

1. Implement a new download path in `kraken_client.py` (or a new client) for
   `futures.kraken.com/api/charts/v1/trade/{symbol}/1d`.
2. Download daily OHLCV for all PF_ pairs. Catalog pair count and history
   depth.
3. Validate: compare BTC perpetual closes to BTC spot closes. The basis
   should be small and mean-reverting.

### Phase 1: Signal sweep

4. Run the same lookback grid as `fin-crypto-lab` (90, 180, 365 days) with
   cross-sectional momentum on futures data.
5. Apply the honesty battery: DSR, PBO, kill conditions. Use 5bp/side cost.
6. If any config passes, compare to the equivalent spot config. The
   improvement should come from lower costs, not from a different data window.

### Phase 2: Execution design (only if Phase 1 passes)

7. Weekly rotation on Kraken perpetual futures.
8. Funding rate drag model: momentum is a directional strategy, so funding
   costs apply (unlike delta-neutral carry).

## Honesty method

Same battery as `fin-crypto-lab`: DSR, PBO, train/test Sharpe comparison,
kill conditions. Copy `metrics_overfit.py` from `fin-equity-lab` (verbatim).

## Key risk

**Short history.** Only ~4 years of data. The train window is 2 years. With
6 grid configs and 3 lookbacks, multiple-testing correction is essential.
A PASS here carries lower conviction than a PASS on 9 years of spot data.

## Reference

Liu, Tsyvinski & Wu 2022: "Common Risk Factors in Cryptocurrency"
(https://doi.org/10.1093/rfs/hhab066).
