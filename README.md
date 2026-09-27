# fin-crypto-futures-momentum

Cross-sectional momentum on Kraken perpetual futures. This project tests the
same signal family as `fin-crypto-lab`, but on a futures venue (5bp/side) not
on spot (80bp/side).

## Source

This project came from research idea #1 in `fin-crypto-lab` (2026-09-19). The
spot momentum strategy did not pass the battery: test Sharpe 0.20, DSR 0.899. The
hypothesis is that the signal works but spot costs (80bp/side) remove the edge.
Futures cost 16x less.

## What we know

- Kraken perpetual futures (PF_ prefix) started on 2022-03-22.
- The candle endpoint (`futures.kraken.com/api/charts/v1/trade/{symbol}/1d`)
  gives the full history from the start date. No auth is necessary.
- The BTC perpetual has about 1642 daily candles. 99 pairs have a signal at a
  90-day lookback. 89 pairs have a signal at a 180-day lookback.
- Cost: 5bp/side taker.
- The `fin-crypto-lab` spot pipeline uses `api.kraken.com`. Futures use a
  different API at `futures.kraken.com`.

## Train/test divide

The `fin-crypto-lab` divide (2017-2022 train / 2022-2026 test) was for spot data.
It keeps the futures train window empty.

This project uses: 2022-07 to 2024-06 train, 2024-07 to 2026-09 test. The
history is short, so the overfitting risk is high. The honesty battery is more
important here than in `fin-crypto-lab`.

## Plan

### Phase 0: Data pipeline

1. Add a download path in `kraken_client.py` for
   `futures.kraken.com/api/charts/v1/trade/{symbol}/1d`.
2. Download daily OHLCV for all PF_ pairs. Record the pair count and history depth.
3. Compare BTC perpetual daily closes with BTC spot daily closes. The basis
   must be small and mean-reverting.

### Phase 1: Signal sweep

4. Operate the same lookback grid as `fin-crypto-lab` (90, 180, 365 days) with
   cross-sectional momentum on the futures data.
5. Apply the honesty battery: DSR, PBO, kill conditions. Use 5bp/side cost.
6. If a config passes, compare it to the same spot config. The improvement must
   come from lower costs, not from a different data window.

### Phase 2: Execution design (only if Phase 1 passes)

7. Weekly rotation on Kraken perpetual futures.
8. Add a funding-rate drag model. Momentum is a directional strategy, so funding
   costs apply (unlike delta-neutral carry).

## Honesty method

This project uses the same battery as `fin-crypto-lab`: DSR, PBO, train/test
Sharpe, kill conditions. The file `metrics_overfit.py` is a verbatim copy from
`fin-equity-lab`. Do not change it.

## Key risk

Only about 4 years of data exist. The train window is 2 years. With 6 grid
configs and 3 lookbacks, multiple-testing correction is necessary. A PASS here
has lower conviction than a PASS on 9 years of spot data.

## Paper

Liu, Tsyvinski & Wu 2022: "Common Risk Factors in Cryptocurrency"
(https://doi.org/10.1093/rfs/hhab066).
