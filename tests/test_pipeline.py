"""Smoke tests for the futures momentum pipeline.
Runs on synthetic data, no API calls."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from fin_crypto_futures_momentum.backtest import run_backtest
from fin_crypto_futures_momentum.metrics_overfit import (
    deflated_sharpe_ratio,
    pbo_cscv,
    sharpe_moments,
)
from fin_crypto_futures_momentum.panel import Panel, _ffill_after_first
from fin_crypto_futures_momentum.signals import momentum
from fin_crypto_futures_momentum.sweep import (
    config_name,
    period_returns,
    topn_targets,
    weekly_cagr,
    weekly_sharpe,
)


def _synthetic_panel(n_days=200, n_syms=5, seed=42):
    rng = np.random.default_rng(seed)
    sessions = pl.date_range(
        dt.date(2023, 1, 1),
        dt.date(2023, 1, 1) + dt.timedelta(days=n_days - 1),
        eager=True,
    )
    prices = 100.0 * np.exp(np.cumsum(
        rng.normal(0.0005, 0.02, (n_days, n_syms)), axis=0))
    opens = prices * (1 + rng.normal(0, 0.001, prices.shape))
    volumes = rng.uniform(1000, 10000, prices.shape)
    return Panel(
        sessions=sessions,
        symbols=[f"SYM{i}USD" for i in range(n_syms)],
        open=opens,
        close=prices,
        close_ff=prices.copy(),
        volume=volumes,
        tradable=np.ones((n_days, n_syms), dtype=bool),
    )


def test_momentum_signal():
    panel = _synthetic_panel()
    sig = momentum(panel, f_idx=100, lookback=90, skip=7)
    assert sig.shape == (5,)
    assert np.all(np.isfinite(sig))


def test_momentum_lookback_guard():
    panel = _synthetic_panel()
    with pytest.raises(ValueError):
        momentum(panel, f_idx=50, lookback=90)


def test_backtest_runs():
    panel = _synthetic_panel()
    targets = pl.DataFrame({
        "formation_date": [panel.sessions[100]] * 2,
        "symbol": ["SYM0USD", "SYM1USD"],
        "weight": [0.5, 0.5],
    })
    res = run_backtest(
        panel, targets, nav0=100_000.0, slip_bp=10.0,
        cost_frac_fn=lambda n: 5.0 / 1e4,
    )
    assert res.nav.height == len(panel.sessions)
    assert res.nav["nav"][-1] > 0


def test_topn_targets():
    universe = pl.DataFrame({
        "snapshot_date": [dt.date(2023, 4, 9)] * 5,
        "symbol": [f"SYM{i}USD" for i in range(5)],
        "rank": list(range(1, 6)),
        "avg_daily_volume": [1e6] * 5,
    })
    sig = {dt.date(2023, 4, 9): {f"SYM{i}USD": float(i) for i in range(5)}}
    tgt = topn_targets(sig, universe, n=3, min_names=2)
    assert tgt.height == 3
    chosen = set(tgt["symbol"].to_list())
    assert "SYM4USD" in chosen
    assert "SYM3USD" in chosen


def test_weekly_sharpe_and_cagr():
    rets = [0.01, -0.005, 0.02, 0.003, -0.01]
    sr = weekly_sharpe(rets)
    cagr = weekly_cagr(rets)
    assert np.isfinite(sr)
    assert np.isfinite(cagr)


def test_dsr():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 100)
    sr, skew, kurt = sharpe_moments(r)
    dsr = deflated_sharpe_ratio(sr, 100, skew, kurt, [sr, sr * 0.5])
    assert 0.0 <= dsr <= 1.0


def test_pbo():
    rng = np.random.default_rng(0)
    m = rng.normal(0.001, 0.02, (160, 4))
    p = pbo_cscv(m, s_blocks=8)
    assert 0.0 <= p <= 1.0


def test_ffill():
    a = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan]])
    out = _ffill_after_first(a)
    assert np.isnan(out[0, 0])
    assert out[1, 0] == 2.0
    assert out[2, 0] == 2.0
    assert out[0, 1] == 1.0
    assert out[1, 1] == 1.0
    assert out[2, 1] == 1.0


def test_config_name():
    assert config_name({"top_n": 10, "lookback": 90, "skip": 7}) == "futures_mom_90d_top10"


def test_period_returns():
    nav = pl.DataFrame({
        "date": [dt.date(2023, 1, 1), dt.date(2023, 1, 8),
                 dt.date(2023, 1, 15)],
        "nav": [100.0, 110.0, 105.0],
    })
    rets = period_returns(nav, [dt.date(2023, 1, 1), dt.date(2023, 1, 8),
                                dt.date(2023, 1, 15)])
    assert len(rets) == 2
    assert abs(rets[0] - 0.1) < 1e-10
