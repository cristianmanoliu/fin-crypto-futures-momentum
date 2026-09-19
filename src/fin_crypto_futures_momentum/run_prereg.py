"""Pre-registered single-config verdict.
Run: uv run python -m fin_crypto_futures_momentum.run_prereg
Exit 0 = PASS, 1 = FAIL, 2 = kill condition."""
import datetime as dt
import logging
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from fin_crypto_futures_momentum import config
from fin_crypto_futures_momentum.backtest import slippage_sweep
from fin_crypto_futures_momentum.metrics_overfit import sharpe_moments
from fin_crypto_futures_momentum.panel import (
    build_panel,
    crypto_sessions,
    weekly_formations,
)
from fin_crypto_futures_momentum.signals import momentum
from fin_crypto_futures_momentum.sweep import (
    THRESHOLDS,
    benchmark_targets,
    config_name,
    period_returns,
    topn_targets,
    weekly_cagr,
    weekly_sharpe,
)
from fin_crypto_futures_momentum.universe import build_universe

log = logging.getLogger(__name__)

PREREG_CONFIG = {"top_n": 20, "lookback": 365, "skip": 7}

_ND = NormalDist()


def single_trial_dsr(sr_hat: float, t_obs: int,
                     skew: float, kurt: float) -> float:
    """DSR with one trial: sr0 = 0, no multiple-testing penalty."""
    denom = math.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    return _ND.cdf(sr_hat * math.sqrt(t_obs - 1) / denom)


def write_prereg_verdict(row, checks, family_pass, dsr,
                         results_dir: Path = config.RESULTS_DIR) -> Path:
    out_dir = results_dir / f"prereg_365d_top20_{dt.date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Pre-registered verdict ({dt.date.today()})",
        "",
        f"## CONFIG: {row['name']}",
        f"## VERDICT: {'PASS' if family_pass else 'FAIL'}",
        "",
        f"DSR (single trial): {dsr:.3f}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| train Sharpe | {row['train_sharpe']:.2f} |",
        f"| test Sharpe | {row['test_sharpe']:.2f} |",
        f"| test CAGR | {row['test_cagr']:.2%} |",
        f"| full CAGR | {row['full_cagr']:.2%} |",
        f"| turnover | {row['turnover']:.2f} |",
        f"| min names | {row['min_names']} |",
        "",
        "| check | result | measured |",
        "|---|---|---|",
    ]
    for cid, passed, meas in checks:
        lines.append(f"| {cid} | {'PASS' if passed else 'FAIL'} | {meas} |")
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    return out_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    g = PREREG_CONFIG
    name = config_name(g)
    lb, sk = g["lookback"], g["skip"]

    sessions = crypto_sessions(
        str(config.FORM_START - dt.timedelta(days=400)),
        str(config.FORM_END))
    formations = [d for d in weekly_formations(sessions)
                  if config.FORM_START <= d <= config.FORM_END]

    ohlcv_dir = config.DATA_DIR / "ohlcv"
    universe = build_universe(ohlcv_dir, formations, lookback=90,
                              top_n=g["top_n"])
    log.info("universe: %d snapshots, %d unique symbols",
             universe["snapshot_date"].n_unique(),
             universe["symbol"].n_unique())

    symbols = sorted(universe["symbol"].unique().to_list())
    panel = build_panel(symbols, sessions, data_dir=config.DATA_DIR)
    f_idx = {d: panel.session_index(d) for d in formations}

    bm_targets = benchmark_targets(universe, formations)
    bm_sw = slippage_sweep(panel, bm_targets, nav0=config.NAV_DEFAULT,
                           slip_levels=config.SLIP_LEVELS_BP,
                           cost_frac_fn=config.cost_frac)
    bm_res = bm_sw[config.DECISION_SLIP_BP]
    bm_weekly = period_returns(bm_res.nav, formations)
    bm_marks = bm_res.nav.filter(
        pl.col("date").is_in(formations)).sort("date")
    bm_dates = bm_marks["date"].to_list()[1:]
    bm_test = [r for r, d in zip(bm_weekly, bm_dates)
               if d > config.TRAIN_END]

    sig_by_form: dict[dt.date, dict[str, float]] = {}
    for d in formations:
        fi = f_idx[d]
        if fi < lb:
            continue
        raw = momentum(panel, fi, lookback=lb, skip=sk)
        sig_by_form[d] = dict(zip(panel.symbols, raw.tolist()))

    tgt = topn_targets(sig_by_form, universe, n=g["top_n"],
                       min_names=THRESHOLDS["KC6_MIN_NAMES"])
    sw = slippage_sweep(panel, tgt, nav0=config.NAV_DEFAULT,
                        slip_levels=config.SLIP_LEVELS_BP,
                        cost_frac_fn=config.cost_frac)
    res = sw[config.DECISION_SLIP_BP]

    nav = res.nav["nav"].to_numpy()
    nonfinite = bool((~np.isfinite(nav)).any() or (nav <= 0).any())

    cfg_weekly = period_returns(res.nav, formations)
    marks = res.nav.filter(pl.col("date").is_in(formations)).sort("date")
    dates = marks["date"].to_list()[1:]
    train = [r for r, d in zip(cfg_weekly, dates)
             if d <= config.TRAIN_END]
    test = [r for r, d in zip(cfg_weekly, dates)
            if d > config.TRAIN_END]
    n_days = res.nav.height - 1

    names_per_form = (
        tgt.group_by("formation_date")
        .agg(pl.col("symbol").n_unique().alias("n_names"))
    )
    min_names = (int(names_per_form["n_names"].min())
                 if names_per_form.height else 0)

    row = {
        "name": name,
        "train_sharpe": weekly_sharpe(train),
        "test_sharpe": weekly_sharpe(test),
        "test_cagr": weekly_cagr(test),
        "full_cagr": weekly_cagr(cfg_weekly),
        "turnover": float(
            res.turnover["turnover"].sum() * 365 / n_days),
        "min_names": min_names,
    }
    log.info("%s: train SR %.2f test SR %.2f", name,
             row["train_sharpe"], row["test_sharpe"])

    own = np.asarray(cfg_weekly)
    sr, skew, kurt = sharpe_moments(own)
    dsr = single_trial_dsr(sr, len(own), skew, kurt)

    nav_test = res.nav.filter(pl.col("date") > config.TRAIN_END)
    nav_s = nav_test["nav"]
    test_dd = float(-(nav_s / nav_s.cum_max() - 1.0).min())

    gross_nav = sw[0.0].nav
    years = n_days / 365.0
    drag = (
        (gross_nav["nav"][-1] / gross_nav["nav"][0]) ** (1 / years)
        - (res.nav["nav"][-1] / res.nav["nav"][0]) ** (1 / years)
    )
    costs_frac = (float(res.trades["cost"].sum())
                  / float(res.nav["nav"].mean()) / years)
    recon_err = (abs(drag - costs_frac) / costs_frac
                 if costs_frac else float("inf"))

    weight_sums = (
        tgt.group_by("formation_date")
        .agg(pl.col("weight").sum().alias("wsum"))
    )
    min_wsum = float(weight_sums["wsum"].min())
    max_wsum = float(weight_sums["wsum"].max())
    tol = THRESHOLDS["KC5_WEIGHT_TOL"]
    kc5_pass = (min_wsum >= 1.0 - tol) and (max_wsum <= 1.0 + tol)

    bm_test_sharpe = weekly_sharpe(bm_test)

    checks = [
        ("PC-1", row["test_sharpe"] >= bm_test_sharpe,
         f"{row['test_sharpe']:.2f} vs benchmark {bm_test_sharpe:.2f}"),
        ("PC-3", dsr >= THRESHOLDS["DSR_MIN"],
         f"DSR {dsr:.3f} (single trial, sr0=0)"),
        ("PC-5", recon_err <= THRESHOLDS["COST_RECON_TOL"],
         f"drag {drag:.2%} vs costs {costs_frac:.2%} "
         f"(err {recon_err:.0%})"),
        ("KC-1", row["full_cagr"] <= THRESHOLDS["KC1_MAX_CAGR"],
         f"full CAGR {row['full_cagr']:.2%}"),
        ("KC-2", row["turnover"] <= THRESHOLDS["KC2_MAX_TURNOVER"],
         f"turnover {row['turnover']:.2f}"),
        ("KC-3", not nonfinite, str(not nonfinite)),
        ("KC-4", test_dd <= THRESHOLDS["KC4_MAX_DD"],
         f"test maxDD {test_dd:.2%}"),
        ("KC-5", kc5_pass,
         f"min Σw {min_wsum:.4f}, max {max_wsum:.4f}"),
        ("KC-6", min_names >= THRESHOLDS["KC6_MIN_NAMES"],
         f"min names {min_names}"),
    ]
    kills = [c for c in checks if c[0].startswith("KC") and not c[1]]
    family_pass = all(passed for _, passed, _ in checks)

    out = write_prereg_verdict(row, checks, family_pass, dsr)
    for cid, passed, meas in checks:
        log.info("%-5s %s  %s", cid, "PASS" if passed else "FAIL", meas)
    log.info("VERDICT: %s — %s", "PASS" if family_pass else "FAIL", out)
    return 2 if kills else (0 if family_pass else 1)


if __name__ == "__main__":
    sys.exit(main())
