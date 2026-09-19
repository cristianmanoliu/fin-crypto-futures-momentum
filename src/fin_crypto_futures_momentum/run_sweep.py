"""Futures momentum sweep runner. Full honesty battery.
Run: uv run python -m fin_crypto_futures_momentum.run_sweep
Exit 0 = PASS, 1 = FAIL, 2 = kill condition."""
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fin_crypto_futures_momentum import config
from fin_crypto_futures_momentum.backtest import slippage_sweep
from fin_crypto_futures_momentum.metrics_overfit import (
    deflated_sharpe_ratio,
    pbo_cscv,
    sharpe_moments,
)
from fin_crypto_futures_momentum.panel import (
    build_panel,
    crypto_sessions,
    weekly_formations,
)
from fin_crypto_futures_momentum.signals import momentum
from fin_crypto_futures_momentum.sweep import (
    GRID,
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


def write_verdict(rows, checks, family_pass, extra,
                  results_dir: Path = config.RESULTS_DIR) -> Path:
    out_dir = results_dir / f"futures_momentum_{dt.date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Futures momentum verdict ({dt.date.today()})",
        "",
        f"## FAMILY: {'PASS' if family_pass else 'FAIL'}",
        "",
        f"Selected: **{extra['selected']}** "
        f"(DSR {extra['dsr']:.3f}, PBO {extra['pbo']:.3f})",
        "",
        "| check | result | measured |",
        "|---|---|---|",
    ]
    for cid, passed, meas in checks:
        lines.append(f"| {cid} | {'PASS' if passed else 'FAIL'} | {meas} |")
    lines += [
        "",
        "## Grid",
        "",
        "| config | train Sharpe | test Sharpe | test CAGR | full CAGR "
        "| turnover | min names |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["train_sharpe"]):
        label = f"**{r['name']}** **selected**" if r["selected"] else r["name"]
        lines.append(
            f"| {label} | {r['train_sharpe']:.2f} | {r['test_sharpe']:.2f} "
            f"| {r['test_cagr']:.2%} | {r['full_cagr']:.2%} "
            f"| {r['turnover']:.2f} | {r['min_names']} |"
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    return out_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    sessions = crypto_sessions(
        str(config.FORM_START - dt.timedelta(days=400)),
        str(config.FORM_END))
    formations = [d for d in weekly_formations(sessions)
                  if config.FORM_START <= d <= config.FORM_END]

    ohlcv_dir = config.DATA_DIR / "ohlcv"
    max_n = max(g["top_n"] for g in GRID)
    universe = build_universe(ohlcv_dir, formations, lookback=90, top_n=max_n)
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

    rows, weekly_own = [], {}
    results_cache: dict[str, dict] = {}
    nonfinite = False

    for g in GRID:
        name = config_name(g)
        lb, sk = g["lookback"], g["skip"]

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
        results_cache[name] = {"sw": sw, "res": res, "tgt": tgt}
        nav = res.nav["nav"].to_numpy()
        nonfinite |= bool((~np.isfinite(nav)).any() or (nav <= 0).any())

        cfg_weekly = period_returns(res.nav, formations)
        weekly_own[name] = list(cfg_weekly)

        marks = res.nav.filter(
            pl.col("date").is_in(formations)).sort("date")
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

        rows.append({
            "name": name,
            "train_sharpe": weekly_sharpe(train),
            "test_sharpe": weekly_sharpe(test),
            "test_cagr": weekly_cagr(test),
            "full_cagr": weekly_cagr(cfg_weekly),
            "turnover": float(
                res.turnover["turnover"].sum() * 365 / n_days),
            "min_names": min_names,
            "selected": False,
        })
        log.info("%s: train SR %.2f test SR %.2f", name,
                 rows[-1]["train_sharpe"], rows[-1]["test_sharpe"])

    sel = max(rows, key=lambda r: r["train_sharpe"])
    sel["selected"] = True

    own_sel = np.asarray(weekly_own[sel["name"]])
    sr, skew, kurt = sharpe_moments(own_sel)
    trial_srs = [
        float(np.mean(np.asarray(weekly_own[r["name"]])) /
              np.std(np.asarray(weekly_own[r["name"]]), ddof=1))
        for r in rows
    ]
    dsr = deflated_sharpe_ratio(
        sr_hat=sr, t_obs=len(own_sel), skew=skew, kurt=kurt,
        trial_sharpes=trial_srs,
        n_trials=THRESHOLDS["N_TRIALS"])
    matrix = np.column_stack([weekly_own[r["name"]] for r in rows])
    pbo = pbo_cscv(matrix, s_blocks=THRESHOLDS["S_BLOCKS"])

    cached = results_cache[sel["name"]]
    sel_res = cached["res"]
    sel_sw = cached["sw"]
    sel_tgt = cached["tgt"]

    sel_nav_test = sel_res.nav.filter(pl.col("date") > config.TRAIN_END)
    nav_s = sel_nav_test["nav"]
    test_dd = float(-(nav_s / nav_s.cum_max() - 1.0).min())

    gross_nav = sel_sw[0.0].nav
    n_days_sel = sel_res.nav.height - 1
    years_sel = n_days_sel / 365.0
    drag = (
        (gross_nav["nav"][-1] / gross_nav["nav"][0]) ** (1 / years_sel)
        - (sel_res.nav["nav"][-1] / sel_res.nav["nav"][0]) ** (1 / years_sel)
    )
    costs_frac = (float(sel_res.trades["cost"].sum())
                  / float(sel_res.nav["nav"].mean()) / years_sel)
    recon_err = (abs(drag - costs_frac) / costs_frac
                 if costs_frac else float("inf"))

    weight_sums = (
        sel_tgt.group_by("formation_date")
        .agg(pl.col("weight").sum().alias("wsum"))
    )
    min_wsum = float(weight_sums["wsum"].min())
    max_wsum = float(weight_sums["wsum"].max())
    tol = THRESHOLDS["KC5_WEIGHT_TOL"]
    kc5_pass = (min_wsum >= 1.0 - tol) and (max_wsum <= 1.0 + tol)

    sel_min_names = sel["min_names"]
    bm_test_sharpe = weekly_sharpe(bm_test)

    checks = [
        ("PC-1", sel["test_sharpe"] >= bm_test_sharpe,
         f"{sel['test_sharpe']:.2f} vs benchmark {bm_test_sharpe:.2f}"),
        ("PC-3", dsr >= THRESHOLDS["DSR_MIN"], f"DSR {dsr:.3f}"),
        ("PC-4", pbo <= THRESHOLDS["PBO_MAX"], f"PBO {pbo:.3f}"),
        ("PC-5", recon_err <= THRESHOLDS["COST_RECON_TOL"],
         f"drag {drag:.2%} vs costs {costs_frac:.2%} "
         f"(err {recon_err:.0%})"),
        ("KC-1",
         all(r["full_cagr"] <= THRESHOLDS["KC1_MAX_CAGR"] for r in rows),
         f"max full CAGR {max(r['full_cagr'] for r in rows):.2%}"),
        ("KC-2",
         all(r["turnover"] <= THRESHOLDS["KC2_MAX_TURNOVER"]
             for r in rows),
         f"max turnover {max(r['turnover'] for r in rows):.2f}"),
        ("KC-3", not nonfinite, str(not nonfinite)),
        ("KC-4", test_dd <= THRESHOLDS["KC4_MAX_DD"],
         f"test maxDD {test_dd:.2%}"),
        ("KC-5", kc5_pass,
         f"min Σw {min_wsum:.4f}, max {max_wsum:.4f}"),
        ("KC-6", sel_min_names >= THRESHOLDS["KC6_MIN_NAMES"],
         f"min names {sel_min_names}"),
    ]
    kills = [c for c in checks if c[0].startswith("KC") and not c[1]]
    family_pass = all(passed for _, passed, _ in checks)

    out = write_verdict(
        rows, checks, family_pass,
        {"selected": sel["name"], "dsr": dsr, "pbo": pbo},
    )
    for cid, passed, meas in checks:
        log.info("%-5s %s  %s", cid, "PASS" if passed else "FAIL", meas)
    log.info("FAMILY: %s — %s", "PASS" if family_pass else "FAIL", out)
    if kills:
        return 2
    return 0 if family_pass else 1


if __name__ == "__main__":
    sys.exit(main())
