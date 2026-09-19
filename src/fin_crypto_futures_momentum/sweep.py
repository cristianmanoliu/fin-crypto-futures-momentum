"""Grid, thresholds, target construction, metrics.
Pre-registered constants: the runner never defines grid or thresholds."""
import datetime as dt

import numpy as np
import polars as pl

GRID = [
    {"top_n": n, "lookback": lb, "skip": 7}
    for lb in (90, 180, 365)
    for n in (10, 20, 30)
]

THRESHOLDS = {
    "DSR_MIN": 0.90,
    "PBO_MAX": 0.50,
    "COST_RECON_TOL": 0.50,
    "KC1_MAX_CAGR": 1.00,
    "KC2_MAX_TURNOVER": 52.0,
    "KC4_MAX_DD": 0.80,
    "KC5_WEIGHT_TOL": 0.001,
    "KC6_MIN_NAMES": 5,
    "S_BLOCKS": 16,
    "N_TRIALS": 9,
}


def config_name(g: dict) -> str:
    return f"futures_mom_{g['lookback']}d_top{g['top_n']}"


def topn_targets(
    signal_by_formation: dict[dt.date, dict[str, float]],
    universe: pl.DataFrame,
    n: int,
    min_names: int = 0,
) -> pl.DataFrame:
    rows = []
    for f_date in sorted(signal_by_formation):
        snap = universe.filter(pl.col("snapshot_date") == f_date)
        if snap.height == 0:
            past = universe.filter(pl.col("snapshot_date") <= f_date)
            if past.height == 0:
                continue
            latest = past["snapshot_date"].max()
            snap = universe.filter(pl.col("snapshot_date") == latest)
        members = set(snap["symbol"].to_list())
        eligible = sorted(
            ((v, s) for s, v in signal_by_formation[f_date].items()
             if s in members and v is not None and not np.isnan(v)),
            key=lambda t: (-t[0], t[1]),
        )
        chosen = [s for _, s in eligible[:n]]
        if len(chosen) < max(min_names, 1):
            continue
        w = 1.0 / len(chosen)
        rows += [{"formation_date": f_date, "symbol": s, "weight": w}
                 for s in chosen]
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "formation_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64,
    })


def period_returns(
    nav: pl.DataFrame, formations: list[dt.date],
) -> list[float]:
    marks = nav.filter(pl.col("date").is_in(formations)).sort("date")
    navs = marks["nav"].to_list()
    return [(navs[i] / navs[i - 1]) - 1.0 for i in range(1, len(navs))]


def weekly_sharpe(weekly_rets) -> float:
    r = np.asarray(weekly_rets, dtype=float)
    sd = np.std(r, ddof=1)
    if sd < 1e-15:
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(52.0))


def weekly_cagr(weekly_rets) -> float:
    r = np.asarray(weekly_rets, dtype=float)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return float("nan")
    return growth ** (52.0 / len(r)) - 1.0


def benchmark_targets(
    universe: pl.DataFrame, formations: list[dt.date],
) -> pl.DataFrame:
    rows = []
    for f_date in formations:
        snap = universe.filter(pl.col("snapshot_date") == f_date)
        if snap.height == 0:
            past = universe.filter(pl.col("snapshot_date") <= f_date)
            if past.height == 0:
                continue
            latest = past["snapshot_date"].max()
            snap = universe.filter(pl.col("snapshot_date") == latest)
        members = sorted(snap["symbol"].to_list())
        if not members:
            continue
        w = 1.0 / len(members)
        rows += [{"formation_date": f_date, "symbol": s, "weight": w}
                 for s in members]
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "formation_date": pl.Date, "symbol": pl.Utf8, "weight": pl.Float64,
    })
