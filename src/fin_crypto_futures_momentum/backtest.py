"""NAV engine. Percentage-based costs.
Signal on close of formation day, fill at open of next session."""
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl

from fin_crypto_futures_momentum.panel import Panel

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    nav: pl.DataFrame
    returns: pl.DataFrame
    turnover: pl.DataFrame
    trades: pl.DataFrame
    summary: dict = field(default_factory=dict)


def run_backtest(
    panel: Panel,
    targets: pl.DataFrame,
    nav0: float,
    slip_bp: float,
    cost_frac_fn: Callable[[float], float],
) -> BacktestResult:
    sym_idx = {s: j for j, s in enumerate(panel.symbols)}
    t_len = len(panel.sessions)
    sessions_list = panel.sessions.to_list()

    formations = sorted(targets["formation_date"].unique().to_list())
    form_idxs = [panel.session_index(d) for d in formations]

    target_by_form: dict[dt.date, dict[str, float]] = {}
    for f_date, grp in targets.group_by("formation_date"):
        key = f_date[0] if isinstance(f_date, tuple) else f_date
        target_by_form[key] = dict(grp.select("symbol", "weight").iter_rows())

    shares = np.zeros(len(panel.symbols))
    cash = nav0
    nav_path = np.empty(t_len)
    trade_rows: list[dict] = []
    turn_rows: list[dict] = []

    next_form = 0
    for t in range(t_len):
        if next_form < len(form_idxs) and t == form_idxs[next_form] + 1:
            f_date = formations[next_form]
            w = target_by_form[f_date]
            open_px = panel.open[t, :]
            val_px = panel.close_ff[t, :]

            pretrade_nav = cash + float(
                np.nansum(np.where(shares != 0.0, shares * val_px, 0.0))
            )
            new_shares = shares.copy()
            new_cash = cash
            traded = 0.0

            touched = set(w) | {
                panel.symbols[j] for j in np.nonzero(shares)[0]
            }
            for s in sorted(touched):
                j = sym_idx[s]
                target_w = w.get(s, 0.0)
                if not bool(panel.tradable[t, j]):
                    continue
                px = float(open_px[j])
                cur_notional = float(shares[j]) * px
                tgt_notional = target_w * pretrade_nav
                delta = tgt_notional - cur_notional
                if abs(delta) < 1e-9:
                    continue

                notional = abs(delta)
                cost = cost_frac_fn(notional) * notional
                slip_cost = slip_bp / 1e4 * notional

                new_shares[j] = tgt_notional / px
                new_cash -= delta + cost + slip_cost
                traded += notional

                trade_rows.append({
                    "exec_date": sessions_list[t],
                    "symbol": s,
                    "notional": delta,
                    "cost": cost + slip_cost,
                })

            shares = new_shares
            cash = new_cash
            turn_rows.append({
                "exec_date": sessions_list[t],
                "traded_notional": traded,
                "pretrade_nav": pretrade_nav,
                "turnover": traded / pretrade_nav if pretrade_nav else 0.0,
            })
            next_form += 1

        val_px = panel.close_ff[t, :]
        nav_path[t] = cash + float(
            np.nansum(np.where(shares != 0.0, shares * val_px, 0.0))
        )

    nav_df = pl.DataFrame({"date": sessions_list, "nav": nav_path})
    rets = np.empty(t_len)
    rets[0] = nav_path[0] / nav0 - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rets[1:] = nav_path[1:] / nav_path[:-1] - 1.0
    ret_df = pl.DataFrame({"date": sessions_list, "ret": rets})

    empty_turn = {
        "exec_date": pl.Series([], dtype=pl.Date),
        "traded_notional": pl.Series([], dtype=pl.Float64),
        "pretrade_nav": pl.Series([], dtype=pl.Float64),
        "turnover": pl.Series([], dtype=pl.Float64),
    }
    empty_trades = {
        "exec_date": pl.Series([], dtype=pl.Date),
        "symbol": pl.Series([], dtype=pl.Utf8),
        "notional": pl.Series([], dtype=pl.Float64),
        "cost": pl.Series([], dtype=pl.Float64),
    }

    return BacktestResult(
        nav=nav_df, returns=ret_df,
        turnover=(pl.DataFrame(turn_rows) if turn_rows
                  else pl.DataFrame(empty_turn)),
        trades=(pl.DataFrame(trade_rows) if trade_rows
                else pl.DataFrame(empty_trades)),
    )


def slippage_sweep(
    panel: Panel, targets: pl.DataFrame, nav0: float,
    slip_levels: tuple[float, ...],
    cost_frac_fn: Callable[[float], float],
) -> dict[float, BacktestResult]:
    return {
        bp: run_backtest(panel, targets, nav0=nav0, slip_bp=bp,
                         cost_frac_fn=cost_frac_fn)
        for bp in slip_levels
    }
