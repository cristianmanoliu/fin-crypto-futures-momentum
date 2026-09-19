"""Aligned [T sessions x N symbols] numpy matrices.
24/7 calendar: every calendar day is a session."""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fin_crypto_futures_momentum import config, store


@dataclass
class Panel:
    sessions: pl.Series
    symbols: list[str]
    open: np.ndarray
    close: np.ndarray
    close_ff: np.ndarray
    volume: np.ndarray
    tradable: np.ndarray

    def session_index(self, date: dt.date) -> int:
        idx = self.sessions.search_sorted(date)
        if idx >= len(self.sessions) or self.sessions[int(idx)] != date:
            raise KeyError(f"{date} is not a session")
        return int(idx)


def crypto_sessions(start: str, end: str) -> pl.Series:
    return pl.date_range(
        dt.date.fromisoformat(start),
        dt.date.fromisoformat(end),
        eager=True,
    )


def weekly_formations(sessions: pl.Series) -> list[dt.date]:
    return [d for d in sessions.to_list() if d.weekday() == 6]


def build_panel(
    symbols: list[str], sessions: pl.Series,
    data_dir: Path = config.DATA_DIR,
) -> Panel:
    t, n = len(sessions), len(symbols)
    grid = pl.DataFrame({"date": sessions})
    opens = np.full((t, n), np.nan)
    closes = np.full((t, n), np.nan)
    volumes = np.full((t, n), np.nan)

    for j, sym in enumerate(symbols):
        df = store.read_ohlcv(sym, data_dir=data_dir)
        aligned = grid.join(df, on="date", how="left")
        opens[:, j] = aligned["open"].cast(pl.Float64).to_numpy()
        closes[:, j] = aligned["close"].cast(pl.Float64).to_numpy()
        volumes[:, j] = aligned["volume"].cast(pl.Float64).to_numpy()

    close_ff = _ffill_after_first(closes)
    tradable = (~np.isnan(opens)) & (volumes > 0)
    return Panel(
        sessions=sessions, symbols=list(symbols),
        open=opens, close=closes, close_ff=close_ff,
        volume=volumes, tradable=tradable,
    )


def _ffill_after_first(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    t = a.shape[0]
    idx = np.where(~np.isnan(a), np.arange(t)[:, None], -1).astype(np.intp)
    np.maximum.accumulate(idx, axis=0, out=idx)
    valid = idx >= 0
    cols = np.broadcast_to(np.arange(a.shape[1]), a.shape)
    out[valid] = a[idx[valid], cols[valid]]
    return out
