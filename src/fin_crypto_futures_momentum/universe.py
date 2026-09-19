"""Point-in-time volume-ranked universe. At each formation date,
rank by trailing average daily USD volume."""
import datetime as dt
from pathlib import Path

import polars as pl

UNIVERSE_SCHEMA = {
    "snapshot_date": pl.Date,
    "symbol": pl.Utf8,
    "rank": pl.UInt32,
    "avg_daily_volume": pl.Float64,
}

MIN_DAYS = 30


def build_universe(
    ohlcv_dir: Path,
    formations: list[dt.date],
    lookback: int = 90,
    top_n: int = 20,
) -> pl.DataFrame:
    pair_data: dict[str, pl.DataFrame] = {}
    for f in sorted(ohlcv_dir.glob("*.parquet")):
        pair_data[f.stem] = pl.read_parquet(f)

    rows = []
    for f_date in sorted(formations):
        lo = f_date - dt.timedelta(days=lookback)
        eligible = []
        for pair, df in pair_data.items():
            window = df.filter(
                (pl.col("date") > lo) & (pl.col("date") <= f_date)
            )
            if window.height < MIN_DAYS:
                continue
            avg_vol = float((window["close"] * window["volume"]).mean())
            eligible.append((pair, avg_vol))

        eligible.sort(key=lambda x: -x[1])
        for rank, (pair, avg_vol) in enumerate(eligible[:top_n], 1):
            rows.append({
                "snapshot_date": f_date,
                "symbol": pair,
                "rank": rank,
                "avg_daily_volume": avg_vol,
            })
    if not rows:
        return pl.DataFrame(schema=UNIVERSE_SCHEMA)
    return pl.DataFrame(rows, schema=UNIVERSE_SCHEMA)
