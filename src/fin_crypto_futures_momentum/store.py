"""Parquet OHLCV store. One file per pair plus a manifest."""
import datetime as dt
from pathlib import Path

import polars as pl

from fin_crypto_futures_momentum import config

MANIFEST_SCHEMA = {
    "pair": pl.Utf8,
    "first_date": pl.Date,
    "last_date": pl.Date,
    "rows": pl.Int64,
    "downloaded_at": pl.Datetime("us"),
}


def read_manifest(data_dir: Path = config.DATA_DIR) -> pl.DataFrame:
    path = data_dir / "manifest.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=MANIFEST_SCHEMA)


def write_ohlcv(df: pl.DataFrame, pair: str,
                data_dir: Path = config.DATA_DIR) -> None:
    df = df.sort("date")
    ohlcv_dir = data_dir / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(ohlcv_dir / f"{pair}.parquet")

    entry = pl.DataFrame({
        "pair": [pair],
        "first_date": [df["date"][0]],
        "last_date": [df["date"][-1]],
        "rows": [df.height],
        "downloaded_at": [dt.datetime.now()],
    }, schema=MANIFEST_SCHEMA)
    manifest = read_manifest(data_dir).filter(pl.col("pair") != pair)
    path = data_dir / "manifest.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([manifest, entry]).write_parquet(path)


def read_ohlcv(pair: str, data_dir: Path = config.DATA_DIR) -> pl.DataFrame:
    path = data_dir / "ohlcv" / f"{pair}.parquet"
    if not path.exists():
        raise KeyError(f"{pair}: not found at {path}")
    return pl.read_parquet(path)
