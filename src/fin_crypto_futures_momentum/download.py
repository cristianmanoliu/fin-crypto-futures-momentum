"""Download all perpetual futures OHLCV from Kraken.
Run: uv run python -m fin_crypto_futures_momentum.download"""
import logging
import sys
import time

import polars as pl

from fin_crypto_futures_momentum import config
from fin_crypto_futures_momentum.kraken_client import (
    KrakenError,
    RATE_LIMIT_S,
    get_instruments,
    get_ohlc,
)
from fin_crypto_futures_momentum.store import write_ohlcv

log = logging.getLogger(__name__)


def download_all() -> int:
    instruments = get_instruments()
    log.info("%d USD perpetuals found", len(instruments))
    errors = 0
    for i, (altname, info) in enumerate(sorted(instruments.items()), 1):
        log.info("[%d/%d] %s (%s)", i, len(instruments), altname,
                 info["futures_symbol"])
        try:
            rows = get_ohlc(info["futures_symbol"])
            if not rows:
                log.warning("%s: no candles", altname)
                continue
            df = pl.DataFrame(rows)
            write_ohlcv(df, altname, data_dir=config.DATA_DIR)
            log.info("%s: %d days", altname, df.height)
        except Exception:
            log.exception("FAILED: %s", altname)
            errors += 1
        time.sleep(RATE_LIMIT_S)
    return errors


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    errors = download_all()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
