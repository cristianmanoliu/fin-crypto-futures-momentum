"""Kraken futures API client. Rate limit: 1 req/sec."""
import datetime as _dt
import logging
import time

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://futures.kraken.com"
RATE_LIMIT_S = 1.1
_MAX_RETRIES = 5
_RETRY_BACKOFF = (10, 30, 60, 120, 300)


class KrakenError(Exception):
    pass


def _get_with_retry(url: str, **kwargs) -> httpx.Response:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = httpx.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            log.warning("retry %d/%d in %ds: %s", attempt + 1, _MAX_RETRIES, wait, exc)
            time.sleep(wait)
    return httpx.get(url, **kwargs)


def get_instruments() -> dict[str, dict]:
    """Fetch all USD-quoted tradeable perpetual futures.
    Returns {altname: {futures_symbol, base, openingDate}}."""
    resp = _get_with_retry(f"{BASE_URL}/derivatives/api/v3/instruments", timeout=30)
    data = resp.json()
    if data.get("result") != "success":
        raise KrakenError(str(data))
    out = {}
    for inst in data["instruments"]:
        sym = inst["symbol"]
        if not sym.startswith("PF_") or inst.get("quote") != "USD":
            continue
        if not inst.get("tradeable", False):
            continue
        base = inst.get("base", "")
        altname = f"{base}USD"
        out[altname] = {
            "futures_symbol": sym,
            "base": base,
            "openingDate": inst.get("openingDate", ""),
        }
    return out


def get_ohlc(futures_symbol: str) -> list[dict]:
    """Full daily OHLCV history for a perpetual futures instrument."""
    resp = _get_with_retry(
        f"{BASE_URL}/api/charts/v1/trade/{futures_symbol}/1d", timeout=30)
    data = resp.json()
    rows = []
    for c in data.get("candles", []):
        rows.append({
            "date": _dt.date.fromtimestamp(int(c["time"]) // 1000),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["volume"]),
        })
    return rows
