"""Futures momentum config. All constants pre-registered;
verdict runs must not override them ad hoc."""
import datetime as dt
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

NAV_DEFAULT = 100_000.0

TAKER_BP = 5.0

SLIP_LEVELS_BP = (0.0, 5.0, 10.0, 20.0)

DECISION_SLIP_BP = 10.0

# Perps launched 2022-03-22; use 2022-07 to let pairs accumulate history
FORM_START = dt.date(2022, 7, 3)
TRAIN_END = dt.date(2024, 6, 30)
FORM_END = dt.date(2026, 9, 14)


def cost_frac(notional: float) -> float:
    return TAKER_BP / 1e4
