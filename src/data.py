"""Load and validate the bundled statsmodels U.S. macro dataset."""

from __future__ import annotations

import pandas as pd
from statsmodels.datasets import macrodata

EXPECTED_ROWS = 203
REQUIRED_COLUMNS = {"year", "quarter", "realgdp", "infl", "tbilrate", "unemp"}


def load_macrodata() -> pd.DataFrame:
    """Return validated source data indexed by a quarterly PeriodIndex.

    The dataset ships inside statsmodels, so this function requires no network
    access and remains reproducible from a fresh environment.
    """
    frame = macrodata.load_pandas().data.copy()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Source data are missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Source data contain unexpected missing values")

    years = frame["year"].astype(int)
    quarters = frame["quarter"].astype(int)
    if not quarters.between(1, 4).all():
        raise ValueError("Quarter values must be integers from 1 through 4")
    period = pd.PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q")
    period.name = "quarter"
    if period.has_duplicates or not period.is_monotonic_increasing:
        raise ValueError("Source quarters must be unique and sorted")
    expected = pd.period_range(period[0], period[-1], freq="Q")
    if not period.equals(expected):
        raise ValueError("Source quarters must be contiguous")

    frame.index = period
    return frame
