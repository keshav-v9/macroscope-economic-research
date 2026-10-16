"""Feature engineering with explicit forecast-origin/target alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 4
PREDICTORS = ["unemp", "infl", "tbilrate", "gdp_growth"]
TARGET = "unemp_change_4q"


def build_model_frame(raw: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Create origin-t predictors and the t+h unemployment-change target.

    GDP growth uses only t and t-1 values. ``shift(-horizon)`` is used solely
    to construct the evaluation label; model training later filters labels by
    their target date so that no unavailable outcome enters a fit.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    frame = pd.DataFrame(index=raw.index.copy())
    frame["unemp"] = raw["unemp"]
    frame["infl"] = raw["infl"]
    frame["tbilrate"] = raw["tbilrate"]
    frame["gdp_growth"] = 100.0 * np.log(raw["realgdp"] / raw["realgdp"].shift(1))
    frame["target_quarter"] = frame.index + horizon
    frame[TARGET] = raw["unemp"].shift(-horizon) - raw["unemp"]
    frame.index.name = "origin_quarter"
    return frame


def complete_model_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically retain rows with complete predictors and labels."""
    return feature_frame.dropna(subset=PREDICTORS + [TARGET]).copy()
