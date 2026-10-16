import numpy as np
import pandas as pd

from src.data import load_macrodata
from src.features import PREDICTORS, TARGET, build_model_frame, complete_model_frame


def test_macrodata_is_complete_sorted_quarterly_source():
    raw = load_macrodata()
    assert len(raw) == 203
    assert raw.index[0] == pd.Period("1959Q1", freq="Q")
    assert raw.index[-1] == pd.Period("2009Q3", freq="Q")
    assert raw.index.is_monotonic_increasing
    assert not raw.index.has_duplicates
    assert raw.index.equals(pd.period_range("1959Q1", "2009Q3", freq="Q", name="quarter"))


def test_gdp_growth_uses_only_current_and_previous_quarter():
    raw = load_macrodata()
    frame = build_model_frame(raw)
    expected = 100 * np.log(raw.loc["1959Q2", "realgdp"] / raw.loc["1959Q1", "realgdp"])
    assert np.isnan(frame.loc["1959Q1", "gdp_growth"])
    assert np.isclose(frame.loc["1959Q2", "gdp_growth"], expected)


def test_four_quarter_target_alignment_and_unavailable_tail():
    raw = load_macrodata()
    frame = build_model_frame(raw)
    origin = pd.Period("1970Q2", freq="Q")
    expected = raw.loc[origin + 4, "unemp"] - raw.loc[origin, "unemp"]
    assert frame.loc[origin, "target_quarter"] == origin + 4
    assert np.isclose(frame.loc[origin, TARGET], expected)
    assert frame[TARGET].tail(4).isna().all()
    assert len(complete_model_frame(frame)) == 198
    assert not complete_model_frame(frame)[PREDICTORS + [TARGET]].isna().any().any()
