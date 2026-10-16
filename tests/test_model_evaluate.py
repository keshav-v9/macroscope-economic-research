import numpy as np

from src.data import load_macrodata
from src.evaluate import error_metrics, improvement_percent, primary_metrics
from src.features import build_model_frame, complete_model_frame
from src.model import ForecastConfig, eligible_training_rows, expanding_forecasts


def _frame():
    return complete_model_frame(build_model_frame(load_macrodata()))


def test_training_window_never_contains_a_future_label():
    frame = _frame()
    origin = frame.index[100]
    train = eligible_training_rows(frame, origin)
    assert (train["target_quarter"] <= origin).all()
    assert train["target_quarter"].max() == origin
    assert train.index.max() == origin - 4
    assert not frame.loc[[origin]].index.isin(train.index).any()


def test_forecast_audit_fields_prove_temporal_alignment():
    predictions = expanding_forecasts(_frame(), ForecastConfig(min_train_size=80, rolling_window=80))
    assert len(predictions) == 115
    assert (predictions["baseline_prediction"] == 0.0).all()
    for row in predictions.itertuples():
        assert row.latest_known_target <= row.origin_quarter
        assert row.train_end_origin < row.origin_quarter
        assert row.train_size >= 80


def test_metrics_match_hand_calculation():
    metrics = error_metrics([1.0, -1.0, 3.0], [0.0, 0.0, 1.0])
    assert np.isclose(metrics["mae"], 4 / 3)
    assert np.isclose(metrics["rmse"], np.sqrt(2.0))
    assert np.isclose(improvement_percent(0.8, 1.0), 20.0)


def test_primary_metrics_use_exact_zero_baseline():
    predictions = expanding_forecasts(_frame())
    metrics = primary_metrics(predictions)
    manual = error_metrics(predictions["y_true"], np.zeros(len(predictions)))
    assert metrics["no_change_baseline"] == manual
