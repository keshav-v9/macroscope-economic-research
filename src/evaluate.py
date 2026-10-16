"""Forecast error metrics and robustness summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_metrics(y_true, prediction) -> dict[str, float]:
    """Return MAE and RMSE for equal-length numeric arrays."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Inputs must be non-empty and have the same shape")
    errors = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def improvement_percent(model_error: float, baseline_error: float) -> float:
    if baseline_error == 0:
        raise ValueError("Baseline error must be nonzero")
    return 100.0 * (baseline_error - model_error) / baseline_error


def primary_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    model = error_metrics(predictions["y_true"], predictions["model_prediction"])
    baseline = error_metrics(predictions["y_true"], predictions["baseline_prediction"])
    return {
        "evaluation_observations": int(len(predictions)),
        "first_origin": str(predictions["origin_quarter"].iloc[0]),
        "last_origin": str(predictions["origin_quarter"].iloc[-1]),
        "first_target": str(predictions["target_quarter"].iloc[0]),
        "last_target": str(predictions["target_quarter"].iloc[-1]),
        "model": model,
        "no_change_baseline": baseline,
        "improvement_percent": {
            "mae": improvement_percent(model["mae"], baseline["mae"]),
            "rmse": improvement_percent(model["rmse"], baseline["rmse"]),
        },
    }


def robustness_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, column in (
        ("expanding_ols", "model_prediction"),
        ("no_change", "baseline_prediction"),
        ("historical_mean", "historical_mean_prediction"),
        ("rolling_80q_ols", "rolling_prediction"),
    ):
        metrics = error_metrics(predictions["y_true"], predictions[column])
        rows.append({"model": name, **metrics, "n_forecasts": len(predictions)})
    return pd.DataFrame(rows)
