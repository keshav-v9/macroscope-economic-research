"""OLS estimation and temporally honest forecast generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from .features import PREDICTORS, TARGET


@dataclass(frozen=True)
class ForecastConfig:
    min_train_size: int = 80
    rolling_window: int = 80


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(frame[PREDICTORS], has_constant="add")


def fit_full_sample(frame: pd.DataFrame, maxlags: int = 4):
    """Fit full-sample OLS with Newey-West/HAC covariance."""
    return sm.OLS(frame[TARGET], _design(frame)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )


def regression_table(result) -> pd.DataFrame:
    """Return publication-ready HAC coefficient statistics."""
    confidence = result.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "hac_std_error": result.bse.values,
            "t_statistic": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_95_lower": confidence.iloc[:, 0].values,
            "ci_95_upper": confidence.iloc[:, 1].values,
        }
    )


def eligible_training_rows(frame: pd.DataFrame, origin: pd.Period) -> pd.DataFrame:
    """Rows whose outcome had been observed by ``origin`` (target <= origin)."""
    return frame.loc[frame["target_quarter"] <= origin]


def expanding_forecasts(
    frame: pd.DataFrame, config: ForecastConfig = ForecastConfig()
) -> pd.DataFrame:
    """Produce one-origin-at-a-time expanding and rolling OLS forecasts."""
    records: list[dict[str, object]] = []
    for origin, row in frame.iterrows():
        train = eligible_training_rows(frame, origin)
        if len(train) < config.min_train_size:
            continue

        expanding = sm.OLS(train[TARGET], _design(train)).fit()
        x_origin = _design(row.to_frame().T)
        # Preserve the training column order across statsmodels/pandas versions.
        x_origin = x_origin.reindex(columns=expanding.model.exog_names)
        prediction = float(expanding.predict(x_origin).iloc[0])

        rolling_train = train.iloc[-config.rolling_window :]
        rolling = sm.OLS(rolling_train[TARGET], _design(rolling_train)).fit()
        rolling_x = x_origin.reindex(columns=rolling.model.exog_names)
        rolling_prediction = float(rolling.predict(rolling_x).iloc[0])
        historical_mean = float(train[TARGET].mean())
        truth = float(row[TARGET])

        records.append(
            {
                "origin_quarter": str(origin),
                "target_quarter": str(row["target_quarter"]),
                "train_start_origin": str(train.index[0]),
                "train_end_origin": str(train.index[-1]),
                "latest_known_target": str(train["target_quarter"].max()),
                "train_size": len(train),
                "y_true": truth,
                "model_prediction": prediction,
                "baseline_prediction": 0.0,
                "historical_mean_prediction": historical_mean,
                "rolling_prediction": rolling_prediction,
                "model_error": truth - prediction,
                "baseline_error": truth,
                "historical_mean_error": truth - historical_mean,
                "rolling_error": truth - rolling_prediction,
            }
        )
    if not records:
        raise ValueError("No evaluation origins meet min_train_size")
    return pd.DataFrame.from_records(records)
