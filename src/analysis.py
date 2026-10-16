"""Command-line entry point for the complete MacroScope pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from .data import load_macrodata
from .evaluate import primary_metrics, robustness_metrics
from .features import complete_model_frame, build_model_frame
from .model import ForecastConfig, expanding_forecasts, fit_full_sample, regression_table
from .plots import plot_coefficients, plot_forecasts, plot_indicators

LOGGER = logging.getLogger("macroscope")


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def write_research_brief(path: Path, metrics: dict, sample_size: int, robust) -> None:
    m = metrics["model"]
    b = metrics["no_change_baseline"]
    imp = metrics["improvement_percent"]
    rows = "\n".join(
        f"| {r.model} | {r.mae:.4f} | {r.rmse:.4f} | {int(r.n_forecasts)} |"
        for r in robust.itertuples()
    )
    text = f"""# MacroScope Research Brief

## Executive summary

A simple four-variable linear model was tested as a forecasting tool for the change in U.S. unemployment four quarters ahead. In {metrics['evaluation_observations']} real-time-style historical forecasts, its RMSE was {_fmt(m['rmse'])} percentage points versus {_fmt(b['rmse'])} for predicting no change, an improvement of {_fmt(imp['rmse'])}%. This is a predictive comparison, not evidence that any predictor causes unemployment.

## Abstract and research question

How accurately can current unemployment, inflation, the 3-month Treasury bill rate, and real GDP growth predict the change in unemployment four quarters ahead? MacroScope estimates a full-sample descriptive OLS model and evaluates genuine pseudo-out-of-sample forecasts against transparent benchmarks.

## Data source and provenance

The bundled `statsmodels.datasets.macrodata` release contains 203 quarterly observations from 1959Q1 through 2009Q3. Statsmodels documents real GDP as billions of chained 2005 dollars at a seasonally adjusted annual rate; unemployment as a seasonally adjusted percent; inflation as `400 × ln(CPI_t/CPI_t-1)`; and the Treasury bill field as the quarterly monthly average of the monthly 3-month secondary-market rate. The source was compiled from FRED and BLS data accessed in 2009. See the [statsmodels documentation](https://www.statsmodels.org/stable/datasets/generated/macrodata.html).

## Features and target

At forecast origin *t*, predictors are unemployment (%), inflation (annualized %), the 3-month Treasury bill rate (%), and quarter-over-quarter real GDP log growth, `100 × [ln(realgdp_t) − ln(realgdp_t−1)]`. The target is `unemployment_t+4 − unemployment_t` in percentage points. Removing the first GDP-growth row and final four unavailable labels yields {sample_size} complete modeling rows.

## Method and inference

Full-sample multivariate OLS includes an intercept. Reported standard errors, test statistics, p-values, and 95% intervals use Newey–West/HAC covariance with four lags. These statistics describe associations in a revised historical dataset; they are not causal monetary-policy estimates.

## Validation protocol

The headline evaluation uses one expanding-window forecast per origin from {metrics['first_origin']} through {metrics['last_origin']} (targets {metrics['first_target']}–{metrics['last_target']}). Each fit requires at least 80 labeled observations and includes only row *s* when its label date `s+4` is no later than forecast origin *t*. Predictors dated *t* are allowed; `unemployment_t+4` is never available to the fit. The no-change forecast is exactly zero. Ancillary checks use the historical mean of currently observable changes and an 80-quarter rolling OLS window.

## Results

### Headline expanding-window comparison

| Metric | OLS model | No-change baseline | Improvement |
|---|---:|---:|---:|
| MAE (pp) | {_fmt(m['mae'])} | {_fmt(b['mae'])} | {_fmt(imp['mae'])}% |
| RMSE (pp) | {_fmt(m['rmse'])} | {_fmt(b['rmse'])} | {_fmt(imp['rmse'])}% |

### Robustness checks

| Forecast | MAE (pp) | RMSE (pp) | N |
|---|---:|---:|---:|
{rows}

The expanding model's relative error is the principal result. Historical-mean and rolling-window figures are ancillary specification checks, not separately tuned models.

## Interpretation and limitations

The experiment measures predictive accuracy within this historical sample. It does not identify a causal effect: the Treasury bill rate is endogenous to economic conditions, relationships can change, and predictors are correlated. The data are old, revised rather than vintage-real-time releases, and end in the Great Recession. Four-quarter targets overlap, inducing serial dependence in forecast errors. Model selection and the fixed initial window also create design sensitivity. Accordingly, the results should not be described as real-time policy impact or statistical superiority.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.analysis
python -m pytest -q
```
"""
    path.write_text(text, encoding="utf-8")


def update_readme_results(path: Path, metrics: dict, robust) -> None:
    """Replace the bounded README result block with calculated values."""
    if not path.exists():
        return
    m = metrics["model"]
    b = metrics["no_change_baseline"]
    imp = metrics["improvement_percent"]
    historical = robust.loc[robust["model"] == "historical_mean"].iloc[0]
    rolling = robust.loc[robust["model"] == "rolling_80q_ols"].iloc[0]
    replacement = f"""<!-- RESULTS_START -->
Using {metrics['evaluation_observations']} pseudo-out-of-sample forecasts (origins {metrics['first_origin']}–{metrics['last_origin']}), the expanding-window OLS achieved **{_fmt(m['mae'])} pp MAE** and **{_fmt(m['rmse'])} pp RMSE**. The no-change benchmark produced **{_fmt(b['mae'])} pp MAE** and **{_fmt(b['rmse'])} pp RMSE**, so the model reduced MAE by **{_fmt(imp['mae'])}%** and RMSE by **{_fmt(imp['rmse'])}%**. These values come from the checked-in [`metrics.json`](outputs/metrics.json), generated by the pipeline—not from hardcoded assumptions.

As ancillary robustness checks, the historical-mean forecast had {_fmt(historical['mae'])} pp MAE / {_fmt(historical['rmse'])} pp RMSE, while the 80-quarter rolling OLS had {_fmt(rolling['mae'])} pp MAE / {_fmt(rolling['rmse'])} pp RMSE. The expanding-window comparison remains the pre-specified headline result.
<!-- RESULTS_END -->"""
    current = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->",
        replacement,
        current,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("README result markers are missing or duplicated")
    path.write_text(updated, encoding="utf-8")


def run_pipeline(output_dir: Path, min_train_size: int = 80, rolling_window: int = 80) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_macrodata()
    feature_frame = build_model_frame(raw)
    complete = complete_model_frame(feature_frame)
    result = fit_full_sample(complete)
    table = regression_table(result)
    predictions = expanding_forecasts(complete, ForecastConfig(min_train_size, rolling_window))
    metrics = primary_metrics(predictions)
    metrics.update({
        "source_observations": len(raw),
        "complete_model_observations": len(complete),
        "forecast_horizon_quarters": 4,
        "minimum_training_observations": min_train_size,
        "rolling_window_quarters": rolling_window,
    })
    robust = robustness_metrics(predictions)

    table.to_csv(output_dir / "regression_table.csv", index=False)
    predictions.to_csv(output_dir / "forecast_predictions.csv", index=False)
    robust.to_csv(output_dir / "robustness_metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_indicators(feature_frame, output_dir / "macro_indicators.png")
    plot_forecasts(predictions, output_dir / "forecast_comparison.png")
    plot_coefficients(table, output_dir / "regression_coefficients.png")
    write_research_brief(output_dir / "RESEARCH_BRIEF.md", metrics, len(complete), robust)
    if output_dir.resolve() == Path("outputs").resolve():
        update_readme_results(Path("README.md"), metrics, robust)
    LOGGER.info("Wrote analysis artifacts to %s", output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--min-train-size", type=int, default=80)
    parser.add_argument("--rolling-window", type=int, default=80)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_pipeline(args.output_dir, args.min_train_size, args.rolling_window)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
