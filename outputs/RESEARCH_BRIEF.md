# MacroScope Research Brief

## Executive summary

A simple four-variable linear model was tested as a forecasting tool for the change in U.S. unemployment four quarters ahead. In 115 real-time-style historical forecasts, its RMSE was 0.9238 percentage points versus 1.0515 for predicting no change, an improvement of 12.1438%. This is a predictive comparison, not evidence that any predictor causes unemployment.

## Abstract and research question

How accurately can current unemployment, inflation, the 3-month Treasury bill rate, and real GDP growth predict the change in unemployment four quarters ahead? MacroScope estimates a full-sample descriptive OLS model and evaluates genuine pseudo-out-of-sample forecasts against transparent benchmarks.

## Data source and provenance

The bundled `statsmodels.datasets.macrodata` release contains 203 quarterly observations from 1959Q1 through 2009Q3. Statsmodels documents real GDP as billions of chained 2005 dollars at a seasonally adjusted annual rate; unemployment as a seasonally adjusted percent; inflation as `400 × ln(CPI_t/CPI_t-1)`; and the Treasury bill field as the quarterly monthly average of the monthly 3-month secondary-market rate. The source was compiled from FRED and BLS data accessed in 2009. See the [statsmodels documentation](https://www.statsmodels.org/stable/datasets/generated/macrodata.html).

## Features and target

At forecast origin *t*, predictors are unemployment (%), inflation (annualized %), the 3-month Treasury bill rate (%), and quarter-over-quarter real GDP log growth, `100 × [ln(realgdp_t) − ln(realgdp_t−1)]`. The target is `unemployment_t+4 − unemployment_t` in percentage points. Removing the first GDP-growth row and final four unavailable labels yields 198 complete modeling rows.

## Method and inference

Full-sample multivariate OLS includes an intercept. Reported standard errors, test statistics, p-values, and 95% intervals use Newey–West/HAC covariance with four lags. These statistics describe associations in a revised historical dataset; they are not causal monetary-policy estimates.

## Validation protocol

The headline evaluation uses one expanding-window forecast per origin from 1980Q1 through 2008Q3 (targets 1981Q1–2009Q3). Each fit requires at least 80 labeled observations and includes only row *s* when its label date `s+4` is no later than forecast origin *t*. Predictors dated *t* are allowed; `unemployment_t+4` is never available to the fit. The no-change forecast is exactly zero. Ancillary checks use the historical mean of currently observable changes and an 80-quarter rolling OLS window.

## Results

### Headline expanding-window comparison

| Metric | OLS model | No-change baseline | Improvement |
|---|---:|---:|---:|
| MAE (pp) | 0.6447 | 0.7487 | 13.8932% |
| RMSE (pp) | 0.9238 | 1.0515 | 12.1438% |

### Robustness checks

| Forecast | MAE (pp) | RMSE (pp) | N |
|---|---:|---:|---:|
| expanding_ols | 0.6447 | 0.9238 | 115 |
| no_change | 0.7487 | 1.0515 | 115 |
| historical_mean | 0.7698 | 1.0718 | 115 |
| rolling_80q_ols | 0.6370 | 0.8988 | 115 |

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
