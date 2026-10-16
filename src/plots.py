"""GitHub-friendly research figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_indicators(feature_frame: pd.DataFrame, path: Path) -> None:
    specs = [
        ("unemp", "Unemployment rate", "Percent"),
        ("infl", "Inflation rate", "Annualized percent"),
        ("tbilrate", "3-month Treasury bill rate", "Percent"),
        ("gdp_growth", "Real GDP quarter-over-quarter log growth", "Percent"),
    ]
    x = feature_frame.index.to_timestamp(how="end")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (column, title, unit) in zip(axes.flat, specs):
        ax.plot(x, feature_frame[column], color="#176B87", linewidth=1.25)
        ax.axhline(0, color="#777777", linewidth=0.6, alpha=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.2)
    fig.suptitle("U.S. macroeconomic indicators, 1959Q1–2009Q3", fontweight="bold")
    fig.supxlabel("Quarter")
    fig.tight_layout()
    _save(fig, path)


def plot_forecasts(predictions: pd.DataFrame, path: Path) -> None:
    x = pd.PeriodIndex(predictions["target_quarter"], freq="Q").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, predictions["y_true"], label="Actual 4-quarter change", color="#111827", linewidth=1.8)
    ax.plot(x, predictions["model_prediction"], label="Expanding-window OLS", color="#D55E00", linewidth=1.4)
    ax.plot(x, predictions["baseline_prediction"], label="No-change baseline", color="#0072B2", linestyle="--")
    ax.set(title="Four-quarter-ahead unemployment-change forecasts", xlabel="Target quarter", ylabel="Percentage points")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, path)


def plot_coefficients(table: pd.DataFrame, path: Path) -> None:
    display = table.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        display["coefficient"], display["term"],
        xerr=[display["coefficient"] - display["ci_95_lower"], display["ci_95_upper"] - display["coefficient"]],
        fmt="o", color="#176B87", ecolor="#D55E00", capsize=4,
    )
    ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
    ax.set(title="OLS coefficients with 95% HAC confidence intervals", xlabel="Effect on 4-quarter unemployment change (percentage points)", ylabel="Predictor at forecast origin")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, path)
