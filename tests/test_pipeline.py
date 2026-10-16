import json

import pandas as pd
from matplotlib import image as mpimg

from src.analysis import run_pipeline


def test_pipeline_writes_expected_artifacts_and_fields(tmp_path):
    metrics = run_pipeline(tmp_path)
    expected = {
        "regression_table.csv",
        "forecast_predictions.csv",
        "metrics.json",
        "robustness_metrics.csv",
        "macro_indicators.png",
        "forecast_comparison.png",
        "regression_coefficients.png",
        "RESEARCH_BRIEF.md",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}

    table = pd.read_csv(tmp_path / "regression_table.csv")
    assert {
        "term", "coefficient", "hac_std_error", "t_statistic", "p_value",
        "ci_95_lower", "ci_95_upper",
    } == set(table.columns)

    predictions = pd.read_csv(tmp_path / "forecast_predictions.csv")
    required = {
        "origin_quarter", "target_quarter", "y_true", "model_prediction",
        "baseline_prediction", "model_error", "baseline_error", "train_size",
    }
    assert required.issubset(predictions.columns)
    assert (predictions["baseline_prediction"] == 0).all()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    for filename in ("macro_indicators.png", "forecast_comparison.png", "regression_coefficients.png"):
        image = mpimg.imread(tmp_path / filename)
        assert image.shape[0] > 100 and image.shape[1] > 100
