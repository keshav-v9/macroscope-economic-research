import json

import pandas as pd

from src.ingest import export_matlab_input


def test_matlab_input_export_is_validated_and_language_neutral(tmp_path):
    csv_path, metadata_path = export_matlab_input(tmp_path)
    frame = pd.read_csv(csv_path)
    metadata = json.loads(metadata_path.read_text())

    assert len(frame) == 203
    assert frame["quarter_label"].iloc[0] == "1959Q1"
    assert frame["quarter_label"].iloc[-1] == "2009Q3"
    assert frame["quarter_index"].tolist() == list(range(203))
    assert {"realgdp", "infl", "tbilrate", "unemp"}.issubset(frame.columns)
    assert metadata["source_observations"] == 203
    assert metadata["dataset"] == "statsmodels.datasets.macrodata"
