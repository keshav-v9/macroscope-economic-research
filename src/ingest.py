"""Validate statsmodels macrodata and export a language-neutral MATLAB input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import EXPECTED_ROWS, load_macrodata


def export_matlab_input(output_dir: Path = Path("data")) -> tuple[Path, Path]:
    """Write validated source observations and provenance metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_macrodata()
    exported = raw.reset_index(drop=True).copy()
    exported.insert(0, "quarter_label", raw.index.astype(str))
    exported.insert(1, "quarter_index", range(len(exported)))
    csv_path = output_dir / "macrodata.csv"
    exported.to_csv(csv_path, index=False)

    metadata = {
        "dataset": "statsmodels.datasets.macrodata",
        "source_observations": EXPECTED_ROWS,
        "first_quarter": str(raw.index[0]),
        "last_quarter": str(raw.index[-1]),
        "source_documentation": "https://www.statsmodels.org/stable/datasets/generated/macrodata.html",
        "units": {
            "realgdp": "Billions of chained 2005 USD, seasonally adjusted annual rate",
            "infl": "Annualized percent: 400 * ln(CPI_t / CPI_t-1)",
            "tbilrate": "Quarterly monthly average, 3-month Treasury bill secondary-market rate, percent",
            "unemp": "Seasonally adjusted percent",
        },
    }
    metadata_path = output_dir / "provenance.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    csv_path, metadata_path = export_matlab_input(args.output_dir)
    print(f"Wrote {csv_path} and {metadata_path}")


if __name__ == "__main__":
    main()
