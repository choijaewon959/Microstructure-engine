from pathlib import Path

import pandas as pd


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_sample_csv_files_are_commit_safe_and_readable() -> None:
    expected_files = {
        "GS_1s_2025-01-02_prices_sample.csv",
        "GS_1s_2025-01-02_quotes_sample.csv",
        "GS_1s_2025-01-02_trades_sample.csv",
    }

    actual_files = {path.name for path in SAMPLE_DIR.glob("*_sample.csv")}
    assert expected_files.issubset(actual_files)

    for file_name in expected_files:
        sample = pd.read_csv(SAMPLE_DIR / file_name)
        assert 0 < len(sample) <= 120
        assert "ts" in sample.columns
        assert "symbol" in sample.columns
