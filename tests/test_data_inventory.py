from pathlib import Path

import pandas as pd
import pytest

from microdash.data_inventory import (
    missing_quote_trade_pairs,
    parse_raw_filename,
    scan_raw_inventory,
    summarize_coverage,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_parse_price_filename() -> None:
    record = parse_raw_filename("data/raw/prices/GS_1s_2025-01-02.parquet")

    assert record.data_type == "prices"
    assert record.symbol == "GS"
    assert record.frequency == "1s"
    assert record.date == pd.Timestamp("2025-01-02")


def test_parse_quote_filename_rejects_wrong_folder_label() -> None:
    with pytest.raises(ValueError, match="does not match folder"):
        parse_raw_filename("data/raw/trades/GS_quotes_1s_2025-01-02.parquet", data_type="trades")


def test_scan_raw_inventory_and_summarize_coverage(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    touch(raw / "prices" / "GS_1s_2025-01-02.parquet")
    touch(raw / "quotes" / "GS_quotes_1s_2025-01-02.parquet")
    touch(raw / "quotes" / "GS_quotes_1s_2025-01-03.parquet")
    touch(raw / "trades" / "GS_trades_1s_2025-01-02.parquet")

    inventory = scan_raw_inventory(raw)

    assert set(inventory["data_type"]) == {"prices", "quotes", "trades"}
    assert len(inventory) == 4

    summary = summarize_coverage(inventory)
    quotes = summary[(summary["data_type"] == "quotes") & (summary["symbol"] == "GS")].iloc[0]
    assert quotes["file_count"] == 2
    assert quotes["start_date"] == pd.Timestamp("2025-01-02")
    assert quotes["end_date"] == pd.Timestamp("2025-01-03")


def test_missing_quote_trade_pairs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    touch(raw / "quotes" / "GS_quotes_1s_2025-01-02.parquet")
    touch(raw / "trades" / "GS_trades_1s_2025-01-02.parquet")
    touch(raw / "quotes" / "GS_quotes_1s_2025-01-03.parquet")
    touch(raw / "trades" / "MS_trades_1s_2025-01-04.parquet")

    inventory = scan_raw_inventory(raw)
    missing = missing_quote_trade_pairs(inventory)

    assert len(missing) == 2
    assert {
        (row.symbol, row.date.strftime("%Y-%m-%d"), row.has_quotes, row.has_trades)
        for row in missing.itertuples(index=False)
    } == {
        ("GS", "2025-01-03", True, False),
        ("MS", "2025-01-04", False, True),
    }

