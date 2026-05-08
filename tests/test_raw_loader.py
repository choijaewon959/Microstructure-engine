from pathlib import Path

import pandas as pd
import pytest

from microdash.raw_loader import RawDataLoadError, load_prices, load_quotes, load_trades


def write_parquet(path: Path, data: dict[str, list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path)
    return path


def test_load_prices_normalizes_timestamp_and_sorts(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "GS_1s_2025-01-02.parquet",
        {
            "ts": [pd.Timestamp("2025-01-02 09:30:02"), pd.Timestamp("2025-01-02 09:30:01")],
            "symbol": ["GS", "GS"],
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.5, 99.5],
            "close": [101.5, 100.5],
            "volume": [20.0, 10.0],
            "vwap": [101.3, 100.2],
            "n_trades": [2.0, 1.0],
        },
    )

    loaded = load_prices(path)

    assert list(loaded["timestamp"]) == [
        pd.Timestamp("2025-01-02 09:30:01"),
        pd.Timestamp("2025-01-02 09:30:02"),
    ]
    assert "ts" in loaded.columns


def test_load_quotes_validates_required_columns(tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path / "GS_quotes_1s_2025-01-02.parquet",
        {
            "ts": [pd.Timestamp("2025-01-02 09:30:00")],
            "symbol": ["GS"],
            "bid_price": [99.0],
            "ask_price": [101.0],
            "bid_size": [100.0],
            "ask_size": [200.0],
            "mid": [100.0],
        },
    )

    with pytest.raises(RawDataLoadError, match="missing required columns"):
        load_quotes(path)


def test_load_trades_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "GS_trades_1s_2025-01-02.parquet"
    pd.DataFrame(columns=["ts", "symbol", "last", "volume", "n_trades", "vwap"]).to_parquet(path)

    with pytest.raises(RawDataLoadError, match="empty"):
        load_trades(path)


def test_load_trades_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_trades("missing.parquet")
