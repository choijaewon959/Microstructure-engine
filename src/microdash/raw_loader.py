"""Raw parquet loading and schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from microdash.data_inventory import RAW_DATA_TYPES


TIMESTAMP_COLUMN: Final = "timestamp"
SOURCE_TIMESTAMP_COLUMN: Final = "ts"

REQUIRED_COLUMNS: Final[dict[str, set[str]]] = {
    "prices": {"ts", "symbol", "open", "high", "low", "close", "volume", "vwap", "n_trades"},
    "quotes": {
        "ts",
        "symbol",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "mid",
        "spread",
    },
    "trades": {"ts", "symbol", "last", "volume", "n_trades", "vwap"},
}


class RawDataLoadError(ValueError):
    """Raised when a raw parquet file cannot satisfy the expected data contract."""


def validate_raw_schema(df: pd.DataFrame, data_type: str) -> None:
    """Validate that a raw dataframe has the required columns for its data type."""

    if data_type not in RAW_DATA_TYPES:
        raise RawDataLoadError(f"Unsupported raw data type: {data_type!r}")

    if df.empty:
        raise RawDataLoadError(f"{data_type} dataframe is empty")

    missing = sorted(REQUIRED_COLUMNS[data_type].difference(df.columns))
    if missing:
        raise RawDataLoadError(f"{data_type} dataframe is missing required columns: {missing}")


def normalize_raw_dataframe(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Validate, add canonical columns, and sort a raw dataframe."""

    validate_raw_schema(df, data_type)

    normalized = df.copy()
    normalized[TIMESTAMP_COLUMN] = pd.to_datetime(normalized[SOURCE_TIMESTAMP_COLUMN])
    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized = normalized.sort_values(["symbol", TIMESTAMP_COLUMN]).reset_index(drop=True)
    return normalized


def load_raw_parquet(path: str | Path, data_type: str) -> pd.DataFrame:
    """Load one raw parquet file and normalize timestamp and symbol columns."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw parquet file does not exist: {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:  # pragma: no cover - pandas/pyarrow provide detailed exception types.
        raise RawDataLoadError(f"Failed to read raw parquet file: {file_path}") from exc

    return normalize_raw_dataframe(df, data_type)


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load and normalize one raw price parquet file."""

    return load_raw_parquet(path, "prices")


def load_quotes(path: str | Path) -> pd.DataFrame:
    """Load and normalize one raw quote parquet file."""

    return load_raw_parquet(path, "quotes")


def load_trades(path: str | Path) -> pd.DataFrame:
    """Load and normalize one raw trade parquet file."""

    return load_raw_parquet(path, "trades")
