"""Utilities for inventorying local raw market data files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAW_DATA_TYPES = ("prices", "quotes", "trades")

_PRICE_RE = re.compile(
    r"^(?P<symbol>[A-Z]+)_(?P<frequency>\d+s)_(?P<date>\d{4}-\d{2}-\d{2})\.parquet$"
)
_QUOTE_TRADE_RE = re.compile(
    r"^(?P<symbol>[A-Z]+)_(?P<label>quotes|trades)_(?P<frequency>\d+s)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})\.parquet$"
)


@dataclass(frozen=True)
class RawFileRecord:
    """Metadata parsed from a raw parquet file path."""

    path: Path
    data_type: str
    symbol: str
    frequency: str
    date: pd.Timestamp


def parse_raw_filename(path: str | Path, data_type: str | None = None) -> RawFileRecord:
    """Parse symbol, data type, frequency, and date from a raw parquet filename."""

    file_path = Path(path)
    inferred_type = data_type or file_path.parent.name

    if inferred_type not in RAW_DATA_TYPES:
        raise ValueError(f"Unsupported raw data type: {inferred_type!r}")

    if inferred_type == "prices":
        match = _PRICE_RE.match(file_path.name)
    else:
        match = _QUOTE_TRADE_RE.match(file_path.name)

    if match is None:
        raise ValueError(
            f"Raw filename does not match {inferred_type} convention: {file_path.name}"
        )

    fields = match.groupdict()
    if inferred_type in {"quotes", "trades"} and fields["label"] != inferred_type:
        raise ValueError(
            f"Filename label {fields['label']!r} does not match folder {inferred_type!r}"
        )

    return RawFileRecord(
        path=file_path,
        data_type=inferred_type,
        symbol=fields["symbol"],
        frequency=fields["frequency"],
        date=pd.Timestamp(fields["date"]),
    )


def scan_raw_inventory(raw_root: str | Path = "data/raw") -> pd.DataFrame:
    """Scan raw parquet folders and return one metadata row per recognized file."""

    root = Path(raw_root)
    records: list[RawFileRecord] = []

    for data_type in RAW_DATA_TYPES:
        folder = root / data_type
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.parquet")):
            records.append(parse_raw_filename(file_path, data_type=data_type))

    rows = [
        {
            "path": str(record.path),
            "data_type": record.data_type,
            "symbol": record.symbol,
            "frequency": record.frequency,
            "date": record.date,
        }
        for record in records
    ]
    return pd.DataFrame(rows, columns=["path", "data_type", "symbol", "frequency", "date"])


def summarize_coverage(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarize file counts and date coverage by data type and symbol."""

    if inventory.empty:
        return pd.DataFrame(
            columns=["data_type", "symbol", "file_count", "start_date", "end_date", "date_count"]
        )

    summary = (
        inventory.groupby(["data_type", "symbol"], as_index=False)
        .agg(
            file_count=("path", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            date_count=("date", "nunique"),
        )
        .sort_values(["data_type", "symbol"])
        .reset_index(drop=True)
    )
    return summary


def missing_quote_trade_pairs(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return symbol/date/frequency rows where either quote or trade data is missing."""

    if inventory.empty:
        return pd.DataFrame(columns=["symbol", "frequency", "date", "has_quotes", "has_trades"])

    quote_trades = inventory[inventory["data_type"].isin(["quotes", "trades"])]
    if quote_trades.empty:
        return pd.DataFrame(columns=["symbol", "frequency", "date", "has_quotes", "has_trades"])

    availability = (
        quote_trades.assign(present=True)
        .pivot_table(
            index=["symbol", "frequency", "date"],
            columns="data_type",
            values="present",
            aggfunc="any",
            fill_value=False,
        )
        .reset_index()
    )

    for column in ["quotes", "trades"]:
        if column not in availability:
            availability[column] = False

    missing = availability.loc[~(availability["quotes"] & availability["trades"])].copy()
    missing = missing.rename(columns={"quotes": "has_quotes", "trades": "has_trades"})
    return missing[["symbol", "frequency", "date", "has_quotes", "has_trades"]].sort_values(
        ["symbol", "date", "frequency"]
    )
