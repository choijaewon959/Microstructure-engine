"""Quote/trade alignment helpers for microstructure features."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from microdash.raw_loader import TIMESTAMP_COLUMN


SESSION_DATE_COLUMN = "session_date"


class AlignmentError(ValueError):
    """Raised when quote/trade data cannot be aligned safely."""


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise AlignmentError(f"{name} dataframe is missing required columns: {missing}")


def _with_session_date(df: pd.DataFrame) -> pd.DataFrame:
    with_date = df.copy()
    with_date[SESSION_DATE_COLUMN] = pd.to_datetime(with_date[TIMESTAMP_COLUMN]).dt.date
    return with_date


def prepare_quotes_for_alignment(quotes: pd.DataFrame) -> pd.DataFrame:
    """Sort quotes and collapse duplicate symbol/timestamp rows to the last quote."""

    _validate_columns(quotes, {"symbol", TIMESTAMP_COLUMN}, "quotes")
    prepared = _with_session_date(quotes)
    prepared = prepared.sort_values(["symbol", SESSION_DATE_COLUMN, TIMESTAMP_COLUMN])
    return prepared.drop_duplicates(
        subset=["symbol", SESSION_DATE_COLUMN, TIMESTAMP_COLUMN], keep="last"
    ).reset_index(drop=True)


def align_trades_to_quotes(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta | None = "1s",
    direction: Literal["backward", "forward", "nearest"] = "backward",
) -> pd.DataFrame:
    """Align each trade row to a quote from the same symbol and session date.

    The default uses the latest quote at or before the trade timestamp within one second.
    This matches a conservative microstructure convention for assigning prevailing quotes
    before computing trade-aware metrics such as effective spread.
    """

    _validate_columns(trades, {"symbol", TIMESTAMP_COLUMN}, "trades")
    _validate_columns(quotes, {"symbol", TIMESTAMP_COLUMN}, "quotes")

    if trades.empty:
        return trades.copy()
    if quotes.empty:
        result = trades.copy()
        result[SESSION_DATE_COLUMN] = pd.to_datetime(result[TIMESTAMP_COLUMN]).dt.date
        return result

    trade_rows = _with_session_date(trades).sort_values(
        ["symbol", SESSION_DATE_COLUMN, TIMESTAMP_COLUMN]
    )
    quote_rows = prepare_quotes_for_alignment(quotes)
    tolerance_delta = None if tolerance is None else pd.Timedelta(tolerance)

    aligned_groups: list[pd.DataFrame] = []
    group_columns = ["symbol", SESSION_DATE_COLUMN]

    for group_key, trade_group in trade_rows.groupby(group_columns, sort=False):
        symbol, session_date = group_key
        quote_group = quote_rows[
            (quote_rows["symbol"] == symbol) & (quote_rows[SESSION_DATE_COLUMN] == session_date)
        ]
        if quote_group.empty:
            aligned_groups.append(trade_group.reset_index(drop=True))
            continue

        aligned = pd.merge_asof(
            trade_group.sort_values(TIMESTAMP_COLUMN),
            quote_group.sort_values(TIMESTAMP_COLUMN),
            on=TIMESTAMP_COLUMN,
            direction=direction,
            tolerance=tolerance_delta,
            suffixes=("", "_quote"),
        )
        aligned_groups.append(aligned)

    if not aligned_groups:
        return trade_rows.reset_index(drop=True)

    aligned_all = pd.concat(aligned_groups, ignore_index=True)
    return aligned_all.sort_values(["symbol", SESSION_DATE_COLUMN, TIMESTAMP_COLUMN]).reset_index(
        drop=True
    )
