"""Reusable market microstructure feature calculations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd
import numpy as np

BID_PRICE_COLUMN: Final = "bid_price"
ASK_PRICE_COLUMN: Final = "ask_price"
MID_PRICE_COLUMN: Final = "mid"
LEGACY_MID_PRICE_COLUMN: Final = "mid_price"
TIMESTAMP_COLUMN: Final = "timestamp"
QUOTED_SPREAD_COLUMN: Final = "quoted_spread"
RELATIVE_SPREAD_COLUMN: Final = "relative_spread"
BID_SIZE_COLUMN: Final = "bid_size"
ASK_SIZE_COLUMN: Final = "ask_size"
QUOTED_DEPTH_COLUMN: Final = "quoted_depth"
ORDER_IMBALANCE_COLUMN: Final = "order_imbalance"
ROLLING_VOLATILITY_COLUMN: Final = "rolling_volatility"
TRADE_PRICE_COLUMN: Final = "last"
TRADE_DIRECTION_COLUMN: Final = "trade_direction"
EFFECTIVE_SPREAD_COLUMN: Final = "effective_spread"
REALIZED_SPREAD_COLUMN: Final = "realized_spread"


class FeatureCalculationError(ValueError):
    """Raised when a feature cannot be calculated from the provided dataframe."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise FeatureCalculationError(f"dataframe is missing required columns: {missing}")


def _validate_rolling_window(window: int, min_periods: int | None) -> int:
    if window <= 0:
        raise FeatureCalculationError("window must be greater than zero")

    resolved_min_periods = window if min_periods is None else min_periods
    if resolved_min_periods <= 0:
        raise FeatureCalculationError("min_periods must be greater than zero")
    if resolved_min_periods > window:
        raise FeatureCalculationError("min_periods cannot be greater than window")

    return resolved_min_periods


def _resolve_mid_column(df: pd.DataFrame, mid_col: str) -> str:
    if mid_col in df.columns:
        return mid_col
    if mid_col == MID_PRICE_COLUMN and LEGACY_MID_PRICE_COLUMN in df.columns:
        return LEGACY_MID_PRICE_COLUMN
    return mid_col


def add_mid_price(
    df: pd.DataFrame,
    *,
    bid_col: str = BID_PRICE_COLUMN,
    ask_col: str = ASK_PRICE_COLUMN,
    output_col: str = LEGACY_MID_PRICE_COLUMN,
) -> pd.DataFrame:
    """Add mid price using: mid_price_t = (bid_price_t + ask_price_t) / 2."""

    _require_columns(df, [bid_col, ask_col])

    featured = df.copy()
    featured[output_col] = (featured[bid_col] + featured[ask_col]) / 2
    return featured


def add_quoted_spread(
    df: pd.DataFrame,
    *,
    bid_col: str = BID_PRICE_COLUMN,
    ask_col: str = ASK_PRICE_COLUMN,
    output_col: str = QUOTED_SPREAD_COLUMN,
) -> pd.DataFrame:
    """Add quoted spread using: quoted_spread_t = ask_price_t - bid_price_t."""

    _require_columns(df, [bid_col, ask_col])

    featured = df.copy()
    featured[output_col] = featured[ask_col] - featured[bid_col]
    return featured


def add_relative_spread(
    df: pd.DataFrame,
    *,
    bid_col: str = BID_PRICE_COLUMN,
    ask_col: str = ASK_PRICE_COLUMN,
    mid_col: str = LEGACY_MID_PRICE_COLUMN,
    output_col: str = RELATIVE_SPREAD_COLUMN,
) -> pd.DataFrame:
    """Add relative spread using: relative_spread_t = (ask_price_t - bid_price_t) / mid_price_t."""

    _require_columns(df, [bid_col, ask_col])

    featured = df.copy()
    resolved_mid_col = _resolve_mid_column(featured, mid_col)
    if resolved_mid_col not in featured.columns:
        featured[mid_col] = (featured[bid_col] + featured[ask_col]) / 2
        resolved_mid_col = mid_col

    denominator = featured[resolved_mid_col].where(featured[resolved_mid_col] != 0)
    featured[output_col] = (featured[ask_col] - featured[bid_col]) / denominator
    return featured


def add_quoted_depth(
    df: pd.DataFrame,
    *,
    bid_size_col: str = BID_SIZE_COLUMN,
    ask_size_col: str = ASK_SIZE_COLUMN,
    output_col: str = QUOTED_DEPTH_COLUMN,
) -> pd.DataFrame:
    """Add quoted depth using: quoted_depth_t = bid_size_t + ask_size_t."""

    _require_columns(df, [bid_size_col, ask_size_col])

    featured = df.copy()
    featured[output_col] = featured[bid_size_col] + featured[ask_size_col]
    return featured


def add_order_imbalance(
    df: pd.DataFrame,
    *,
    bid_size_col: str = BID_SIZE_COLUMN,
    ask_size_col: str = ASK_SIZE_COLUMN,
    output_col: str = ORDER_IMBALANCE_COLUMN,
) -> pd.DataFrame:
    """Add order imbalance using: (bid_size_t - ask_size_t) / (bid_size_t + ask_size_t)."""

    _require_columns(df, [bid_size_col, ask_size_col])

    featured = df.copy()
    denominator = (featured[bid_size_col] + featured[ask_size_col]).where(
        (featured[bid_size_col] + featured[ask_size_col]) != 0,
    )
    featured[output_col] = (featured[bid_size_col] - featured[ask_size_col]) / denominator
    return featured


def add_rolling_volatility(
    df: pd.DataFrame,
    *,
    price_col: str = MID_PRICE_COLUMN,
    output_col: str = ROLLING_VOLATILITY_COLUMN,
    window: int = 60,
    min_periods: int | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add rolling volatility using the rolling standard deviation of mid-price log returns."""

    resolved_price_col = _resolve_mid_column(df, price_col) if price_col in {MID_PRICE_COLUMN, LEGACY_MID_PRICE_COLUMN} else price_col
    required_columns = [resolved_price_col] if group_col is None else [resolved_price_col, group_col]
    _require_columns(df, required_columns)
    resolved_min_periods = _validate_rolling_window(window, min_periods)

    featured = df.copy()

    def calculate_log_returns(price: pd.Series) -> pd.Series:
        positive_price = price.where(price > 0)
        return np.log(positive_price / positive_price.shift())

    if group_col is None:
        log_returns = calculate_log_returns(featured[resolved_price_col])
        featured[output_col] = log_returns.rolling(
            window=window,
            min_periods=resolved_min_periods,
        ).std()
        return featured

    log_returns = featured.groupby(group_col, sort=False)[resolved_price_col].transform(
        calculate_log_returns
    )
    featured[output_col] = log_returns.groupby(featured[group_col], sort=False).transform(
        lambda series: series.rolling(window=window, min_periods=resolved_min_periods).std(),
    )
    return featured


def add_trade_direction_proxy(
    df: pd.DataFrame,
    *,
    trade_price_col: str = TRADE_PRICE_COLUMN,
    mid_col: str = MID_PRICE_COLUMN,
    output_col: str = TRADE_DIRECTION_COLUMN,
) -> pd.DataFrame:
    """Add trade direction proxy using: trade_direction_t = sign(last_t - mid_price_t)."""

    resolved_mid_col = _resolve_mid_column(df, mid_col)
    _require_columns(df, [trade_price_col, resolved_mid_col])

    featured = df.copy()
    featured[output_col] = np.sign(featured[trade_price_col] - featured[resolved_mid_col])
    return featured


def add_effective_spread(
    df: pd.DataFrame,
    *,
    trade_price_col: str = TRADE_PRICE_COLUMN,
    mid_col: str = MID_PRICE_COLUMN,
    trade_direction_col: str = TRADE_DIRECTION_COLUMN,
    output_col: str = EFFECTIVE_SPREAD_COLUMN,
) -> pd.DataFrame:
    """Add effective spread using: 2 * trade_direction_t * (last_t - mid_price_t)."""

    resolved_mid_col = _resolve_mid_column(df, mid_col)
    _require_columns(df, [trade_price_col, resolved_mid_col])

    featured = df.copy()
    if trade_direction_col not in featured.columns:
        featured[trade_direction_col] = np.sign(featured[trade_price_col] - featured[resolved_mid_col])

    featured[output_col] = (
        2 * featured[trade_direction_col] * (featured[trade_price_col] - featured[resolved_mid_col])
    )
    return featured


def add_realized_spread(
    df: pd.DataFrame,
    *,
    trade_price_col: str = TRADE_PRICE_COLUMN,
    mid_col: str = MID_PRICE_COLUMN,
    trade_direction_col: str = TRADE_DIRECTION_COLUMN,
    output_col: str = REALIZED_SPREAD_COLUMN,
    horizon: int = 300,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add realized spread using: 2 * trade_direction_t * (last_t - mid_price_{t+h})."""

    resolved_mid_col = _resolve_mid_column(df, mid_col)
    required_columns = [trade_price_col, resolved_mid_col]
    if group_col is not None:
        required_columns.append(group_col)
    _require_columns(df, required_columns)

    if horizon <= 0:
        raise FeatureCalculationError("horizon must be greater than zero")

    featured = df.copy()
    if trade_direction_col not in featured.columns:
        featured[trade_direction_col] = np.sign(featured[trade_price_col] - featured[resolved_mid_col])

    if group_col is None:
        future_mid = featured[resolved_mid_col].shift(-horizon)
    else:
        grouping_keys: list[pd.Series | str] = [group_col]
        if TIMESTAMP_COLUMN in featured.columns:
            grouping_keys.append(featured[TIMESTAMP_COLUMN].dt.normalize())
        future_mid = featured.groupby(grouping_keys, sort=False)[resolved_mid_col].shift(-horizon)

    featured[output_col] = (
        2 * featured[trade_direction_col] * (featured[trade_price_col] - future_mid)
    )
    return featured
