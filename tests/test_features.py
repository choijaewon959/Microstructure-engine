import math

import pandas as pd
import pytest

from microdash.features import (
    FeatureCalculationError,
    add_effective_spread,
    add_mid_price,
    add_order_imbalance,
    add_quoted_depth,
    add_quoted_spread,
    add_realized_spread,
    add_relative_spread,
    add_rolling_volatility,
    add_trade_direction_proxy,
)


def test_add_mid_price_uses_average_of_bid_and_ask() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0, 100.5, 101.0],
            "ask_price": [101.0, 101.5, 103.0],
        },
    )

    featured = add_mid_price(quotes)

    assert featured["mid_price"].tolist() == [100.0, 101.0, 102.0]


def test_add_mid_price_preserves_input_dataframe() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0],
            "ask_price": [101.0],
        },
    )

    add_mid_price(quotes)

    assert "mid_price" not in quotes.columns


def test_add_mid_price_allows_custom_column_names() -> None:
    quotes = pd.DataFrame(
        {
            "bid": [10.0],
            "ask": [10.2],
        },
    )

    featured = add_mid_price(quotes, bid_col="bid", ask_col="ask", output_col="mid")

    assert featured.loc[0, "mid"] == 10.1


def test_add_mid_price_requires_bid_and_ask_columns() -> None:
    quotes = pd.DataFrame({"bid_price": [99.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_mid_price(quotes)


def test_add_quoted_spread_uses_ask_minus_bid() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0, 100.5, 101.0],
            "ask_price": [101.0, 101.5, 103.0],
        },
    )

    featured = add_quoted_spread(quotes)

    assert featured["quoted_spread"].tolist() == [2.0, 1.0, 2.0]


def test_add_quoted_spread_preserves_existing_raw_spread_column() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0],
            "ask_price": [101.0],
            "spread": [1.9],
        },
    )

    featured = add_quoted_spread(quotes)

    assert featured.loc[0, "spread"] == 1.9
    assert featured.loc[0, "quoted_spread"] == 2.0


def test_add_quoted_spread_allows_custom_output_column() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0],
            "ask_price": [101.0],
        },
    )

    featured = add_quoted_spread(quotes, output_col="spread")

    assert featured.loc[0, "spread"] == 2.0


def test_add_quoted_spread_requires_bid_and_ask_columns() -> None:
    quotes = pd.DataFrame({"ask_price": [101.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_quoted_spread(quotes)


def test_add_relative_spread_uses_spread_divided_by_mid_price() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0, 100.5],
            "ask_price": [101.0, 101.5],
            "mid_price": [100.0, 101.0],
        },
    )

    featured = add_relative_spread(quotes)

    assert featured["relative_spread"].tolist() == [0.02, 1.0 / 101.0]


def test_add_relative_spread_calculates_mid_price_when_missing() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [99.0],
            "ask_price": [101.0],
        },
    )

    featured = add_relative_spread(quotes)

    assert featured.loc[0, "mid_price"] == 100.0
    assert featured.loc[0, "relative_spread"] == 0.02


def test_add_relative_spread_returns_nan_for_zero_mid_price() -> None:
    quotes = pd.DataFrame(
        {
            "bid_price": [-1.0],
            "ask_price": [1.0],
            "mid_price": [0.0],
        },
    )

    featured = add_relative_spread(quotes)

    assert pd.isna(featured.loc[0, "relative_spread"])


def test_add_relative_spread_requires_bid_and_ask_columns() -> None:
    quotes = pd.DataFrame({"bid_price": [99.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_relative_spread(quotes)


def test_add_quoted_depth_uses_bid_size_plus_ask_size() -> None:
    quotes = pd.DataFrame(
        {
            "bid_size": [100.0, 75.0, 0.0],
            "ask_size": [150.0, 25.0, 10.0],
        },
    )

    featured = add_quoted_depth(quotes)

    assert featured["quoted_depth"].tolist() == [250.0, 100.0, 10.0]


def test_add_quoted_depth_preserves_input_dataframe() -> None:
    quotes = pd.DataFrame(
        {
            "bid_size": [100.0],
            "ask_size": [150.0],
        },
    )

    add_quoted_depth(quotes)

    assert "quoted_depth" not in quotes.columns


def test_add_quoted_depth_allows_custom_column_names() -> None:
    quotes = pd.DataFrame(
        {
            "bid_qty": [100.0],
            "ask_qty": [150.0],
        },
    )

    featured = add_quoted_depth(
        quotes,
        bid_size_col="bid_qty",
        ask_size_col="ask_qty",
        output_col="depth",
    )

    assert featured.loc[0, "depth"] == 250.0


def test_add_quoted_depth_requires_bid_and_ask_size_columns() -> None:
    quotes = pd.DataFrame({"bid_size": [100.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_quoted_depth(quotes)


def test_add_order_imbalance_uses_depth_difference_over_total_depth() -> None:
    quotes = pd.DataFrame(
        {
            "bid_size": [150.0, 25.0, 100.0],
            "ask_size": [50.0, 75.0, 100.0],
        },
    )

    featured = add_order_imbalance(quotes)

    assert featured["order_imbalance"].tolist() == [0.5, -0.5, 0.0]


def test_add_order_imbalance_returns_nan_for_zero_total_depth() -> None:
    quotes = pd.DataFrame(
        {
            "bid_size": [0.0],
            "ask_size": [0.0],
        },
    )

    featured = add_order_imbalance(quotes)

    assert pd.isna(featured.loc[0, "order_imbalance"])


def test_add_order_imbalance_allows_custom_column_names() -> None:
    quotes = pd.DataFrame(
        {
            "bid_qty": [150.0],
            "ask_qty": [50.0],
        },
    )

    featured = add_order_imbalance(
        quotes,
        bid_size_col="bid_qty",
        ask_size_col="ask_qty",
        output_col="imbalance",
    )

    assert featured.loc[0, "imbalance"] == 0.5


def test_add_order_imbalance_requires_bid_and_ask_size_columns() -> None:
    quotes = pd.DataFrame({"ask_size": [100.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_order_imbalance(quotes)


def test_add_rolling_volatility_uses_log_return_rolling_std() -> None:
    quotes = pd.DataFrame({"mid_price": [100.0, 101.0, 102.0, 103.0]})

    featured = add_rolling_volatility(quotes, window=2, min_periods=2)
    log_returns = pd.Series(
        [
            float("nan"),
            math.log(101.0 / 100.0),
            math.log(102.0 / 101.0),
            math.log(103.0 / 102.0),
        ],
    )
    expected_returns = log_returns.rolling(window=2, min_periods=2).std()

    pd.testing.assert_series_equal(
        featured["rolling_volatility"],
        expected_returns,
        check_names=False,
    )


def test_add_rolling_volatility_calculates_independently_per_group() -> None:
    quotes = pd.DataFrame(
        {
            "symbol": ["GS", "GS", "MS", "MS"],
            "mid_price": [100.0, 101.0, 200.0, 202.0],
        },
    )

    featured = add_rolling_volatility(quotes, window=2, min_periods=1, group_col="symbol")

    assert pd.isna(featured.loc[0, "rolling_volatility"])
    assert pd.isna(featured.loc[1, "rolling_volatility"])
    assert pd.isna(featured.loc[2, "rolling_volatility"])
    assert pd.isna(featured.loc[3, "rolling_volatility"])


def test_add_rolling_volatility_returns_nan_for_non_positive_prices() -> None:
    quotes = pd.DataFrame({"mid_price": [100.0, 0.0, 102.0]})

    featured = add_rolling_volatility(quotes, window=2, min_periods=1)

    assert pd.isna(featured.loc[1, "rolling_volatility"])
    assert pd.isna(featured.loc[2, "rolling_volatility"])


def test_add_rolling_volatility_requires_price_column() -> None:
    quotes = pd.DataFrame({"bid_price": [99.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_rolling_volatility(quotes)


def test_add_rolling_volatility_rejects_invalid_window() -> None:
    quotes = pd.DataFrame({"mid_price": [100.0]})

    with pytest.raises(FeatureCalculationError, match="window must be greater than zero"):
        add_rolling_volatility(quotes, window=0)


def test_add_trade_direction_proxy_uses_sign_of_trade_price_minus_mid_price() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, 99.0, 100.0],
            "mid_price": [100.0, 100.0, 100.0],
        },
    )

    featured = add_trade_direction_proxy(trades)

    assert featured["trade_direction"].tolist() == [1.0, -1.0, 0.0]


def test_add_trade_direction_proxy_keeps_missing_inputs_as_nan() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, None],
            "mid_price": [None, 100.0],
        },
    )

    featured = add_trade_direction_proxy(trades)

    assert pd.isna(featured.loc[0, "trade_direction"])
    assert pd.isna(featured.loc[1, "trade_direction"])


def test_add_trade_direction_proxy_allows_custom_column_names() -> None:
    trades = pd.DataFrame(
        {
            "price": [101.0],
            "mid": [100.0],
        },
    )

    featured = add_trade_direction_proxy(
        trades,
        trade_price_col="price",
        mid_col="mid",
        output_col="side",
    )

    assert featured.loc[0, "side"] == 1.0


def test_add_trade_direction_proxy_requires_trade_price_and_mid_columns() -> None:
    trades = pd.DataFrame({"last": [101.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_trade_direction_proxy(trades)


def test_add_effective_spread_uses_signed_distance_from_mid_price() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, 99.0, 100.0],
            "mid_price": [100.0, 100.0, 100.0],
            "trade_direction": [1.0, -1.0, 0.0],
        },
    )

    featured = add_effective_spread(trades)

    assert featured["effective_spread"].tolist() == [2.0, 2.0, 0.0]


def test_add_effective_spread_calculates_trade_direction_when_missing() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, 99.0],
            "mid_price": [100.0, 100.0],
        },
    )

    featured = add_effective_spread(trades)

    assert featured["trade_direction"].tolist() == [1.0, -1.0]
    assert featured["effective_spread"].tolist() == [2.0, 2.0]


def test_add_effective_spread_keeps_missing_inputs_as_nan() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, None],
            "mid_price": [None, 100.0],
        },
    )

    featured = add_effective_spread(trades)

    assert pd.isna(featured.loc[0, "effective_spread"])
    assert pd.isna(featured.loc[1, "effective_spread"])


def test_add_effective_spread_allows_custom_column_names() -> None:
    trades = pd.DataFrame(
        {
            "price": [101.0],
            "mid": [100.0],
            "side": [1.0],
        },
    )

    featured = add_effective_spread(
        trades,
        trade_price_col="price",
        mid_col="mid",
        trade_direction_col="side",
        output_col="eff_spread",
    )

    assert featured.loc[0, "eff_spread"] == 2.0


def test_add_effective_spread_requires_trade_price_and_mid_columns() -> None:
    trades = pd.DataFrame({"last": [101.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_effective_spread(trades)


def test_add_realized_spread_uses_future_mid_price() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, 99.0, 100.0],
            "mid_price": [100.0, 100.5, 98.5],
            "trade_direction": [1.0, -1.0, 0.0],
        },
    )

    featured = add_realized_spread(trades, horizon=1)

    assert featured.loc[0, "realized_spread"] == 1.0
    assert featured.loc[1, "realized_spread"] == -1.0
    assert pd.isna(featured.loc[2, "realized_spread"])


def test_add_realized_spread_calculates_trade_direction_when_missing() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0, 99.0, 100.0],
            "mid_price": [100.0, 100.5, 98.5],
        },
    )

    featured = add_realized_spread(trades, horizon=1)

    assert featured["trade_direction"].tolist() == [1.0, -1.0, 1.0]
    assert featured.loc[0, "realized_spread"] == 1.0
    assert featured.loc[1, "realized_spread"] == -1.0


def test_add_realized_spread_shifts_future_mid_independently_per_group() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["GS", "GS", "MS", "MS"],
            "last": [101.0, 102.0, 201.0, 202.0],
            "mid_price": [100.0, 100.5, 200.0, 200.5],
            "trade_direction": [1.0, 1.0, 1.0, 1.0],
        },
    )

    featured = add_realized_spread(trades, horizon=1, group_col="symbol")

    assert featured.loc[0, "realized_spread"] == 1.0
    assert pd.isna(featured.loc[1, "realized_spread"])
    assert featured.loc[2, "realized_spread"] == 1.0
    assert pd.isna(featured.loc[3, "realized_spread"])




def test_add_realized_spread_respects_session_boundaries_when_timestamp_present() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["GS", "GS", "GS", "GS"],
            "timestamp": pd.to_datetime([
                "2024-01-02 15:59:59",
                "2024-01-02 16:00:00",
                "2024-01-03 09:30:00",
                "2024-01-03 09:30:01",
            ]),
            "last": [101.0, 101.5, 102.0, 102.5],
            "mid": [100.0, 100.5, 110.0, 110.5],
            "trade_direction": [1.0, 1.0, 1.0, 1.0],
        },
    )

    featured = add_realized_spread(trades, horizon=1, group_col="symbol")

    assert featured.loc[0, "realized_spread"] == 1.0
    assert pd.isna(featured.loc[1, "realized_spread"])
    assert featured.loc[2, "realized_spread"] == -17.0
    assert pd.isna(featured.loc[3, "realized_spread"])

def test_add_realized_spread_allows_custom_column_names() -> None:
    trades = pd.DataFrame(
        {
            "price": [101.0, 102.0],
            "mid": [100.0, 100.5],
            "side": [1.0, 1.0],
        },
    )

    featured = add_realized_spread(
        trades,
        trade_price_col="price",
        mid_col="mid",
        trade_direction_col="side",
        output_col="realized",
        horizon=1,
    )

    assert featured.loc[0, "realized"] == 1.0


def test_add_realized_spread_requires_trade_price_and_mid_columns() -> None:
    trades = pd.DataFrame({"last": [101.0]})

    with pytest.raises(FeatureCalculationError, match="missing required columns"):
        add_realized_spread(trades)


def test_add_realized_spread_rejects_invalid_horizon() -> None:
    trades = pd.DataFrame(
        {
            "last": [101.0],
            "mid_price": [100.0],
        },
    )

    with pytest.raises(FeatureCalculationError, match="horizon must be greater than zero"):
        add_realized_spread(trades, horizon=0)
