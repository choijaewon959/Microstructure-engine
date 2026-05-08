from pathlib import Path

import pandas as pd

from microdash.feature_generation import build_feature_table, generate_feature_file


def quote_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-02 09:30:00",
                    "2025-01-02 09:30:01",
                    "2025-01-02 09:30:02",
                ]
            ),
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 09:30:00",
                    "2025-01-02 09:30:01",
                    "2025-01-02 09:30:02",
                ]
            ),
            "symbol": ["GS", "GS", "GS"],
            "bid_price": [99.0, 100.0, 101.0],
            "ask_price": [101.0, 102.0, 103.0],
            "bid_size": [150.0, 100.0, 80.0],
            "ask_size": [50.0, 100.0, 120.0],
            "mid": [100.0, 101.0, 102.0],
            "spread": [2.0, 2.0, 2.0],
        }
    )


def trade_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-02 09:30:00",
                    "2025-01-02 09:30:01",
                    "2025-01-02 09:30:02",
                ]
            ),
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 09:30:00",
                    "2025-01-02 09:30:01",
                    "2025-01-02 09:30:02",
                ]
            ),
            "symbol": ["GS", "GS", "GS"],
            "last": [100.5, 100.0, 103.0],
            "volume": [10.0, 20.0, 30.0],
            "n_trades": [1.0, 2.0, 3.0],
            "vwap": [100.5, 100.0, 103.0],
        }
    )


def test_build_feature_table_adds_quote_features_with_expected_values() -> None:
    features = build_feature_table(quote_rows(), volatility_window=2)

    assert features["mid_price"].tolist() == [100.0, 101.0, 102.0]
    assert features["quoted_spread"].tolist() == [2.0, 2.0, 2.0]
    assert features["relative_spread"].tolist() == [0.02, 2.0 / 101.0, 2.0 / 102.0]
    assert features["quoted_depth"].tolist() == [200.0, 200.0, 200.0]
    assert features["order_imbalance"].tolist() == [0.5, 0.0, -0.2]
    assert pd.isna(features.loc[0, "rolling_volatility"])
    assert pd.isna(features.loc[1, "rolling_volatility"])
    assert pd.notna(features.loc[2, "rolling_volatility"])


def test_build_feature_table_adds_aligned_trade_features_with_expected_values() -> None:
    features = build_feature_table(
        quote_rows(),
        trade_rows(),
        volatility_window=2,
        realized_horizon=1,
    )

    assert features["trade_direction"].tolist() == [1.0, -1.0, 1.0]
    assert features["effective_spread"].tolist() == [1.0, 2.0, 2.0]
    assert features.loc[0, "realized_spread"] == -1.0
    assert features.loc[1, "realized_spread"] == 4.0
    assert pd.isna(features.loc[2, "realized_spread"])


def test_generate_feature_file_writes_parquet_output(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    quote_path = raw_root / "quotes" / "GS_quotes_1s_2025-01-02.parquet"
    trade_path = raw_root / "trades" / "GS_trades_1s_2025-01-02.parquet"
    output_path = tmp_path / "features" / "GS_features_1s_2025-01-02.parquet"
    quote_path.parent.mkdir(parents=True)
    trade_path.parent.mkdir(parents=True)
    quote_rows().drop(columns=["timestamp"]).to_parquet(quote_path)
    trade_rows().drop(columns=["timestamp"]).to_parquet(trade_path)

    result = generate_feature_file(
        quote_path,
        output_path,
        trade_path=trade_path,
        volatility_window=2,
        realized_horizon=1,
    )

    written = pd.read_parquet(output_path)
    assert result.row_count == 3
    assert result.used_trades is True
    assert written["effective_spread"].tolist() == [1.0, 2.0, 2.0]
