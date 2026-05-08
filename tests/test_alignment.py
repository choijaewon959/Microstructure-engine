import pandas as pd
import pytest

from microdash.alignment import (
    SESSION_DATE_COLUMN,
    AlignmentError,
    align_trades_to_quotes,
    prepare_quotes_for_alignment,
)


def test_align_trades_to_prior_quote_and_sorts_inputs() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["GS", "GS"],
            "timestamp": [
                pd.Timestamp("2025-01-02 09:30:02"),
                pd.Timestamp("2025-01-02 09:30:01"),
            ],
            "last": [101.0, 100.0],
        }
    )
    quotes = pd.DataFrame(
        {
            "symbol": ["GS", "GS"],
            "timestamp": [
                pd.Timestamp("2025-01-02 09:30:00"),
                pd.Timestamp("2025-01-02 09:30:02"),
            ],
            "bid_price": [99.0, 100.0],
            "ask_price": [101.0, 102.0],
        }
    )

    aligned = align_trades_to_quotes(trades, quotes, tolerance="2s")

    assert list(aligned["timestamp"]) == [
        pd.Timestamp("2025-01-02 09:30:01"),
        pd.Timestamp("2025-01-02 09:30:02"),
    ]
    assert list(aligned["bid_price"]) == [99.0, 100.0]


def test_prepare_quotes_keeps_last_duplicate_timestamp() -> None:
    quotes = pd.DataFrame(
        {
            "symbol": ["GS", "GS"],
            "timestamp": [pd.Timestamp("2025-01-02 09:30:00")] * 2,
            "bid_price": [99.0, 99.5],
        }
    )

    prepared = prepare_quotes_for_alignment(quotes)

    assert len(prepared) == 1
    assert prepared.iloc[0]["bid_price"] == 99.5


def test_alignment_respects_symbol_boundaries() -> None:
    trades = pd.DataFrame(
        {"symbol": ["MS"], "timestamp": [pd.Timestamp("2025-01-02 09:30:01")], "last": [100.0]}
    )
    quotes = pd.DataFrame(
        {
            "symbol": ["GS"],
            "timestamp": [pd.Timestamp("2025-01-02 09:30:00")],
            "bid_price": [99.0],
        }
    )

    aligned = align_trades_to_quotes(trades, quotes, tolerance="5s")

    assert "bid_price" not in aligned.columns or aligned["bid_price"].isna().all()
    assert aligned.iloc[0]["symbol"] == "MS"


def test_alignment_respects_session_date_boundaries() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["GS"],
            "timestamp": [pd.Timestamp("2025-01-03 09:30:00")],
            "last": [100.0],
        }
    )
    quotes = pd.DataFrame(
        {
            "symbol": ["GS"],
            "timestamp": [pd.Timestamp("2025-01-02 15:59:59")],
            "bid_price": [99.0],
        }
    )

    aligned = align_trades_to_quotes(trades, quotes, tolerance="1d")

    assert "bid_price" not in aligned.columns or aligned["bid_price"].isna().all()
    assert aligned.iloc[0][SESSION_DATE_COLUMN] == pd.Timestamp("2025-01-03").date()


def test_alignment_uses_tolerance() -> None:
    trades = pd.DataFrame(
        {"symbol": ["GS"], "timestamp": [pd.Timestamp("2025-01-02 09:30:10")], "last": [100.0]}
    )
    quotes = pd.DataFrame(
        {
            "symbol": ["GS"],
            "timestamp": [pd.Timestamp("2025-01-02 09:30:00")],
            "bid_price": [99.0],
        }
    )

    aligned = align_trades_to_quotes(trades, quotes, tolerance="1s")

    assert pd.isna(aligned.iloc[0]["bid_price"])


def test_alignment_validates_required_columns() -> None:
    with pytest.raises(AlignmentError, match="missing required columns"):
        align_trades_to_quotes(
            pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-02")]}),
            pd.DataFrame({"symbol": ["GS"], "timestamp": [pd.Timestamp("2025-01-02")]}),
        )
