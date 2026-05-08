"""Batch feature generation utilities for raw quote and trade data."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from microdash.alignment import align_trades_to_quotes
from microdash.data_inventory import RawFileRecord, parse_raw_filename, scan_raw_inventory
from microdash.features import (
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
from microdash.raw_loader import normalize_raw_dataframe

LOGGER = logging.getLogger(__name__)
DEFAULT_SYMBOLS = ("GS", "MS", "SPY", "IVV", "XLF")
FEATURE_FILE_TEMPLATE = "{symbol}_features_{frequency}_{date}.parquet"
QUOTE_SAMPLE_RE = re.compile(
    r"^(?P<symbol>[A-Z]+)_(?P<frequency>\d+s)_(?P<date>\d{4}-\d{2}-\d{2})_quotes_sample\.csv$"
)


@dataclass(frozen=True)
class FeatureGenerationResult:
    """Summary for one generated feature file."""

    symbol: str
    date: pd.Timestamp
    output_path: Path
    row_count: int
    used_trades: bool


def read_raw_table(path: str | Path, data_type: str) -> pd.DataFrame:
    """Read a CSV or parquet raw table and normalize its core schema."""

    file_path = Path(path)
    if file_path.suffix == ".parquet":
        raw = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        raw = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported raw file extension: {file_path.suffix!r}")

    return normalize_raw_dataframe(raw, data_type)


def add_quote_features(
    quotes: pd.DataFrame,
    *,
    volatility_window: int = 60,
) -> pd.DataFrame:
    """Add reusable quote-derived features to a normalized quote dataframe."""

    featured = add_mid_price(quotes)
    featured = add_quoted_spread(featured)
    featured = add_relative_spread(featured)
    featured = add_quoted_depth(featured)
    featured = add_order_imbalance(featured)
    return add_rolling_volatility(
        featured,
        window=volatility_window,
        group_col="symbol",
    )


def add_aligned_trade_features(
    trades: pd.DataFrame,
    quote_features: pd.DataFrame,
    *,
    alignment_tolerance: str | pd.Timedelta | None = "1s",
    realized_horizon: int = 300,
) -> pd.DataFrame:
    """Align trades to quote features and add trade-aware metrics."""

    aligned = align_trades_to_quotes(
        trades,
        quote_features,
        tolerance=alignment_tolerance,
    )
    featured = add_trade_direction_proxy(aligned)
    featured = add_effective_spread(featured)
    return add_realized_spread(featured, horizon=realized_horizon, group_col="symbol")


def build_feature_table(
    quotes: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    volatility_window: int = 60,
    alignment_tolerance: str | pd.Timedelta | None = "1s",
    realized_horizon: int = 300,
) -> pd.DataFrame:
    """Build quote-only or aligned trade/quote feature rows."""

    quote_features = add_quote_features(quotes, volatility_window=volatility_window)
    if trades is None or trades.empty:
        return quote_features

    return add_aligned_trade_features(
        trades,
        quote_features,
        alignment_tolerance=alignment_tolerance,
        realized_horizon=realized_horizon,
    )


def output_path_for_record(record: RawFileRecord, output_root: str | Path) -> Path:
    """Return the feature output path for one raw quote record."""

    return Path(output_root) / FEATURE_FILE_TEMPLATE.format(
        symbol=record.symbol,
        frequency=record.frequency,
        date=record.date.date().isoformat(),
    )


def quote_record_from_path(path: str | Path) -> RawFileRecord:
    """Parse raw or sample quote filename metadata."""

    file_path = Path(path)
    try:
        return parse_raw_filename(file_path, data_type="quotes")
    except ValueError:
        match = QUOTE_SAMPLE_RE.match(file_path.name)
        if match is None:
            raise

    fields = match.groupdict()
    return RawFileRecord(
        path=file_path,
        data_type="quotes",
        symbol=fields["symbol"],
        frequency=fields["frequency"],
        date=pd.Timestamp(fields["date"]),
    )


def generate_feature_file(
    quote_path: str | Path,
    output_path: str | Path,
    *,
    trade_path: str | Path | None = None,
    volatility_window: int = 60,
    alignment_tolerance: str | pd.Timedelta | None = "1s",
    realized_horizon: int = 300,
    overwrite: bool = False,
) -> FeatureGenerationResult:
    """Generate one feature parquet file from one quote file and optional trade file."""

    quote_record = quote_record_from_path(quote_path)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        LOGGER.info("Skipping existing feature file: %s", destination)
        return FeatureGenerationResult(
            symbol=quote_record.symbol,
            date=quote_record.date,
            output_path=destination,
            row_count=len(pd.read_parquet(destination)),
            used_trades=trade_path is not None,
        )

    quotes = read_raw_table(quote_path, "quotes")
    trades = read_raw_table(trade_path, "trades") if trade_path is not None else None
    features = build_feature_table(
        quotes,
        trades,
        volatility_window=volatility_window,
        alignment_tolerance=alignment_tolerance,
        realized_horizon=realized_horizon,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    LOGGER.info("Wrote %s rows to %s", len(features), destination)
    return FeatureGenerationResult(
        symbol=quote_record.symbol,
        date=quote_record.date,
        output_path=destination,
        row_count=len(features),
        used_trades=trades is not None,
    )


def _filtered_quote_records(
    raw_root: str | Path,
    *,
    symbols: set[str],
    dates: set[str] | None,
) -> list[RawFileRecord]:
    inventory = scan_raw_inventory(raw_root)
    if inventory.empty:
        return []

    quotes = inventory[inventory["data_type"] == "quotes"].copy()
    quotes = quotes[quotes["symbol"].isin(symbols)]
    if dates is not None:
        quotes = quotes[quotes["date"].dt.date.astype(str).isin(dates)]

    return [
        RawFileRecord(
            path=Path(row.path),
            data_type=row.data_type,
            symbol=row.symbol,
            frequency=row.frequency,
            date=row.date,
        )
        for row in quotes.itertuples(index=False)
    ]


def _trade_lookup(raw_root: str | Path) -> dict[tuple[str, str, str], Path]:
    inventory = scan_raw_inventory(raw_root)
    trades = inventory[inventory["data_type"] == "trades"] if not inventory.empty else inventory
    return {
        (row.symbol, row.frequency, row.date.date().isoformat()): Path(row.path)
        for row in trades.itertuples(index=False)
    }


def generate_from_raw_root(
    raw_root: str | Path = "data/raw",
    output_root: str | Path = "data/features",
    *,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    dates: tuple[str, ...] | None = None,
    volatility_window: int = 60,
    alignment_tolerance: str | pd.Timedelta | None = "1s",
    realized_horizon: int = 300,
    overwrite: bool = False,
) -> list[FeatureGenerationResult]:
    """Generate feature files for selected raw quote files under a raw data root."""

    quote_records = _filtered_quote_records(
        raw_root,
        symbols=set(symbols),
        dates=set(dates) if dates is not None else None,
    )
    trade_paths = _trade_lookup(raw_root)
    results: list[FeatureGenerationResult] = []

    if not quote_records:
        LOGGER.warning("No quote files found for symbols=%s dates=%s", symbols, dates)
        return results

    for quote_record in quote_records:
        lookup_key = (
            quote_record.symbol,
            quote_record.frequency,
            quote_record.date.date().isoformat(),
        )
        trade_path = trade_paths.get(lookup_key)
        if trade_path is None:
            LOGGER.warning(
                "Missing trade file for %s %s %s; writing quote-only features",
                quote_record.symbol,
                quote_record.frequency,
                quote_record.date.date().isoformat(),
            )

        results.append(
            generate_feature_file(
                quote_record.path,
                output_path_for_record(quote_record, output_root),
                trade_path=trade_path,
                volatility_window=volatility_window,
                alignment_tolerance=alignment_tolerance,
                realized_horizon=realized_horizon,
                overwrite=overwrite,
            )
        )

    return results


def generate_sample_features(
    sample_root: str | Path = "data/sample",
    output_root: str | Path = "data/features",
    *,
    overwrite: bool = True,
) -> FeatureGenerationResult:
    """Generate the commit-safe GS sample feature file."""

    sample_dir = Path(sample_root)
    quote_path = sample_dir / "GS_1s_2025-01-02_quotes_sample.csv"
    trade_path = sample_dir / "GS_1s_2025-01-02_trades_sample.csv"
    output_path = Path(output_root) / "GS_features_1s_2025-01-02_sample.parquet"
    return generate_feature_file(
        quote_path,
        output_path,
        trade_path=trade_path,
        overwrite=overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the feature generation CLI parser."""

    parser = argparse.ArgumentParser(description="Generate market microstructure feature tables.")
    parser.add_argument(
        "--raw-root", default="data/raw", help="Root folder containing raw parquet data."
    )
    parser.add_argument(
        "--output-root", default="data/features", help="Folder for feature parquet outputs."
    )
    parser.add_argument(
        "--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Symbols to process."
    )
    parser.add_argument("--dates", nargs="*", help="Optional YYYY-MM-DD dates to process.")
    parser.add_argument("--volatility-window", type=int, default=60)
    parser.add_argument("--realized-horizon", type=int, default=300)
    parser.add_argument("--alignment-tolerance", default="1s")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sample", action="store_true", help="Generate the commit-safe sample feature file."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the feature generation CLI."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.sample:
        result = generate_sample_features(output_root=args.output_root, overwrite=args.overwrite)
        LOGGER.info("Generated sample feature file: %s", result.output_path)
        return 0

    results = generate_from_raw_root(
        raw_root=args.raw_root,
        output_root=args.output_root,
        symbols=tuple(args.symbols),
        dates=tuple(args.dates) if args.dates else None,
        volatility_window=args.volatility_window,
        alignment_tolerance=args.alignment_tolerance,
        realized_horizon=args.realized_horizon,
        overwrite=args.overwrite,
    )
    LOGGER.info("Generated %s feature file(s)", len(results))
    return 0
