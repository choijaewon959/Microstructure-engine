# Sample Data

This folder contains small commit-safe CSV samples so the project can run without the full local raw data lake.

The current sample is a 120-row slice of GS 1-second data from `2025-01-02`:

- `GS_1s_2025-01-02_prices_sample.csv`
- `GS_1s_2025-01-02_quotes_sample.csv`
- `GS_1s_2025-01-02_trades_sample.csv`

Generation source:

```text
data/raw/prices/GS_1s_2025-01-02.parquet
data/raw/quotes/GS_quotes_1s_2025-01-02.parquet
data/raw/trades/GS_trades_1s_2025-01-02.parquet
```

The `ts` column is serialized as text in CSV form. These files are intended for dashboard fallbacks, examples, and lightweight tests.

