# Data Conventions

This project separates local raw market data from commit-safe project artifacts.

## Directory Layout

- `data/raw/`: local raw 1-second prices, quotes, and trades. Do not commit this folder.
- `data/processed/`: reproducible cleaned intermediate outputs. Keep generated files local.
- `data/features/`: reproducible feature tables. Keep generated files local.
- `data/sample/`: small commit-safe demo data for tests, examples, and dashboard fallback flows.

## Raw Data

Raw parquet files should remain local because they are large and vendor/source-specific. The expected local raw layout is:

```text
data/raw/
├── prices/
├── quotes/
└── trades/
```

Do not mutate raw files in place. Cleaning and feature generation should write new files to `data/processed/` or `data/features/`.

## Commit Rules

- Commit code, tests, docs, and small sample data.
- Do not commit raw parquet files.
- Do not commit generated processed or feature outputs unless explicitly requested.
- Keep `.gitkeep` placeholders so the expected folders exist in fresh clones.
