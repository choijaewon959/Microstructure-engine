# Microstructure Engine

Streamlit dashboard for market microstructure analytics using 1-second price, quote, and trade data.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python -m pytest
streamlit run dashboard/app.py
```

## Data Layout

Raw market data stays local under `data/raw/` and is ignored by Git. Reproducible outputs belong in `data/processed/` and `data/features/`, while small commit-safe examples belong in `data/sample/`.

See [docs/data-conventions.md](docs/data-conventions.md) for the full storage rules.
