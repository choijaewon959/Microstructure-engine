# AGENTS.md

## Project Overview

This repo builds a Streamlit market microstructure analytics dashboard using 6 months of 1-second prices, quotes, and trades for GS, MS, SPY, IVV, and XLF.

The project should be interview-ready and demonstrate market data cleaning, quote/trade alignment, feature engineering, liquidity analytics, execution-style metrics, dashboard design, and clean Python engineering.

## Core Goals

- Load 1-second quote and trade data.
- Compute reusable microstructure features.
- Visualize single-symbol behavior.
- Compare liquidity across symbols.
- Summarize daily and intraday metrics.
- Leave room for future market-making simulator extensions.

## MVP Non-Goals

- Do not build a full market-making simulator.
- Do not build a complex backtesting engine yet.
- Do not use Google Sheets as the main raw data store.
- Do not over-engineer before the core dashboard works.

## Expected Structure

```text
microstructure-analytics-dashboard/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── sample/
├── src/
│   └── microdash/
├── dashboard/
├── scripts/
├── tests/
└── docs/
```

## Git and Linear Workflow

- Each task should correspond to a Linear issue when available.
- Start by reading the Linear issue title, description, acceptance criteria, and comments.
- Before coding, summarize the intended implementation plan.
- Keep changes scoped to the assigned issue.
- Every task should be implemented on a separate Git branch.
- Never commit directly to `main` or `master`.
- Branch names should follow `linear-<ISSUE-ID>-short-description`, for example `linear-MICRO-12-add-ofi-feature`.
- Include the Linear issue ID in every commit message.
- Open a pull request into `main` after implementation.
- PR title should start with the Linear issue ID, for example `[MICRO-12] Add OFI feature`.
- PR description should include the Linear issue link or issue ID, summary of changes, files modified, tests run, and known limitations.
- After opening the PR, update the Linear issue with the PR link and a short implementation summary.
- Do not mark the Linear issue as Done unless the PR has passed tests and is ready for review.

## Data Handling

- Keep raw data in `data/raw/`; never mutate it in place.
- Put cleaned intermediate files in `data/processed/`.
- Put reusable feature tables in `data/features/`.
- Put commit-safe demo data in `data/sample/`.
- Do not commit large raw market data files unless explicitly requested.
- Document expected columns such as `timestamp`, `symbol`, `bid`, `ask`, `bid_size`, `ask_size`, `price`, and `size`.

## Feature Priorities

Start with transparent, interview-explainable features:

- mid price
- quoted spread
- relative spread
- effective spread
- realized spread
- trade direction proxy
- order imbalance
- quoted depth
- rolling volatility
- intraday volume profile
- liquidity summary by symbol and day

## Dashboard Priorities

The Streamlit app should feel like an analytical tool, not a landing page.

Prioritize symbol and date controls, quote/trade quality summaries, price and mid-price charts, spread charts, depth charts, volume and trade-count charts, cross-symbol liquidity comparisons, and daily summary tables.

## Code Style

- Keep code modular, readable, and reviewable.
- Prefer small, composable Python modules under `src/microdash/`.
- Keep Streamlit code in `dashboard/` focused on layout, controls, and visualization.
- Use type hints for public functions.
- Prefer readable vectorized dataframe operations.
- Avoid notebook-only logic in production modules.
- Add comments only where they clarify non-obvious logic.
- Do not hard-code machine-specific absolute paths in library code.

## Testing

- Run the relevant test suite before finishing.
- If tests cannot be run, explain why in the final update.
- Do not mark a Linear issue as done unless implementation and tests are complete.
- Add focused tests for timestamp parsing, quote/trade alignment, spread calculations, rolling metrics, missing-data behavior, and symbol-level aggregations.
- Prefer small synthetic fixtures over large data files.

## MVP Definition

The MVP is complete when a user can run the Streamlit app locally, load sample data, select a symbol and date range, inspect core microstructure metrics, and compare liquidity across the target symbols.

Future simulator or backtesting work should build on the cleaned data and feature modules rather than replacing them.
