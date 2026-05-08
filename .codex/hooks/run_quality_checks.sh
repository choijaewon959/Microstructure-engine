#!/usr/bin/env bash
set -e

echo "Running Ruff lint..."
ruff check .

echo "Running Ruff format check..."
ruff format --check .

echo "Running tests..."
pytest
