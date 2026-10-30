#!/bin/bash
# Check formatting and lint without modifying files. Exits non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Checking import order (isort)..."
uv run isort --check-only --diff .

echo "Checking formatting (black)..."
uv run black --check --diff .

echo "Linting (flake8)..."
uv run flake8 .

echo "All checks passed."
