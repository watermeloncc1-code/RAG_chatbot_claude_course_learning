#!/bin/bash
# Auto-format all Python code (sorts imports, then applies black).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Sorting imports with isort..."
uv run isort .

echo "Formatting with black..."
uv run black .
