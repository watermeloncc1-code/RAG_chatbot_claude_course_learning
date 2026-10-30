#!/bin/bash
# Run all quality checks, then the offline test suite. Extra args go to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/lint.sh

echo "Running tests (excluding live API tests)..."
uv run pytest -m 'not live' "$@"
