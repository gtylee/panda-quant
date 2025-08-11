#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "Running unit tests..."
$PYTHON -m unittest discover -s tests -p 'test_*.py'

echo "Running legacy root-level tests (if any)..."
$PYTHON -m unittest discover -s . -p 'test_*.py'

echo "All tests completed."


