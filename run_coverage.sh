#!/usr/bin/env bash
set -e

echo "=== Calculating Code Coverage for WMilvus ==="
PYTHONPATH=src pytest --cov=wmilvus --cov-report=term-missing --cov-report=html tests/
echo "=== Coverage Report Generated in htmlcov/index.html ==="
