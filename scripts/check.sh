#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m compileall -q src scripts tests
uvx --from ruff==0.12.11 ruff check src scripts tests
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python3 scripts/fixture_smoke.py
for receipt in receipts/redacted/*.json receipts/calibration/*.json receipts/failures/*.json; do
  PYTHONPATH=src python3 -m ax0s_bench.cli verify --receipt "$receipt"
done
PYTHONPATH=src uv run --no-project --with jsonschema==4.25.1 python scripts/validate_public_receipts.py
