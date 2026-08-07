#!/usr/bin/env bash
# One paste-safe command that reproduces vol-lab's verification end to end.
# Usage:  bash scripts/verify.sh        (from the repo root, with the venv created)
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> ruff"
ruff check .

echo "==> mypy"
mypy

echo "==> tests + coverage (>=90% gate)"
coverage run -m pytest
coverage report --fail-under=90

echo "==> surface / verification report (every statistic)"
python scripts/report_surface.py --all-snapshots

echo "==> figures"
python scripts/make_figures.py

echo ""
echo "verify: OK — ruff + mypy clean, coverage >=90%, stats + figures regenerated."
