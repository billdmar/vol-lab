# vol-lab — common developer tasks. Run `make help` for the list.
# Assumes the project venv is active (see docs/ENV.md): `pip install -e ".[dev]"`.

.PHONY: help install lint typecheck test coverage verify figures report snapshot clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create/refresh the editable install with dev extras
	pip install -e ".[dev]"

lint:  ## Ruff lint
	ruff check .

typecheck:  ## mypy type-check (src/)
	mypy

test:  ## Run the test suite
	pytest

coverage:  ## Tests with coverage + the >=90% engine gate
	coverage run -m pytest
	coverage report --fail-under=90

verify:  ## Full verification (ruff + mypy + coverage gate + report + figures)
	bash scripts/verify.sh

figures:  ## Regenerate all showcase figures into docs/figures/
	python scripts/make_figures.py

report:  ## Print the surface/verification report for every committed snapshot
	python scripts/report_surface.py --all-snapshots

snapshot:  ## Collect one fresh Deribit snapshot (live public API; skips same-UTC-day)
	python scripts/collect_snapshot.py

clean:  ## Remove caches and build artifacts (keeps fixtures + figures)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis .coverage coverage.xml \
	       htmlcov *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
