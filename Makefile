# =============================================================================
# File        : Makefile
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Developer-facing convenience targets.
#               -> Thin wrappers around the canonical commands (pytest, dvc,
#                  docker, ruff) so contributors don't have to remember flags.
#
#               -> Why a Makefile when DVC already exists:
#                    Make targets bundle multi-step actions (`make dev`, `make
#                    lint`, `make docker-up`) and serve as live documentation
#                    of the project's verbs. DVC owns the data/model DAG;
#                    Make owns developer ergonomics.
# =============================================================================

PYTHON       := python
PIP          := pip
PKG          := stock_chart_predictor
SRC_DIR      := src
TESTS_DIR    := tests

.PHONY: help install dev lint format test repro train serve docker-up docker-down clean

help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:                    ## Install runtime dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev:                        ## Install dev + runtime dependencies
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install || true

lint:                       ## Run ruff + mypy
	ruff check $(SRC_DIR) $(TESTS_DIR)
	mypy $(SRC_DIR)

format:                     ## Auto-format with ruff
	ruff format $(SRC_DIR) $(TESTS_DIR)
	ruff check --fix $(SRC_DIR) $(TESTS_DIR)

test:                       ## Run the test suite
	pytest $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

repro:                      ## Re-run the full DVC pipeline
	dvc repro

train:                      ## Run only the training stage
	dvc repro train

serve:                      ## Launch the FastAPI inference server locally
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:                         ## Launch the Gradio demo UI
	$(PYTHON) -m src.ui.gradio_app

docker-up:                  ## Build and start the full stack
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:                ## Stop the stack
	docker compose -f docker/docker-compose.yml down

clean:                      ## Remove caches, build artifacts, and __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info