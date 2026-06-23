# tests/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'tests' module.
#               Marks the test directory as a Python package so
#               pytest can collect tests and resolve imports
#               consistently across local runs and CI.
#               Most fixtures live in conftest.py, which pytest
#               discovers automatically — no manual import needed.
#
# Modules in this package:
#   conftest.py      : Shared pytest fixtures (synthetic OHLCV
#                      frames, temp dirs, etc.). Auto-discovered
#                      by pytest at collection time.
#   test_data.py     : Data-layer unit tests — schema normalization,
#                      validation, sliding-window math, label
#                      thresholds.
#   test_model.py    : ViT smoke test — builds the model, runs a
#                      single forward pass on random pixels,
#                      asserts the output shape.
#   test_api.py      : FastAPI smoke tests using the Starlette
#                      TestClient — asserts /healthz returns 200.
#   test_pipeline.py : End-to-end data-layer pipeline tests on
#                      synthetic data — verifies contracts between
#                      stages (column names, dtypes, time order).
# --------------------------------------------------------------