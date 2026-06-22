# src/inference/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'inference' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from src.inference.predictor import get_predictor
#                   from src.inference.pipeline  import run
#
# Modules in this package:
#   predictor.py : Singleton vision-only predictor. Loads the
#                  checkpoint once (thread-safe lazy init), exposes
#                  a `predict(image) -> dict` returning label,
#                  confidence, and per-class probabilities.
#   pipeline.py  : Orchestrator. Combines `predictor.predict` with
#                  the optional `llm_explainer` step into a single
#                  `run(image, metadata) -> dict` used by the
#                  FastAPI routes and the Gradio UI.
# --------------------------------------------------------------