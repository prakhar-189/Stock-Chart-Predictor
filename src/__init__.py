# src/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Top-level package initializer for the
#               stock-chart-predictor project.
#               Makes the `src` directory a Python package so that
#               every module is importable via a fully-qualified path:
#                   from src.data.load_ohlcv     import load_and_filter
#                   from src.models.vision_model import build_model
#                   from src.api.main            import app
#
# Subpackages in this project:
#   data      : OHLCV loading, sliding-window labeling, chart
#               rendering, train/val/test manifest building.
#   models    : ViT chart classifier + LLM rationale explainer.
#   training  : Manifest-driven Dataset, training loop, evaluation.
#   inference : Singleton predictor + orchestrated prediction pipeline.
#   api       : FastAPI app, routes, schemas, request middleware.
#   ui        : Gradio interactive demo that talks to the FastAPI app.
# --------------------------------------------------------------