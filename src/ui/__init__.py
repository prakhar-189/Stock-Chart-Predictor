# src/ui/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'ui' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from src.ui.gradio_app import build_demo
#
# Modules in this package:
#   gradio_app.py : Gradio Blocks demo. Lets a user upload a
#                   candlestick chart PNG plus minimal metadata,
#                   then displays the predicted direction, class
#                   probabilities, and the LLM explanation by
#                   calling the FastAPI /predict endpoint over HTTP.
#                   Runs as its own process so the UI and inference
#                   tiers scale independently.
# --------------------------------------------------------------