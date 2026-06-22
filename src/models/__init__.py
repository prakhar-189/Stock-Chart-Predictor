# src/models/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'models' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from src.models.vision_model  import build_model, load_checkpoint
#                   from src.models.llm_explainer import explain_prediction
#
# Modules in this package:
#   vision_model.py  : ViT-base wrapper (HuggingFace transformers).
#                      Loads the backbone declared in params.yaml,
#                      attaches a Dropout + Linear 3-class head, and
#                      exposes a shared ImageProcessor so training
#                      and inference use identical preprocessing.
#   llm_explainer.py : Generates a one-paragraph natural-language
#                      rationale for each chart prediction via
#                      LiteLLM. Falls back to a static string on
#                      failure so the deterministic prediction
#                      always ships.
# --------------------------------------------------------------