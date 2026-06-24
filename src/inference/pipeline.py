# =============================================================================
# File        : src/inference/pipeline.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> End-to-end inference orchestrator.
#               -> Combines the vision predictor with the optional LLM
#                  explainer into a single `run(image, metadata)` call used
#                  by the API and UI layers.
#
#               -> Why a separate pipeline module:
#                    Keeps `predictor.py` deterministic and unit-testable
#                    in isolation, while the orchestration (which can fail
#                    or be disabled by config) lives here.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging              : Standard library — pipeline-level diagnostics.
# pathlib              : Standard library — locate model_config.yaml.
# Iterable             : Standard library typing — accept list / tuple / numpy
#                        arrays of prices uniformly.
# yaml                 : Parses the `explainer` block of model_config.yaml.
# PIL.Image            : Type hint for the input image; not used directly.
# explain_prediction   : Local — the LLM-driven rationale step (non-blocking
#                        in the sense that failure falls back to a static
#                        string, so the deterministic prediction always ships).
# get_predictor        : Local — warm-loaded singleton for the ViT predictor.
# =============================================================================
import logging
from collections.abc import Iterable
from pathlib import Path

import yaml
from PIL import Image

from src.inference.predictor import get_predictor
from src.models.llm_explainer import explain_prediction

logger = logging.getLogger(__name__)

REPO_ROOT          = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH  = REPO_ROOT / "config" / "model_config.yaml"


def _load_explainer_cfg() -> dict:
    with MODEL_CONFIG_PATH.open() as f:
        return yaml.safe_load(f).get("explainer", {})


# =============================================================================
# run
# -----------------------------------------------------------------------------
# Parameters:
#   image       : PIL Image of the candlestick chart.
#   symbol      : str   Ticker (for the explainer prompt).
#   end_date    : str   YYYY-MM-DD end of the window.
#   window_size : int   Candles in the window.
#   horizon     : int   Forward days the label represents.
#   closes      : Iterable[float]  Window's closing prices, oldest first.
#   opens       : Iterable[float]  Window's opening prices.
#
# Returns:
#   dict : {
#       "label"         : str,
#       "confidence"    : float,
#       "probabilities" : dict[str, float],
#       "explanation"   : str,
#   }
# =============================================================================
def run(
    image       : Image.Image,
    symbol      : str,
    end_date    : str,
    window_size : int,
    horizon     : int,
    closes      : Iterable[float],
    opens       : Iterable[float],
) -> dict:

    predictor = get_predictor()
    prediction = predictor.predict(image)

    explainer_cfg = _load_explainer_cfg()
    if explainer_cfg.get("enabled", False):
        explanation = explain_prediction(
            prediction    = prediction["label"],
            confidence    = prediction["confidence"],
            symbol        = symbol,
            end_date      = end_date,
            window_size   = window_size,
            horizon       = horizon,
            closes        = closes,
            opens         = opens,
            model         = explainer_cfg.get("model"),
            temperature   = explainer_cfg.get("temperature", 0.2),
            max_tokens    = explainer_cfg.get("max_tokens", 200),
            fallback_text = explainer_cfg.get("fallback_text", "Explanation unavailable."),
        )
    else:
        explanation = ""

    return {**prediction, "explanation": explanation}