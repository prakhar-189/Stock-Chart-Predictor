# =============================================================================
# File        : src/models/llm_explainer.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Natural-language explainer for chart predictions.
#               -> Given (predicted_label, confidence, candlestick stats),
#                  asks an LLM to produce a one-paragraph rationale a human
#                  trader would find plausible.
#
#               -> Pipeline position - Inference enrichment.
#                  Called by src/inference/pipeline.py after the ViT scores
#                  an image. Output is attached to the API response.
#
#               -> Why a separate module:
#                    Decouples the deterministic vision pipeline from the
#                    non-deterministic LLM step. The vision model is the
#                    source of truth for the prediction; the LLM only
#                    decorates it. Failures here degrade gracefully to a
#                    static fallback string.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging    : Standard library — logs LLM failures at WARNING before falling
#              back to the static text. Failures must NOT crash the API.
# os         : Standard library — reads LLM_EXPLAINER_MODEL from the env so
#              the deployed model can be swapped without a redeploy.
# Iterable   : Standard library typing protocol — accepts lists, tuples, or
#              numpy arrays of prices without forcing one concrete type.
# completion : LiteLLM's unified chat-completion function. Same call used in
#              the LLM Regressor MLOps project (judge.py) — provider-agnostic,
#              swap the `model` kwarg to switch from OpenAI to Anthropic, etc.
# =============================================================================
import logging
import os
from typing import Iterable

from litellm import completion


logger = logging.getLogger(__name__)


# =============================================================================
# EXPLAIN_TEMPLATE
# -----------------------------------------------------------------------------
# Strict prompt: tell the LLM exactly the shape of the answer (one paragraph,
# 2-3 sentences, no hedging boilerplate). Keep it short — this is a
# decoration, not the prediction itself.
# =============================================================================
EXPLAIN_TEMPLATE = """You are a concise technical-analysis assistant.

A vision model has examined a {window_size}-day candlestick chart for
ticker {symbol} ending {end_date} and predicted "{prediction}" (model
confidence {confidence:.0%}) for the {horizon}-day forward direction.

Recent candle statistics over the window:
- open  range : {open_min:.2f} - {open_max:.2f}
- close range : {close_min:.2f} - {close_max:.2f}
- last close  : {last_close:.2f}
- first close : {first_close:.2f}
- raw return  : {window_return:+.2%}

Write 2-3 sentences in plain English explaining what pattern in the recent
price action could justify the predicted direction. Do not hedge with
"may", "might", or disclaimers. Do not output bullet points or headings."""


# =============================================================================
# explain_prediction
# -----------------------------------------------------------------------------
# Parameters:
#   prediction   : str    One of "up" | "sideways" | "down".
#   confidence   : float  Model softmax for the predicted class, [0, 1].
#   symbol       : str    Ticker symbol, e.g. "AAPL".
#   end_date     : str    YYYY-MM-DD end of the chart window.
#   window_size  : int    Candles in the chart.
#   horizon      : int    Forward days the label corresponds to.
#   closes       : Iterable[float]  Closing prices in the window (oldest first).
#   opens        : Iterable[float]  Opening prices in the window.
#   model        : str    LiteLLM-supported model id (e.g. "gpt-4o-mini").
#   temperature  : float  Low (0.2) for stable phrasing across runs.
#   max_tokens   : int    Response cap (200 keeps it to a paragraph).
#   fallback_text: str    Returned if the LLM call fails for any reason.
#
# Returns:
#   str : the explanation, or `fallback_text` on failure.
# =============================================================================
def explain_prediction(
    prediction    : str,
    confidence    : float,
    symbol        : str,
    end_date      : str,
    window_size   : int,
    horizon       : int,
    closes        : Iterable[float],
    opens         : Iterable[float],
    model         : str   = None,
    temperature   : float = 0.2,
    max_tokens    : int   = 200,
    fallback_text : str   = "Explanation unavailable.",
) -> str:

    model = model or os.environ.get("LLM_EXPLAINER_MODEL", "gpt-4o-mini")

    closes_list = list(closes)
    opens_list  = list(opens)

    filled = EXPLAIN_TEMPLATE.format(
        symbol        = symbol,
        end_date      = end_date,
        prediction    = prediction,
        confidence    = confidence,
        window_size   = window_size,
        horizon       = horizon,
        open_min      = min(opens_list),
        open_max      = max(opens_list),
        close_min     = min(closes_list),
        close_max     = max(closes_list),
        last_close    = closes_list[-1],
        first_close   = closes_list[0],
        window_return = (closes_list[-1] / closes_list[0]) - 1.0,
    )

    try:
        result = completion(
            model       = model,
            messages    = [{"role": "user", "content": filled}],
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return result.choices[0].message.content.strip()
    except Exception as e:                          # noqa: BLE001
        logger.warning("LLM explainer failed: %s — using fallback", e)
        return fallback_text