# =============================================================================
# File        : src/models/llm_explainer.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Natural-language explainer for chart predictions.
#               -> Given (predicted_label, confidence, candlestick stats),
#                  asks an OpenAI chat model to produce a one-paragraph
#                  rationale a human trader would find plausible.
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
# logging  : Standard library — logs LLM failures at WARNING before falling
#            back to the static text. Failures must NOT crash the API.
# os       : Standard library — reads LLM_EXPLAINER_MODEL from the env so
#            the deployed model can be swapped without a redeploy.
# Iterable : Standard library typing protocol — accepts lists, tuples, or
#            numpy arrays of prices without forcing one concrete type.
# OpenAI   : Official OpenAI Python SDK client. Reads OPENAI_API_KEY from
#            the environment automatically; can be overridden via the
#            constructor for testing.
# =============================================================================
import logging
import os
from collections.abc import Iterable

from openai import OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level client (lazy singleton)
# -----------------------------------------------------------------------------
# Initialized on first use so the module can be imported in environments
# that don't have OPENAI_API_KEY set (e.g. unit tests where the explainer
# is disabled via config). One instance is shared across all calls.
# =============================================================================
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()                          # reads OPENAI_API_KEY from env
    return _client


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


# Used when the caller doesn't pass enough price data to compute window stats
# (e.g. a Gradio demo where the user only uploads the chart and label fields).
EXPLAIN_TEMPLATE_BASIC = """You are a concise technical-analysis assistant.

A vision model has examined a {window_size}-day candlestick chart for
ticker {symbol} ending {end_date} and predicted "{prediction}" (model
confidence {confidence:.0%}) for the {horizon}-day forward direction.

Write 2-3 sentences in plain English explaining what type of candlestick
pattern typically supports this prediction in a {window_size}-day window.
Do not hedge with "may", "might", or disclaimers. Do not output bullet
points or headings."""


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
#   model        : str    OpenAI model id (e.g. "gpt-4o-mini"). Defaults to
#                         LLM_EXPLAINER_MODEL env var, then "gpt-4o-mini".
#   temperature  : float  Low (0.2) for stable phrasing across runs.
#   max_tokens   : int    Response cap (200 keeps it to a paragraph).
#   fallback_text: str    Returned if the OpenAI call fails for any reason.
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
    model         : str | None = None,
    temperature   : float = 0.2,
    max_tokens    : int   = 200,
    fallback_text : str   = "Explanation unavailable.",
) -> str:

    model = model or os.environ.get("LLM_EXPLAINER_MODEL", "gpt-4o-mini")

    closes_list = list(closes)
    opens_list  = list(opens)

    # Pick the rich template only when we have enough data to fill its slots.
    # The basic template asks the LLM to reason from the label alone, so the
    # API still produces a sensible explanation when the caller (e.g. the
    # Gradio demo) leaves the price fields empty.
    has_prices = len(closes_list) >= 2 and len(opens_list) >= 1

    try:
        if has_prices:
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
        else:
            filled = EXPLAIN_TEMPLATE_BASIC.format(
                symbol      = symbol,
                end_date    = end_date,
                prediction  = prediction,
                confidence  = confidence,
                window_size = window_size,
                horizon     = horizon,
            )

        result = _get_client().chat.completions.create(
            model                 = model,
            messages              = [{"role": "user", "content": filled}],
            temperature           = temperature,
            max_completion_tokens = max_tokens,    # OpenAI replaced max_tokens for newer models
        )
        content = result.choices[0].message.content
        return content.strip() if content else fallback_text
    except Exception as e:                          # noqa: BLE001
        logger.warning("OpenAI explainer failed: %s — using fallback", e)
        return fallback_text
