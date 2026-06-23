# =============================================================================
# File        : src/api/schemas.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Pydantic request / response models for the FastAPI app.
#               -> Centralizes the API contract so OpenAPI docs, validation,
#                  and type hints stay in lockstep.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# BaseModel : Pydantic base class — gives free JSON validation, OpenAPI
#             schema generation, and pythonic attribute access.
# Field     : Per-field constraints (default values, descriptions). The
#             `description` strings show up in the auto-generated /docs UI.
# =============================================================================
from pydantic import BaseModel, Field


# =============================================================================
# PredictRequest
# -----------------------------------------------------------------------------
# Submitted by clients along with a multipart image. Metadata is needed by
# the LLM explainer to write a context-aware rationale.
# =============================================================================
class PredictRequest(BaseModel):
    symbol      : str          = Field(..., description="Ticker symbol, e.g. AAPL")
    end_date    : str          = Field(..., description="YYYY-MM-DD end of the chart window")
    window_size : int          = Field(30,  description="Number of candles in the window")
    horizon     : int          = Field(5,   description="Forward days the label represents")
    closes      : list[float]  = Field(...,  description="Window closing prices, oldest first")
    opens       : list[float]  = Field(...,  description="Window opening prices, oldest first")


# =============================================================================
# PredictResponse
# =============================================================================
class PredictResponse(BaseModel):
    label         : str
    confidence    : float
    probabilities : dict[str, float]
    explanation   : str


# =============================================================================
# HealthResponse
# =============================================================================
class HealthResponse(BaseModel):
    status      : str
    model_loaded: bool