# =============================================================================
# File        : src/api/routes.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> FastAPI route handlers.
#               -> Defines /healthz, /predict, and /metrics endpoints.
#               -> Image upload is via multipart/form-data; metadata is sent
#                  as a JSON-encoded form field to keep the contract simple
#                  for Gradio and curl users alike.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# io               : Standard library — wraps upload bytes in a buffer so
#                    PIL can decode them without a temp file.
# json             : Standard library — parses the metadata form field.
# logging          : Standard library — endpoint-level logs.
# APIRouter        : FastAPI route grouping; mounted by main.py.
# File / Form      : FastAPI multipart parameter decorators for the
#                    image upload and the JSON-encoded metadata field.
# HTTPException    : FastAPI's typed error — converts to a proper 4xx/5xx
#                    response instead of a 500 stack trace.
# UploadFile       : Streaming file upload interface — does NOT load the
#                    full body into memory until .read() is called.
# PIL.Image        : Decodes the uploaded chart bytes.
# HealthResponse,
# PredictRequest,
# PredictResponse  : Local — pydantic contracts.
# run_pipeline     : Local — orchestrates vision predict + LLM explain.
# get_predictor    : Local — checks model_loaded for /healthz.
# =============================================================================
import io
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.inference.pipeline import run as run_pipeline
from src.inference.predictor import get_predictor


logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# /healthz
# -----------------------------------------------------------------------------
# Kubernetes liveness probe. Returns 200 + model_loaded flag so a readiness
# gate can differentiate "process up" from "model warm".
# =============================================================================
@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    try:
        get_predictor()
        loaded = True
    except Exception as e:                              # noqa: BLE001
        logger.warning("predictor not ready: %s", e)
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


# =============================================================================
# /predict
# -----------------------------------------------------------------------------
# Multipart endpoint:
#   image    : the rendered chart PNG (uploaded file).
#   metadata : JSON string matching PredictRequest schema.
# =============================================================================
@router.post("/predict", response_model=PredictResponse)
async def predict(
    image    : UploadFile = File(...),
    metadata : str        = Form(...),
) -> PredictResponse:
    try:
        meta = PredictRequest(**json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid metadata: {e}") from e

    img_bytes = await image.read()
    try:
        pil = Image.open(io.BytesIO(img_bytes))
    except Exception as e:                              # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {e}") from e

    result = run_pipeline(
        image = pil,
        symbol = meta.symbol,
        end_date = meta.end_date,
        window_size = meta.window_size,
        horizon = meta.horizon,
        closes = meta.closes,
        opens = meta.opens,
    )
    return PredictResponse(**result)