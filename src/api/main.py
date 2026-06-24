# =============================================================================
# File        : src/api/main.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> FastAPI application factory.
#               -> Wires routes, middleware, CORS, and Prometheus metrics.
#               -> Uses the lifespan hook to warm the predictor singleton at
#                  startup so the first user request is fast.
#
#               -> Launch:
#                    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging              : Standard library — startup diagnostics.
# asynccontextmanager  : Standard library — decorator for the FastAPI
#                        lifespan hook (replaces the old @app.on_event).
# pathlib              : Standard library — locate serving_config.yaml.
# yaml                 : Parses serving_config.yaml.
# FastAPI              : ASGI app + automatic OpenAPI / Swagger docs.
# CORSMiddleware       : Browser-friendly cross-origin headers; settings come
#                        from serving_config.yaml.
# make_asgi_app        : prometheus_client — turns the Prometheus registry
#                        into a mountable ASGI app so /metrics is scrapable.
# RequestIDMiddleware  : Local — request ID injection + access logging.
# router               : Local — endpoint registrations.
# get_predictor        : Local — invoked at startup to pre-warm the model
#                        and avoid a 5s cold start on the first /predict.
# =============================================================================
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.middleware import RequestIDMiddleware
from src.api.routes import router
from src.inference.predictor import get_predictor

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG = REPO_ROOT / "config" / "serving_config.yaml"


def _load_serving_cfg() -> dict:
    with SERVING_CONFIG.open() as f:
        return yaml.safe_load(f)


# =============================================================================
# lifespan
# -----------------------------------------------------------------------------
# Pre-loads the model at startup so the first /predict isn't a 5s cold start.
# Logs any failure but does NOT crash the process — /healthz will surface
# the unloaded state via `model_loaded=False`.
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_predictor()
        logging.getLogger(__name__).info("predictor warmed at startup")
    except Exception:                                     # noqa: BLE001
        logging.getLogger(__name__).exception("predictor warmup failed")
    yield


def create_app() -> FastAPI:
    cfg = _load_serving_cfg()

    logging.basicConfig(
        level = cfg["server"]["log_level"].upper(),
        format = cfg["observability"]["log_format"],
    )

    app = FastAPI(
        title = "stock-chart-predictor",
        version = "0.1.0",
        lifespan = lifespan,
    )

    cors_cfg = cfg["cors"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins = cors_cfg["allow_origins"],
        allow_methods = cors_cfg["allow_methods"],
        allow_headers = cors_cfg["allow_headers"],
        allow_credentials = cors_cfg["allow_credentials"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(router)
    app.mount(cfg["observability"]["metrics_path"], make_asgi_app())
    return app


app = create_app()