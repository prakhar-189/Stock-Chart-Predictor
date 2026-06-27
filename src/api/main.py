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
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

# Load .env at module import so OPENAI_API_KEY (and any other override) is
# visible to the predictor and explainer. In Docker / Kubernetes the env
# vars come from the orchestrator and load_dotenv is a no-op.
load_dotenv()

# Quiet noisy third-party loggers so the startup output is clean. HF hub auth
# warnings, transformers advisory warnings, every httpx HEAD probe, and HF
# progress bars all clutter the screen without adding signal. Set BEFORE the
# predictor imports so transformers picks up the verbosity env vars at load.
os.environ.setdefault("TRANSFORMERS_VERBOSITY",            "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS",      "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
# Suppress the HF_TOKEN advisory that's emitted via warnings.warn() (not the
# logging module) — it slips past the loggers above.
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

from src.api.middleware import RequestIDMiddleware                    # noqa: E402, I001
from src.api.routes import router                                     # noqa: E402, I001
from src.inference.predictor import get_predictor                     # noqa: E402, I001

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