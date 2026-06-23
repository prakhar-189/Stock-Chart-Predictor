# src/api/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'api' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from src.api.main       import app
#                   from src.api.routes     import router
#                   from src.api.schemas    import PredictRequest, PredictResponse
#                   from src.api.middleware import RequestIDMiddleware
#
# Modules in this package:
#   main.py       : FastAPI app factory. Wires routes, middleware,
#                   CORS, Prometheus /metrics, and uses the lifespan
#                   hook to pre-warm the predictor singleton so the
#                   first request doesn't pay the 5s cold-start cost.
#   routes.py     : Endpoint handlers — /healthz (liveness) and
#                   /predict (multipart image + JSON metadata).
#   schemas.py    : Pydantic request/response models. Centralizes
#                   the public API contract so OpenAPI docs and
#                   validation stay in lockstep.
#   middleware.py : RequestID injection (honored if the client sends
#                   X-Request-ID, otherwise a fresh UUID4) plus
#                   structured access logging with request latency.
# --------------------------------------------------------------