# =============================================================================
# File        : src/api/middleware.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> FastAPI middleware: request ID injection and access logging.
#               -> Each request gets a UUID written to the response header
#                  X-Request-ID and embedded in log records, so a Grafana /
#                  CloudWatch query can stitch a single user's trace together.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging              : Standard library — dedicated "api.access" logger so
#                        access logs can be routed separately from app logs.
# time                 : Standard library — wall-clock timing for request
#                        duration (milliseconds).
# uuid                 : Standard library — fresh UUID4 hex for the
#                        X-Request-ID header when the client doesn't send one.
# BaseHTTPMiddleware   : Starlette base class — implement dispatch() to wrap
#                        the request/response lifecycle.
# Request / Response   : Starlette types for the dispatch contract.
# =============================================================================
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


# =============================================================================
# RequestIDMiddleware
# -----------------------------------------------------------------------------
# If the client passes X-Request-ID, we honor it (useful for distributed
# tracing). Otherwise we mint a fresh UUID4.
# =============================================================================
class RequestIDMiddleware(BaseHTTPMiddleware):

    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(self.HEADER) or uuid.uuid4().hex
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request %s failed: %s %s", rid, request.method, request.url.path)
            raise
        elapsed_ms = (time.time() - start) * 1000.0
        response.headers[self.HEADER] = rid
        logger.info("%s %s %s -> %d (%.1f ms) rid=%s",
                    request.client.host if request.client else "-",
                    request.method, request.url.path,
                    response.status_code, elapsed_ms, rid)
        return response