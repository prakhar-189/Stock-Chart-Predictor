# =============================================================================
# File        : tests/test_api.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> FastAPI smoke tests using starlette's TestClient.
#               -> Asserts /healthz returns 200 even when the model has not
#                  loaded (model_loaded=False is the documented contract).
# =============================================================================


from fastapi.testclient import TestClient

from src.api.main import app


def test_healthz_returns_ok():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body