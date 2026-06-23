# =============================================================================
# File        : src/ui/gradio_app.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Gradio interactive demo.
#               -> Lets a user upload (or pick from examples) a candlestick
#                  chart PNG and minimal metadata, then displays the
#                  predicted direction, class probabilities, and the LLM
#                  explanation. Hits the FastAPI /predict endpoint over HTTP.
#
#               -> Why a separate UI process (not inlined in FastAPI):
#                    Lets the UI restart, scale, and version independently
#                    of the inference service — standard separation of UI
#                    and API tiers.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# io        : Standard library — buffer the PIL image into bytes for the
#             multipart upload to FastAPI.
# json      : Standard library — JSON-encode the metadata form field.
# os        : Standard library — reads API_BASE_URL override from env so
#             the same UI image works locally and in Kubernetes.
# pathlib   : Standard library — config path resolution.
# gradio    : Interactive UI framework — Blocks API for the form layout.
# requests  : Synchronous HTTP client for the FastAPI /predict call.
#             Sync is fine here — Gradio worker threads handle concurrency.
# yaml      : Parses serving_config.yaml.
# PIL.Image : Type hint for the uploaded image + .save() to buffer.
# =============================================================================
import io
import json
import os
from pathlib import Path

import gradio as gr
import requests
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG = REPO_ROOT / "config" / "serving_config.yaml"

def _load_cfg() -> dict:
    with SERVING_CONFIG.open() as f:
        return yaml.safe_load(f)
    

# =============================================================================
# predict_via_api
# -----------------------------------------------------------------------------
# Marshals the Gradio form values into the multipart contract /predict
# expects, then renders the response into Gradio components.
# =============================================================================
def predict_via_api(
        image : Image.Image,
        symbol : str,
        end_date : str,
        window_size : int,
        horizon : int,
        closes_csv : str,
        opens_csv :str,
) -> tuple[str, dict, str]:
    
    cfg = _load_cfg()
    api_base = os.environ.get("API_BASE_URL", cfg["ui"]["api_base_url"])

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    metadata = {
        "symbol" : symbol,
        "end_date" : end_date,
        "window_size" : int(window_size),
        "horizon" : int(horizon),
        "closes" : [float(x.strip()) for x  in closes_csv.split(",") if x.strip()],
        "opens" : [float(x.strip()) for x in opens_csv.split(",") if x.strip()],
    }

    r = requests.post(
        f"{api_base}/predict",
        files = {"image" : ("chart.png", buf, "image/png")},
        data = {"metadata" : json.dumps(metadata)},
        timeout = 30,
    )
    r.raise_for_status()
    out = r.json()
    return out["label"], out["probabilities"], out["explanation"]


# =============================================================================
# build_demo
# -----------------------------------------------------------------------------
# Constructs the Gradio Blocks layout. Kept as a function so tests can
# instantiate it without launching the server.
# =============================================================================
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Stock Chart Predictor") as demo:
        gr.Markdown("# Stock Chart Predictor\nUpload a candlestick chart window; receive a direction prediction and explanation.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Candlestick chart (224x224)")
                symbol = gr.Textbox(label="Symbol",  value="AAPL")
                end_date = gr.Textbox(label="End date (YYYY-MM-DD)", value="2025-12-31")
                window_size = gr.Number(label="Window size",  value=30, precision=0)
                horizon = gr.Number(label="Horizon (days)", value=5, precision=0)
                closes_csv = gr.Textbox(label="Closes (comma-separated, oldest first)")
                opens_csv = gr.Textbox(label="Opens  (comma-separated, oldest first)")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                label_out = gr.Label(label="Prediction")
                probs_out = gr.JSON(label="Probabilities")
                expl_out = gr.Textbox(label="Explanation", lines=4)

        btn.click(
            predict_via_api,
            inputs = [image, symbol, end_date, window_size, horizon, closes_csv, opens_csv],
            outputs = [label_out, probs_out, expl_out],
        )
    return demo


if __name__ == "__main__":
    cfg = _load_cfg()
    demo = build_demo()
    demo.launch(
        server_name = cfg["ui"]["gradio_host"],
        server_port = cfg["ui"]["gradio_port"],
    )