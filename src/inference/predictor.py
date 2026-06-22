# =============================================================================
# File        : src/inference/predictor.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Single-image prediction wrapper.
#               -> Loads the saved checkpoint once (singleton pattern) and
#                  exposes a `predict(image) -> dict` for the FastAPI app
#                  and Gradio UI to call. Returns label, probabilities,
#                  and the top-class confidence.
#
#               -> Why a singleton:
#                    Model loading is the slowest step (~5s for ViT-base).
#                    Loading per-request would hammer cold-start latency.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging                  : Standard library — singleton-load diagnostics.
# pathlib                  : Standard library — config and checkpoint paths.
# Lock                     : Standard library — guards the double-checked
#                            locking pattern used in the singleton accessor.
# torch                    : Inference no_grad context, softmax, device move,
#                            optional fp16 cast.
# yaml                     : Parses config/model_config.yaml at first use.
# PIL.Image                : Decodes inbound bytes from FastAPI / Gradio.
# ID2LABEL,
# build_image_processor,
# load_checkpoint          : Local — shared label maps, processor, and the
#                            strict-loading checkpoint helper. Keeps the
#                            inference transform IDENTICAL to training.
# =============================================================================
import logging
from pathlib import Path
from threading import Lock

import torch
import yaml
from PIL import Image

from src.models.vision_model import ID2LABEL, build_image_processor, load_checkpoint


logger = logging.getLogger(__name__)


REPO_ROOT          = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH  = REPO_ROOT / "config" / "model_config.yaml"


# =============================================================================
# _Predictor
# -----------------------------------------------------------------------------
# Thread-safe lazy-loaded predictor singleton. The first call to
# `get_predictor()` reads the config, picks the device, and loads weights;
# subsequent calls return the cached instance.
# =============================================================================
class _Predictor:
    def __init__(self) -> None:
        with MODEL_CONFIG_PATH.open() as f:
            self.cfg = yaml.safe_load(f)

        ckpt_cfg = self.cfg["checkpoint"]
        device   = ckpt_cfg["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt_path = REPO_ROOT / ckpt_cfg["path"]
        logger.info("loading checkpoint %s on %s", ckpt_path, device)
        self.model     = load_checkpoint(ckpt_path, device=device)
        self.processor = build_image_processor()

        if ckpt_cfg.get("half_precision") and device == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        image    = image.convert("RGB")
        encoded  = self.processor(images=image, return_tensors="pt").to(self.device)
        logits   = self.model(**encoded).logits
        probs    = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.tensor(probs).argmax().item())
        return {
            "label"         : ID2LABEL[pred_idx],
            "confidence"    : float(probs[pred_idx]),
            "probabilities" : {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        }


# =============================================================================
# Module-level singleton accessor
# =============================================================================
_instance: _Predictor | None = None
_instance_lock = Lock()


def get_predictor() -> _Predictor:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = _Predictor()
    return _instance