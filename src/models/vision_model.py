# =============================================================================
# File        : src/models/vision_model.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Vision Transformer wrapper for 3-class chart classification.
#               -> Thin facade around HuggingFace's ViTForImageClassification
#                  that:
#                      - Loads the backbone declared in params.yaml
#                      - Replaces the classifier head with `num_classes=3`
#                      - Exposes a clean (image_tensor -> logits) forward
#
#               -> Why ViT-base over a CNN:
#                    Self-attention captures non-local relationships across
#                    the candle sequence (e.g. a head-and-shoulders pattern
#                    has distant peaks). A CNN's small receptive field
#                    requires deeper stacks to see the same context.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging                    : Standard library — model construction diagnostics.
# pathlib                    : Standard library — locate params.yaml.
# torch                      : Tensor IO, checkpoint load, device placement.
# torch.nn                   : Replacement classifier head (Dropout + Linear)
#                              stacked on top of the ViT pooled output.
# yaml                       : Parses params.yaml for backbone + dropout.
# ViTForImageClassification  : HuggingFace transformers — ViT backbone with a
#                              configurable classification head.
# ViTImageProcessor          : Matching preprocessing pipeline (resize +
#                              ImageNet normalization). Shared between
#                              training Dataset and inference Predictor so
#                              the transform contract is identical end-to-end.
# =============================================================================
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import ViTForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"

LABELS = ["up", "sideways", "down"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS)}


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


# =============================================================================
# build_model
# -----------------------------------------------------------------------------
# Instantiates the ViT classifier with the correct head size and label maps.
# `ignore_mismatched_sizes=True` lets us re-initialize the 1000-class
# ImageNet head into a 3-class chart head without a manual surgery step.
# =============================================================================
def build_model(backbone: str | None = None, dropout: float | None = None) -> nn.Module:
    params   = load_params()["model"]
    backbone = backbone or params["backbone"]
    dropout  = dropout  if dropout is not None else params["dropout"]

    logger.info("building ViT from backbone=%s with dropout=%.2f", backbone, dropout)

    model = ViTForImageClassification.from_pretrained(
        backbone,
        num_labels               = len(LABELS),
        id2label                 = ID2LABEL,
        label2id                 = LABEL2ID,
        ignore_mismatched_sizes  = True,
    )

    # The default head is a single Linear; we wrap it with dropout to combat
    # the relatively small chart dataset overfitting quickly.
    # HF's typing for `model.classifier` resolves through nn.Module.__getattr__
    # (returns `int | Tensor | Module`), so mypy can't verify either the
    # Sequential swap or the in_features arg. Both ignores below match the
    # documented HF head-replacement pattern; runtime contract is rock-solid.
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(dropout),
        nn.Linear(in_features, len(LABELS)),  # type: ignore[arg-type]
    )

    return model


# =============================================================================
# build_image_processor
# -----------------------------------------------------------------------------
# Returns the matching processor (normalization + resize) for the backbone.
# Used by both the training Dataset and the inference predictor so the
# transform contract is identical end-to-end.
# =============================================================================
def build_image_processor(backbone: str | None = None) -> ViTImageProcessor:
    params   = load_params()["model"]
    backbone = backbone or params["backbone"]
    return ViTImageProcessor.from_pretrained(backbone)


# =============================================================================
# load_checkpoint
# -----------------------------------------------------------------------------
# Restores model weights for inference. Strict=True so a silent schema
# mismatch fails loudly rather than producing garbage predictions.
# =============================================================================
def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    model = build_model()
    # mmap=True memory-maps the checkpoint instead of loading it into a single
    # contiguous RAM block. Prevents allocation failures on fragmented memory
    # (common after long Windows uptime) and lowers peak resident memory.
    state = torch.load(checkpoint_path, map_location=device, mmap=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model