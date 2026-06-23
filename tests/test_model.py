# =============================================================================
# File        : tests/test_model.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Smoke tests for the ViT wrapper.
#               -> Builds the model, runs a single forward pass on random
#                  pixels, asserts the output shape matches the 3-class head.
#                  No training, no downloads beyond first-run cache.
# =============================================================================


import pytest
import torch

from src.models.vision_model import LABELS, build_model


@pytest.mark.slow
def test_build_model_forward_shape():
    model = build_model()
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=torch.randn(2, 3, 224, 224)).logits
    assert logits.shape == (2, len(LABELS))