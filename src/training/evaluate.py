# =============================================================================
# File        : src/training/evaluate.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Scores the best checkpoint on the held-out test set.
#               -> Computes accuracy, per-class precision / recall / F1,
#                  and a 3x3 confusion matrix. Writes both a JSON summary
#                  (metrics/test_metrics.json) and a CSV confusion matrix
#                  (metrics/confusion_matrix.csv) consumed by DVC plots.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# json                       : Standard library — writes test_metrics.json
#                              consumed by DVC plots.
# logging                    : Standard library — eval-time logs.
# sys                        : Standard library — exit code from main().
# pathlib                    : Standard library — checkpoint + metric paths.
# numpy                      : Confusion-matrix accumulation and macro-F1.
# pandas                     : Save confusion matrix as a labeled CSV.
# torch                      : Inference forward pass under no_grad().
# yaml                       : Parses batch_size / num_workers from params.yaml.
# DataLoader                 : Test-set mini-batch iteration.
# LABELS, load_checkpoint    : Local — shared label order + strict checkpoint
#                              loader from the vision_model module.
# ChartWindowDataset         : Local — Dataset over the test-split manifest.
# =============================================================================
import json
import logging
import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.vision_model  import LABELS, load_checkpoint
from src.training.dataset     import ChartWindowDataset


logger = logging.getLogger(__name__)


REPO_ROOT   = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"
SPLITS_DIR  = REPO_ROOT / "data"   / "splits"
MODELS_DIR  = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


# =============================================================================
# _per_class_metrics
# -----------------------------------------------------------------------------
# Computes precision, recall, and F1 per class from a confusion matrix.
# `cm[i, j]` = count of examples with true=i predicted=j.
# =============================================================================
def _per_class_metrics(cm: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for i, name in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def main() -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    params = load_params()["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_checkpoint(MODELS_DIR / "best.pt", device=device)

    test_ds     = ChartWindowDataset(SPLITS_DIR / "test.csv")
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"],
                             shuffle=False, num_workers=params["num_workers"])

    n_classes = len(LABELS)
    cm        = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct   = 0
    total     = 0

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)
            preds        = model(pixel_values=pixel_values).logits.argmax(dim=-1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy   = correct / total
    per_class  = _per_class_metrics(cm, LABELS)
    macro_f1   = float(np.mean([per_class[c]["f1"] for c in LABELS]))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "test_metrics.json").open("w") as f:
        json.dump({"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}, f, indent=2)

    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                             columns=[f"pred_{l}" for l in LABELS])
    cm_df.to_csv(METRICS_DIR / "confusion_matrix.csv")

    logger.info("accuracy=%.4f macro_f1=%.4f", accuracy, macro_f1)
    for cls, m in per_class.items():
        logger.info("  %-9s P=%.3f R=%.3f F1=%.3f", cls, m["precision"], m["recall"], m["f1"])

    return 0


if __name__ == "__main__":
    sys.exit(main())