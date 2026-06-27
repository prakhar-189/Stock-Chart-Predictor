# =============================================================================
# File        : src/training/train.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Fine-tunes the ViT chart classifier.
#               -> Loads train/val splits, runs a fixed-epoch training loop
#                  with AdamW + linear warmup, evaluates every epoch, and
#                  saves the best (lowest val loss) checkpoint to
#                  models/best.pt. All hyperparameters, metrics, and
#                  artifacts are logged to MLflow.
#
#               -> Pipeline position - Stage 5 (Training).
#
#               -> Why a plain PyTorch loop instead of HF Trainer:
#                    Maximum transparency for a portfolio piece — every
#                    forward/backward, every metric is visible, no black-box
#                    callbacks. Trainer is a fine production choice but it
#                    hides the loop from a reviewer reading top-to-bottom.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# json                : Standard library — writes metrics/train_metrics.json
#                       consumed by DVC.
# logging             : Standard library — per-epoch progress logs.
# math                : Standard library — ceil() for steps_per_epoch.
# sys                 : Standard library — exit code from main().
# Counter             : Standard library — tallies class frequencies in the
#                       training manifest so we can derive inverse-frequency
#                       weights for CrossEntropyLoss.
# pathlib             : Standard library — checkpoint and metric paths.
# mlflow              : Experiment tracking. Logs hyperparams, per-epoch
#                       metrics, and the best checkpoint as an artifact.
# torch               : Devices, tensors, gradient context.
# torch.nn            : CrossEntropyLoss for 3-class classification.
# yaml                : Parses params.yaml (model + training sections).
# AdamW               : Decoupled weight-decay optimizer — the canonical
#                       choice for transformer fine-tuning.
# DataLoader          : Mini-batch iteration with worker processes for
#                       overlapping image decode with GPU compute.
# LambdaLR            : Implements the linear warmup + linear decay schedule.
# build_model, LABELS : Local — ViT factory and the class label list.
# ChartWindowDataset  : Local — manifest-driven Dataset for the split CSVs.
# =============================================================================
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vision_model import LABELS, build_model
from src.training.dataset import ChartWindowDataset

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"

def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)
    

# =============================================================================
# _linear_warmup_decay
# -----------------------------------------------------------------------------
# Standard transformer-style schedule: linearly warm up to `lr` over the
# first `warmup_steps` then linearly decay to 0 by `total_steps`.
# =============================================================================
def _linear_warmup_decay(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps -warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# _evaluate
# -----------------------------------------------------------------------------
# Runs one epoch in eval mode and returns (loss, accuracy). Called on the
# validation set every epoch and on the test set after training (by
# evaluate.py).
# =============================================================================
@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values = pixel_values)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    params         = load_params()
    train_cfg      = params["training"]
    model_cfg      = params["model"]

    torch.manual_seed(train_cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    train_ds = ChartWindowDataset(SPLITS_DIR / "train.csv")
    val_ds   = ChartWindowDataset(SPLITS_DIR / "val.csv")

    # ----------------------------------------------------------------
    # Class weights for imbalanced training set.
    # Without this, the model defaults to predicting majority classes
    # and the "down" class collapses to near-zero recall. Inverse-frequency
    # weighting normalized by the number of classes:
    #     weight_c = N / (K * n_c)
    # where N = train rows, K = num classes, n_c = count of class c.
    # ----------------------------------------------------------------
    label_counts  = Counter(train_ds.df["label"])
    class_weights = torch.tensor(
        [len(train_ds) / (len(LABELS) * label_counts[c]) for c in LABELS],
        dtype=torch.float,
        device=device,
    )
    logger.info("class weights: %s", dict(zip(LABELS, class_weights.tolist(), strict=False)))

    # pin_memory is only useful when copying to a CUDA device; on CPU it
    # wastes a copy and triggers a UserWarning.
    pin_memory = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True,  num_workers=train_cfg["num_workers"], pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"],
                              shuffle=False, num_workers=train_cfg["num_workers"], pin_memory=pin_memory)

    model     = build_model(model_cfg["backbone"], model_cfg["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                      weight_decay=train_cfg["weight_decay"])

    steps_per_epoch = math.ceil(len(train_ds) / train_cfg["batch_size"])
    total_steps     = steps_per_epoch * train_cfg["epochs"]
    warmup_steps    = int(total_steps * train_cfg["warmup_ratio"])
    scheduler       = _linear_warmup_decay(optimizer, warmup_steps, total_steps)
    loss_fn         = nn.CrossEntropyLoss(weight=class_weights)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("stock-chart-predictor")
    with mlflow.start_run():
        mlflow.log_params({**train_cfg, **{f"model.{k}": v for k, v in model_cfg.items()}})
        mlflow.log_params({f"class_weight.{lbl}": float(w)
                           for lbl, w in zip(LABELS, class_weights.tolist(), strict=False)})

        best_val_loss = float("inf")
        epochs_since_improve = 0
        history = []

        for epoch in range(1, train_cfg["epochs"] + 1):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{train_cfg['epochs']}", unit="batch")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                labels       = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values)
                logits  = outputs.logits
                loss    = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss    += loss.item() * labels.size(0)
                running_correct += (logits.argmax(dim=-1) == labels).sum().item()
                running_total   += labels.size(0)

            train_loss = running_loss    / running_total
            train_acc  = running_correct / running_total
            val_loss, val_acc = _evaluate(model, val_loader, device)

            logger.info("epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                        epoch, train_cfg["epochs"], train_loss, train_acc, val_loss, val_acc)

            mlflow.log_metrics({
                "train_loss" : train_loss,
                "train_acc"  : train_acc,
                "val_loss"   : val_loss,
                "val_acc"    : val_acc,
            }, step=epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                            "val_loss": val_loss, "val_acc": val_acc})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODELS_DIR / "best.pt")
                mlflow.log_artifact(str(MODELS_DIR / "best.pt"))
                epochs_since_improve = 0
                logger.info("  ^ new best — saved models/best.pt")
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= train_cfg["early_stopping_patience"]:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        with (METRICS_DIR / "train_metrics.json").open("w") as f:
            json.dump({"best_val_loss": best_val_loss, "history": history, "labels": LABELS}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())     