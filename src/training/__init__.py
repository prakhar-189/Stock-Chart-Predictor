# src/training/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'training' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from src.training.dataset  import ChartWindowDataset
#                   from src.training.train    import main as train_main
#                   from src.training.evaluate import main as evaluate_main
#
# Modules in this package:
#   dataset.py  : PyTorch Dataset over the train/val/test CSV
#                 manifests. Reuses the vision_model image processor
#                 so training and inference share the exact same
#                 transform — guards against train/serve skew.
#   train.py    : Fixed-epoch fine-tuning loop. AdamW + linear
#                 warmup-decay schedule, per-epoch validation,
#                 best-val-loss checkpointing, MLflow logging,
#                 early stopping.
#   evaluate.py : Scores models/best.pt on the held-out test set.
#                 Writes test_metrics.json (accuracy, per-class
#                 P/R/F1, macro F1) and confusion_matrix.csv for
#                 DVC plots.
# --------------------------------------------------------------