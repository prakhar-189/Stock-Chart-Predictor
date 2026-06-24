# =============================================================================
# File        : src/training/dataset.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> PyTorch Dataset over the train/val/test CSV manifests
#                  produced by src/data/build_dataset.py.
#               -> Each item is a (pixel_values, label_id) tensor pair
#                  preprocessed via the ViT image processor declared by
#                  src/models/vision_model.py — same transform used at
#                  inference for parity.
#
#               -> Why a manifest-driven Dataset (instead of ImageFolder):
#                    The CSV carries date and forward_return, which is
#                    useful for stratified analysis and for the time-aware
#                    split contract. ImageFolder would drop those columns.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# pathlib                 : Standard library — resolve image paths under REPO_ROOT.
# pandas                  : Reads the split manifest CSVs (train/val/test).
# torch                   : Builds the label tensor returned by __getitem__.
# PIL.Image               : Decodes rendered chart PNGs from disk. Lazy — only
#                           the manifest sits in memory; images load per item.
# Dataset                 : Torch base class for the manifest-driven iterator.
# LABEL2ID,
# build_image_processor   : Local re-exports from src.models.vision_model so
#                           training and inference share the SAME label maps
#                           and preprocessing transforms — guards against the
#                           single most common train/serve skew bug.
# =============================================================================
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.models.vision_model import LABEL2ID, build_image_processor

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# ChartWindowDataset
# -----------------------------------------------------------------------------
# Reads a single split CSV (train/val/test). Lightweight — only the manifest
# is held in memory; images are decoded lazily inside __getitem__.
# =============================================================================
class ChartWindowDataset(Dataset):

    def __init__(self, split_csv: Path, processor=None):
        self.df = pd.read_csv(split_csv)
        self.processor = processor or build_image_processor()
        self.repo_root = REPO_ROOT

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.repo_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        # `processor(...)` returns a BatchFeature with `pixel_values` of
        # shape (1, 3, H, W); we squeeze the batch dim for DataLoader.
        encoded = self.processor(images = image, return_tensors = "pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        label_id = LABEL2ID[row["label"]]

        return{
            "pixel_values" : pixel_values,
            "labels" : torch.tensor(label_id, dtype = torch.long),
        }