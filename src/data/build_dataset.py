# =============================================================================
# File        : src/data/build_dataset.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Train/val/test manifest builder.
#               -> Reads the windows parquet, joins rendered image paths,
#                  and writes three CSV manifests under data/splits/.
#                  Each row: image_path, label, symbol, window_end_date,
#                  forward_return.
#
#               -> Pipeline position - Stage 4 (Manifest / split).
#
#               -> Why time-aware split (not random):
#                    Random shuffling leaks future windows into the training
#                    set and inflates test accuracy. Splitting chronologically
#                    is the only honest evaluation for time-series data.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging   : Standard library — split-stage diagnostics (sizes, label dists).
# sys       : Standard library — exit code from main().
# pathlib   : Standard library — cross-platform paths.
# pandas    : Reads windows.parquet, joins rendered image paths, writes
#             train/val/test CSV manifests.
# yaml      : Parses train_frac / val_frac from params.yaml.
# =============================================================================
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


REPO_ROOT        = Path(__file__).resolve().parents[2]
PARAMS_PATH      = REPO_ROOT / "params.yaml"
WINDOWS_PARQUET  = REPO_ROOT / "data" / "interim"   / "windows.parquet"
IMAGE_ROOT       = REPO_ROOT / "data" / "processed"
SPLITS_DIR       = REPO_ROOT / "data" / "splits"


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


def _image_path(label: str, symbol: str, end_date) -> str:
    end_date_str = pd.Timestamp(end_date).strftime("%Y%m%d")
    return f"data/processed/{label}/{symbol}_{end_date_str}.png"


# =============================================================================
# time_split
# -----------------------------------------------------------------------------
# Sort by window_end_date, then carve into three contiguous chronological
# slices. Test set is always the most recent windows, so reported numbers
# reflect performance on data the model has never seen forward in time.
# =============================================================================
def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> dict[str, pd.DataFrame]:
    df       = df.sort_values("window_end_date").reset_index(drop=True)
    n        = len(df)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)
    return {
        "train" : df.iloc[:n_train],
        "val"   : df.iloc[n_train : n_train + n_val],
        "test"  : df.iloc[n_train + n_val :],
    }


def main() -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    params = load_params()["data"]

    windows = pd.read_parquet(WINDOWS_PARQUET)
    windows["image_path"] = [
        _image_path(lbl, sym, dt)
        for lbl, sym, dt in zip(windows["label"], windows["symbol"], windows["window_end_date"], strict=False)
    ]

    exists  = windows["image_path"].apply(lambda p: (REPO_ROOT / p).exists())
    missing = (~exists).sum()
    if missing:
        logger.warning("%d windows have no rendered image — run render_charts first", missing)
        windows = windows[exists]

    splits = time_split(windows, params["train_frac"], params["val_frac"])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "label", "symbol", "window_end_date", "forward_return"]
    for name, split_df in splits.items():
        out = SPLITS_DIR / f"{name}.csv"
        split_df[cols].to_csv(out, index=False)
        logger.info("%s: %d rows -> %s", name, len(split_df), out)
        logger.info("  label dist: %s", split_df["label"].value_counts().to_dict())

    return 0


if __name__ == "__main__":
    sys.exit(main())