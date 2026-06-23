# =============================================================================
# File        : tests/test_pipeline.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Stitches data-layer outputs end-to-end on synthetic data.
#               -> Doesn't touch the model; just verifies windows + manifests
#                  flow consistently. Catches regressions in the contract
#                  between stages (column names, dtypes).
# =============================================================================


import pandas as pd

from src.data.build_dataset import time_split


def test_time_split_proportions():
    df = pd.DataFrame({
        "window_end_date" : pd.date_range("2020-01-01", periods=1000, freq="D"),
        "label"           : ["up"] * 1000,
    })
    splits = time_split(df, train_frac=0.7, val_frac=0.15)
    assert len(splits["train"]) == 700
    assert len(splits["val"])   == 150
    assert len(splits["test"])  == 150
    # Time order preserved
    assert splits["train"]["window_end_date"].max() < splits["val"]["window_end_date"].min()
    assert splits["val"]["window_end_date"].max()   < splits["test"]["window_end_date"].min()