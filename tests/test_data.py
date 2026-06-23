# =============================================================================
# File        : tests/test_data.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Data-layer unit tests.
#               -> Validates schema normalization, window math, and label
#                  threshold logic against synthetic data — fast, fully
#                  offline, no external CSV required.
# =============================================================================


import numpy as np
import pandas as pd

from src.data.label_windows import build_windows_for_ticker
from src.data.load_ohlcv    import _normalize_columns, _validate


# =============================================================================
# Schema normalization
# =============================================================================
def test_normalize_columns_lowercases_and_snake_cases():
    df = pd.DataFrame(columns=["Date", "Symbol", "Adj Close", "Volume"])
    df = _normalize_columns(df)
    assert set(df.columns) == {"date", "symbol", "adj_close", "volume"}


# =============================================================================
# Validation
# =============================================================================
def test_validate_raises_on_missing_ticker(synthetic_ohlcv):
    df = _normalize_columns(synthetic_ohlcv)
    try:
        _validate(df, ["AAA", "ZZZ"])
    except ValueError as e:
        assert "ZZZ" in str(e)
    else:
        raise AssertionError("expected ValueError on missing ticker")


# =============================================================================
# Windowing math
# =============================================================================
def test_build_windows_emits_correct_count(synthetic_ohlcv):
    one = synthetic_ohlcv[synthetic_ohlcv["symbol"] == "AAA"]
    windows = build_windows_for_ticker(one, window_size=30, stride=1, horizon=5, threshold=0.02)
    # n_windows = len - window_size - horizon + 1
    assert len(windows) == len(one) - 30 - 5 + 1


def test_label_thresholds(synthetic_ohlcv):
    one = synthetic_ohlcv[synthetic_ohlcv["symbol"] == "AAA"]
    windows = build_windows_for_ticker(one, window_size=30, stride=1, horizon=5, threshold=0.02)
    # Every label must be drawn from the three-class set
    assert set(windows["label"].unique()) <= {"up", "sideways", "down"}
    # The classification must be consistent with the forward return
    up   = windows[windows["label"] == "up"]
    down = windows[windows["label"] == "down"]
    assert (up["forward_return"]   >  0.02).all()
    assert (down["forward_return"] < -0.02).all()