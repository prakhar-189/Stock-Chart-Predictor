# =============================================================================
# File        : tests/conftest.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Shared pytest fixtures.
#               -> Synthetic OHLCV frames, temp dirs, and stub config so
#                  data-layer tests don't need the real Kaggle CSV.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# pathlib : Standard library — locate REPO_ROOT for path-based fixtures.
# numpy   : Random-walk price generation for the synthetic OHLCV fixture.
# pandas  : Assembles the fixture into the same DataFrame shape that the
#           real pipeline emits, so tests exercise the production contract.
# pytest  : Fixture decorator — the only public API surface this file uses.
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """500 days of random-walk OHLCV for two tickers — enough to test
    windowing without round-tripping the real dataset."""
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rows  = []
    for sym in ["AAA", "BBB"]:
        price = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        for d, p in zip(dates, price, strict=False):
            o = p + rng.normal(0, 0.2)
            h = max(o, p) + abs(rng.normal(0, 0.3))
            lo = min(o, p) - abs(rng.normal(0, 0.3))
            c = p
            rows.append({"date": d, "symbol": sym, "open": o, "high": h,
                         "low": lo, "close": c, "adj_close": c, "volume": int(rng.integers(1e5, 1e6))})
    return pd.DataFrame(rows)