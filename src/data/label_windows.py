# =============================================================================
# File        : src/data/label_windows.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Sliding-window labeler.
#               -> For each window of `window_size` candles, computes the
#                  forward return at `horizon` days using adj_close and
#                  assigns one of three labels:
#                       up        if forward_return >  threshold
#                       down      if forward_return < -threshold
#                       sideways  otherwise
#
#               -> Pipeline position - Stage 2 (Label generation):
#                    Consumes data/interim/ohlcv.parquet from stage 1 and
#                    feeds data/interim/windows.parquet to stages 3 and 4.
#
#               -> Why adj_close for returns but raw OHLC for charts:
#                    adj_close handles splits and dividends — without it,
#                    a 2-for-1 split looks like a -50% return. The chart
#                    renderer uses raw OHLC so the *visual pattern* still
#                    reflects what a trader would have seen on the day.
#
#               -> Why vectorized over a Python loop:
#                    200K+ windows is too slow for `df.at[i, col]` per row.
#                    Numpy slicing brings it down to seconds.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging   : Standard library — structured run logs for the DVC stage output.
# sys       : Standard library — POSIX-style exit code returned by main().
# pathlib   : Standard library — cross-platform path handling (Windows + POSIX).
# numpy     : Vectorized window-index math (np.arange) and label assignment
#             (np.where). The whole point of using numpy here is to avoid
#             a per-row Python loop over 200K+ candidate windows.
# pandas    : Tabular IO — reads ohlcv.parquet, groups by ticker, writes
#             windows.parquet.
# yaml      : Parses params.yaml (window_size, stride, horizon, threshold).
# =============================================================================
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


REPO_ROOT   = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"
IN_PARQUET  = REPO_ROOT / "data" / "interim" / "ohlcv.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "interim" / "windows.parquet"


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


# =============================================================================
# build_windows_for_ticker
# -----------------------------------------------------------------------------
# Generates the window manifest for a single ticker. Vectorized.
#
# Parameters:
#   df          : pd.DataFrame  OHLCV slice for one ticker, will be re-sorted.
#   window_size : int           Candles per window (params.data.window_size).
#   stride      : int           Step between window starts (typically 1).
#   horizon     : int           Days forward to compute the label return.
#   threshold   : float         Absolute return cutoff for up/down vs sideways.
#
# Returns:
#   pd.DataFrame with columns:
#       symbol, window_start_date, window_end_date, label_date,
#       start_idx, end_idx, forward_return, label
# =============================================================================
def build_windows_for_ticker(
    df          : pd.DataFrame,
    window_size : int,
    stride      : int,
    horizon     : int,
    threshold   : float,
) -> pd.DataFrame:

    df = df.sort_values("date").reset_index(drop=True)

    # `last_start` = last valid start index that still leaves room for
    # the full window AND the horizon offset for the label price.
    last_start = len(df) - window_size - horizon
    if last_start < 0:
        return pd.DataFrame()

    starts     = np.arange(0, last_start + 1, stride)
    ends       = starts + window_size - 1
    label_idxs = ends + horizon

    price_now    = df["adj_close"].iloc[ends].to_numpy()
    price_future = df["adj_close"].iloc[label_idxs].to_numpy()
    fwd_return   = price_future / price_now - 1.0

    label = np.where(
        fwd_return >  threshold, "up",
        np.where(fwd_return < -threshold, "down", "sideways"),
    )

    return pd.DataFrame({
        "symbol"            : df["symbol"].iloc[ends].to_numpy(),
        "window_start_date" : df["date"  ].iloc[starts].to_numpy(),
        "window_end_date"   : df["date"  ].iloc[ends].to_numpy(),
        "label_date"        : df["date"  ].iloc[label_idxs].to_numpy(),
        "start_idx"         : starts,
        "end_idx"           : ends,
        "forward_return"    : fwd_return,
        "label"             : label,
    })


def main() -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    params = load_params()["data"]

    df = pd.read_parquet(IN_PARQUET)
    logger.info("loaded %d OHLCV rows for %d tickers", len(df), df["symbol"].nunique())

    frames: list[pd.DataFrame] = []
    for ticker, ticker_df in df.groupby("symbol", sort=False):
        windows = build_windows_for_ticker(
            ticker_df,
            window_size = params["window_size"],
            stride      = params["stride"],
            horizon     = params["horizon"],
            threshold   = params["threshold"],
        )
        if windows.empty:
            logger.warning("%s: not enough rows for any window", ticker)
            continue
        frames.append(windows)
        logger.info("%s: %d windows", ticker, len(windows))

    out = pd.concat(frames, ignore_index=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)

    logger.info("wrote %d windows to %s", len(out), OUT_PARQUET)
    logger.info("label distribution: %s", out["label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())