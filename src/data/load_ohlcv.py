# =============================================================================
# File        : src/data/load_ohlcv.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Static-CSV OHLCV loader.
#               -> Reads the Kaggle "S&P 500 Stocks" dump from data/raw/,
#                  filters to the tickers and date range defined in
#                  params.yaml, validates the schema, and writes a clean
#                  parquet to data/interim/ohlcv.parquet for downstream stages.
#
#               -> Pipeline position - Stage 1 (Data ingest):
#                    First node of the DVC DAG. All later stages depend on
#                    the parquet this writes.
#
#               -> Why a static CSV instead of a live yfinance pull:
#                    Reproducibility. Identical bytes every run, no API
#                    rate limits, no silent schema drift from the provider.
#                    Trade-off: must re-download the Kaggle file to refresh.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging   : Standard library — structured run logs for the DVC stage output.
# sys       : Process exit code for the CLI entrypoint.
# pathlib   : Cross-platform path handling (Windows + POSIX).
# pandas    : Tabular read/filter/write.
# yaml      : Parse params.yaml.
# =============================================================================
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level paths
# -----------------------------------------------------------------------------
# REPO_ROOT  : Resolved at import time so the module is callable from anywhere.
# PARAMS     : DVC-tracked hyperparameters.
# RAW_CSV    : Static Kaggle file the user drops in manually.
# OUT_PARQUET: Stage output declared in dvc.yaml.
# =============================================================================
REPO_ROOT   = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"
RAW_CSV     = REPO_ROOT / "data" / "raw"     / "sp500_stocks.csv"
OUT_PARQUET = REPO_ROOT / "data" / "interim" / "ohlcv.parquet"

EXPECTED_COLS = {"date", "symbol", "adj_close", "close", "high", "low", "open", "volume"}


# =============================================================================
# load_params
# -----------------------------------------------------------------------------
# Reads params.yaml into a dict. Kept as a thin wrapper so tests can monkey-
# patch it without touching the filesystem.
# =============================================================================
def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


# =============================================================================
# _normalize_columns
# -----------------------------------------------------------------------------
# Lowercases and snake_cases column names. The Kaggle file ships with names
# like "Adj Close" — every downstream module assumes "adj_close".
# =============================================================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# =============================================================================
# _validate
# -----------------------------------------------------------------------------
# Hard fails on missing columns or missing tickers. We *want* this to crash
# loudly in CI so that a silent provider change can't propagate downstream.
# =============================================================================
def _validate(df: pd.DataFrame, tickers: list[str]) -> None:
    missing_cols = EXPECTED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"missing expected columns: {sorted(missing_cols)}")

    missing_tickers = set(tickers) - set(df["symbol"].unique())
    if missing_tickers:
        raise ValueError(f"tickers not present in source CSV: {sorted(missing_tickers)}")


# =============================================================================
# load_and_filter
# -----------------------------------------------------------------------------
# Reads, normalizes, validates, filters by ticker and date range, drops null
# adj_close rows, and returns a tidy DataFrame ordered by (symbol, date).
#
# Parameters:
#   csv_path : Path     Static Kaggle file.
#   tickers  : list[str] Tickers to keep.
#   start    : str       ISO date, inclusive.
#   end      : str       ISO date, exclusive.
# =============================================================================
def load_and_filter(csv_path: Path, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    logger.info("reading %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["Date"], dtype={"Symbol": "string"})
    df = _normalize_columns(df)
    _validate(df, tickers)

    df = df[df["symbol"].isin(tickers)]
    df = df[(df["date"] >= start) & (df["date"] < end)]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    null_rows = df["adj_close"].isna().sum()
    if null_rows:
        logger.warning("dropping %d rows with null adj_close", null_rows)
        df = df.dropna(subset=["adj_close"])

    return df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]


# =============================================================================
# main
# -----------------------------------------------------------------------------
# CLI entrypoint invoked by `python -m src.data.load_ohlcv` (or DVC).
# Returns POSIX exit code (0 on success) so CI can detect failures.
# =============================================================================
def main() -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    params = load_params()["data"]
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(RAW_CSV, params["tickers"], params["start"], params["end"])
    df.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)

    logger.info("wrote %d rows to %s", len(df), OUT_PARQUET)
    for ticker, n in df.groupby("symbol").size().items():
        logger.info("  %s: %d rows", ticker, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())