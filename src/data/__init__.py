# src/data/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : Package initializer for the 'data' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from src.data.load_ohlcv     import load_and_filter
#                   from src.data.label_windows  import build_windows_for_ticker
#                   from src.data.build_dataset  import time_split
#
# Modules in this package (in DVC pipeline order):
#   load_ohlcv.py    : Stage 1 — reads the static Kaggle CSV under
#                      data/raw/, filters to the configured tickers
#                      and date range, writes data/interim/ohlcv.parquet.
#   label_windows.py : Stage 2 — sliding-window labeler. Computes
#                      forward returns and assigns up/sideways/down
#                      labels per window. Output: windows.parquet.
#   render_charts.py : Stage 3 — rasterizes each window as a 224x224
#                      candlestick PNG into data/processed/{label}/.
#   build_dataset.py : Stage 4 — chronological train/val/test split
#                      manifests (data/splits/{train,val,test}.csv).
# --------------------------------------------------------------