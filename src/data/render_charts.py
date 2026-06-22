# =============================================================================
# File        : src/data/render_charts.py
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Candlestick rasterizer.
#               -> Renders each labeled window as a 224x224 PNG with no axes,
#                  ticks, or padding — a clean, ViT-ready image. Saved into
#                  data/processed/{label}/{symbol}_{YYYYMMDD}.png so the
#                  directory structure is itself the label (ImageFolder layout).
#
#               -> Pipeline position - Stage 3 (Rasterization).
#
#               -> Why raw matplotlib instead of mplfinance:
#                    mplfinance is slower and emits chrome (axes, gridlines,
#                    volume panes) we don't want feeding into a ViT.
#                    Raw rectangles + wicks let us guarantee the only signal
#                    the model sees is the candlestick pattern.
#
#               -> Why a process pool:
#                    Rendering ~20K images is CPU-bound and embarrassingly
#                    parallel. ProcessPoolExecutor sidesteps the GIL.
#                    Idempotent (skips existing files) so jobs can resume.
# =============================================================================


# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# logging              : Standard library — DVC stage progress logs.
# os                   : Standard library — cpu_count() for sizing the worker
#                        pool AND setting MPLBACKEND before matplotlib loads.
# sys                  : Standard library — exit code from main().
# ProcessPoolExecutor  : Standard library — parallel chart rendering.
# as_completed         : Standard library — stream worker results into logs.
# pathlib              : Standard library — cross-platform paths.
# matplotlib.pyplot    : Figure, axes, Rectangle for candlesticks. Backend
#                        is fixed to "Agg" via MPLBACKEND env var ABOVE the
#                        import so worker processes (no display) don't fail.
# pandas               : Read ohlcv + windows parquet, groupby per ticker.
# yaml                 : Parses params.yaml for model.image_size.
# =============================================================================
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib            import Path

import matplotlib.pyplot as plt
import pandas            as pd
import yaml


logger = logging.getLogger(__name__)


REPO_ROOT        = Path(__file__).resolve().parents[2]
PARAMS_PATH      = REPO_ROOT / "params.yaml"
OHLCV_PARQUET    = REPO_ROOT / "data" / "interim"   / "ohlcv.parquet"
WINDOWS_PARQUET  = REPO_ROOT / "data" / "interim"   / "windows.parquet"
IMAGE_ROOT       = REPO_ROOT / "data" / "processed"

UP_COLOR   = "#26a69a"      # Trading-View green
DOWN_COLOR = "#ef5350"      # Trading-View red


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


# =============================================================================
# _render_one
# -----------------------------------------------------------------------------
# Draws a single window of OHLC tuples as a candlestick chart. The image is
# axis-less and edge-to-edge so the entire pixel budget is the chart itself.
# Idempotent — returns early if the file already exists, so re-runs after a
# crash are cheap.
# =============================================================================
def _render_one(
    ohlc       : list[tuple[float, float, float, float]],
    out_path   : Path,
    image_size : int,
) -> None:

    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)
    width = 0.6

    for i, (o, h, l, c) in enumerate(ohlc):
        color = UP_COLOR if c >= o else DOWN_COLOR
        # Wick
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
        # Body
        bottom, top = (o, c) if c >= o else (c, o)
        ax.add_patch(
            plt.Rectangle((i - width / 2, bottom), width, top - bottom,
                          facecolor=color, edgecolor=color)
        )

    ax.set_xlim(-1, len(ohlc))
    ax.set_ylim(min(l for _, _, l, _ in ohlc), max(h for _, h, _, _ in ohlc))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, pad_inches=0)
    plt.close(fig)


# =============================================================================
# _render_batch
# -----------------------------------------------------------------------------
# Worker-process entrypoint. Batching amortizes matplotlib import overhead
# on Windows spawn semantics (each child re-imports the module).
# =============================================================================
def _render_batch(jobs: list[tuple[list, str, int]]) -> int:
    for ohlc, out_path, image_size in jobs:
        _render_one(ohlc, Path(out_path), image_size)
    return len(jobs)


def main() -> int:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    params      = load_params()
    image_size  = params["model"]["image_size"]

    ohlcv     = pd.read_parquet(OHLCV_PARQUET).sort_values(["symbol", "date"]).reset_index(drop=True)
    by_symbol = {sym: g.reset_index(drop=True) for sym, g in ohlcv.groupby("symbol", sort=False)}

    windows = pd.read_parquet(WINDOWS_PARQUET)
    logger.info("rendering %d windows at %dx%d", len(windows), image_size, image_size)

    jobs: list[tuple[list, str, int]] = []
    for row in windows.itertuples(index=False):
        ticker_df = by_symbol[row.symbol]
        slice_df  = ticker_df.iloc[row.start_idx : row.end_idx + 1]
        ohlc      = list(zip(slice_df["open"], slice_df["high"],
                             slice_df["low"],  slice_df["close"]))
        end_date_str = pd.Timestamp(row.window_end_date).strftime("%Y%m%d")
        out_path     = str(IMAGE_ROOT / row.label / f"{row.symbol}_{end_date_str}.png")
        jobs.append((ohlc, out_path, image_size))

    chunk    = 500
    workers  = max(1, (os.cpu_count() or 2) // 2)
    rendered = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_render_batch, jobs[i : i + chunk])
                   for i in range(0, len(jobs), chunk)]
        for fut in as_completed(futures):
            rendered += fut.result()

    logger.info("rendered %d images under %s", rendered, IMAGE_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())