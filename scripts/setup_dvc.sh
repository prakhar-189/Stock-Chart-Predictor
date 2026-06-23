#!/usr/bin/env bash
# =============================================================================
# File        : scripts/setup_dvc.sh
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> One-shot DVC initialization.
#               -> Installs the dvc[s3] extra, runs `dvc init`, configures a
#                  remote, and adds data/raw/sp500_stocks.csv to DVC tracking.
# =============================================================================

set -euo pipefail

: "${DVC_REMOTE_URL:?DVC_REMOTE_URL not set in environment}"

pip install 'dvc[s3]'

if [ ! -d ".dvc" ]; then
    dvc init
fi

dvc remote add -d origin "${DVC_REMOTE_URL}" --force

if [ -f "data/raw/sp500_stocks.csv" ]; then
    dvc add data/raw/sp500_stocks.csv
    git add data/raw/sp500_stocks.csv.dvc .gitignore
else
    echo "WARNING: data/raw/sp500_stocks.csv not found — skip dvc add" >&2
fi

echo "DVC setup complete. Push raw data with: dvc push"