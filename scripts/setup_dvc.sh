#!/usr/bin/env bash
# =============================================================================
# File        : scripts/setup_dvc.sh
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> One-shot DVC initialization (Google Drive remote).
#               -> Initializes DVC, configures a GDrive folder as the remote,
#                  and registers the static Kaggle CSV as a tracked file.
#               -> First `dvc push` triggers a browser OAuth flow; credentials
#                  cache under ~/.cache/pydrive2fs so subsequent pushes are
#                  non-interactive.
#
#               -> Required env var:
#                    DVC_GDRIVE_FOLDER_ID — the long string after /folders/
#                                           in your Google Drive folder URL.
# =============================================================================

set -euo pipefail

: "${DVC_GDRIVE_FOLDER_ID:?DVC_GDRIVE_FOLDER_ID not set — paste the ID of a Google Drive folder you've created for DVC}"

pip install 'dvc[gdrive]'

if [ ! -d ".dvc" ]; then
    dvc init
fi

dvc remote add -d origin "gdrive://${DVC_GDRIVE_FOLDER_ID}" --force

if [ -f "data/raw/sp500_stocks.csv" ]; then
    dvc add data/raw/sp500_stocks.csv
    git add data/raw/sp500_stocks.csv.dvc .gitignore
else
    echo "WARNING: data/raw/sp500_stocks.csv not found — skip dvc add" >&2
fi

echo "DVC setup complete. Run \`dvc push\` to upload raw data (will prompt for Google OAuth on first run)."
