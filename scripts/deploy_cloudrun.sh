#!/usr/bin/env bash
# =============================================================================
# File        : scripts/deploy_cloudrun.sh
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Builds and deploys the API image to Google Cloud Run.
#               -> Cloud Run is the cheapest "real" deployment for a
#                  portfolio demo — scale-to-zero, pay per request.
# =============================================================================

set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:=us-central1}"
SERVICE="stock-chart-predictor-api"
IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/stock-chart-predictor/${SERVICE}:$(git rev-parse --short HEAD)"

gcloud builds submit --tag "${IMAGE}" --file docker/Dockerfile.api .

gcloud run deploy "${SERVICE}" \
    --image  "${IMAGE}" \
    --region "${GCP_REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 20

echo "Deployed ${IMAGE} to ${SERVICE} in ${GCP_REGION}"