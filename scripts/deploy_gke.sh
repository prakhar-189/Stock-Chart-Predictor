#!/usr/bin/env bash
# =============================================================================
# File        : scripts/deploy_gke.sh
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Applies the k8s manifests to a GKE Autopilot cluster.
#               -> Assumes terraform/main.tf has already provisioned the
#                  cluster and Artifact Registry repo.
# =============================================================================

set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:=us-central1}"
: "${CLUSTER:=stock-chart-predictor}"

gcloud container clusters get-credentials "${CLUSTER}" --region "${GCP_REGION}" --project "${GCP_PROJECT}"

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.example.yaml      # replace with real secret in prod
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/deployment-ui.yaml
kubectl apply -f k8s/service-api.yaml
kubectl apply -f k8s/service-ui.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

kubectl -n stock-chart-predictor rollout status deployment/api
kubectl -n stock-chart-predictor rollout status deployment/ui