# =============================================================================
# File        : terraform/outputs.tf
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Output values exposed by the root module after apply.
#               -> Two consumers in mind:
#                    1. The deploy scripts under ../scripts/ — they grab
#                       the cluster name + region to call
#                       `gcloud container clusters get-credentials`.
#                    2. CI workflows — they configure the DVC remote against
#                       dvc_bucket_url and push container images to
#                       image_registry.
#
#               -> Sensitive outputs:
#                    cluster_endpoint is marked sensitive so it doesn't get
#                    printed to logs in CI by accident. Reveal with
#                    `terraform output -raw cluster_endpoint`.
# =============================================================================


# =============================================================================
# cluster_endpoint
# -----------------------------------------------------------------------------
# Kubernetes API server endpoint. Used only by tooling that talks to the
# control plane directly (kubectl typically resolves this via gcloud).
# =============================================================================
output "cluster_endpoint" {
  description = "Kubernetes API server endpoint for the GKE Autopilot cluster."
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}


# =============================================================================
# cluster_name
# -----------------------------------------------------------------------------
# Cluster name as registered with GKE. Required by `gcloud container
# clusters get-credentials` in scripts/deploy_gke.sh.
# =============================================================================
output "cluster_name" {
  description = "Name of the GKE Autopilot cluster."
  value       = google_container_cluster.primary.name
}


# =============================================================================
# cluster_location
# -----------------------------------------------------------------------------
# Region (Autopilot is regional). Pair with cluster_name for kubeconfig.
# =============================================================================
output "cluster_location" {
  description = "Region of the GKE Autopilot cluster."
  value       = google_container_cluster.primary.location
}


# =============================================================================
# dvc_bucket_url
# -----------------------------------------------------------------------------
# Fully-qualified gs:// URL used by `dvc remote add -d origin <url>`.
# =============================================================================
output "dvc_bucket_url" {
  description = "GCS URL for the DVC remote — `dvc remote add -d origin <this value>`."
  value       = google_storage_bucket.dvc_remote.url
}


# =============================================================================
# dvc_bucket_name
# -----------------------------------------------------------------------------
# Bare bucket name (without gs:// prefix) for tools that want it separately.
# =============================================================================
output "dvc_bucket_name" {
  description = "Name of the GCS bucket backing the DVC remote."
  value       = google_storage_bucket.dvc_remote.name
}


# =============================================================================
# image_registry
# -----------------------------------------------------------------------------
# Artifact Registry repo path; the CD workflow tags images as
# {region}-docker.pkg.dev/{project}/{repo}/{service}:{sha}.
# =============================================================================
output "image_registry" {
  description = "Artifact Registry repository ID for container images."
  value       = google_artifact_registry_repository.images.id
}