# =============================================================================
# File        : terraform/main.tf
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Resource declarations for the project's cloud foundation.
#               -> Creates a GKE Autopilot cluster, a GCS bucket for the DVC
#                  remote, and an Artifact Registry repo for container images.
#
#               -> Companion files:
#                    versions.tf            : Terraform + provider version pins.
#                    variables.tf           : input knobs (project_id, region, name).
#                    outputs.tf             : values exposed after apply.
#                    terraform.tfvars.example: template for required variable values.
#
#               -> Downstream:
#                    The k8s manifests in ../k8s/ deploy the workloads onto
#                    the cluster this file builds. Container images are
#                    pushed to the Artifact Registry repo defined below.
# =============================================================================


# =============================================================================
# Provider configuration
# -----------------------------------------------------------------------------
# Project and region come from variables.tf so the same config can target
# multiple GCP environments without code edits.
# =============================================================================
provider "google" {
  project = var.project_id
  region  = var.region
}


# =============================================================================
# GKE Autopilot cluster
# -----------------------------------------------------------------------------
# Autopilot is the cheapest "real" GKE mode for a portfolio demo — pay per
# pod-second, no node-pool management.
#
# `deletion_protection = false` so `terraform destroy` cleans up without
# manual gcloud intervention. Flip back to true for any environment with
# actual users; portfolio scope only.
# =============================================================================
resource "google_container_cluster" "primary" {
  name                = var.name
  location            = var.region
  enable_autopilot    = true
  deletion_protection = false
}


# =============================================================================
# DVC remote (GCS bucket)
# -----------------------------------------------------------------------------
# Single bucket per environment. Uniform bucket-level access disables ACLs
# in favor of IAM only — GCP's current recommended default and the only
# mode compatible with the DVC GCS remote contract.
#
# `force_destroy = true` mirrors the cluster's deletion_protection stance.
# =============================================================================
resource "google_storage_bucket" "dvc_remote" {
  name                        = "${var.name}-dvc"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}


# =============================================================================
# Artifact Registry repo (Docker images)
# -----------------------------------------------------------------------------
# The CD workflow pushes built API + UI images here, tagged as:
#   <region>-docker.pkg.dev/<project>/<name>/<service>:<git-sha>
# Single repo, no dev/prod split for this portfolio scope.
# =============================================================================
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.name
  format        = "DOCKER"
}