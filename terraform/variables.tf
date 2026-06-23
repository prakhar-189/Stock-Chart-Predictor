# =============================================================================
# File        : terraform/variables.tf
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Input variable declarations for the project's Terraform
#                  root module.
#               -> Splitting variables into their own file (rather than
#                  inlining in main.tf) is Terraform convention — `main.tf`
#                  declares what gets built, `variables.tf` declares what
#                  knobs control it, and `outputs.tf` declares what the
#                  root module exposes to its callers.
#
#               -> Required vs optional:
#                    project_id has no default — must be passed via
#                    `terraform.tfvars`, `-var`, or `TF_VAR_project_id`.
#                    Everything else has sensible defaults so a fresh
#                    `terraform apply` works after only setting project_id.
# =============================================================================


# =============================================================================
# project_id
# -----------------------------------------------------------------------------
# GCP project that owns every resource declared in main.tf. Required.
# Validation rejects the empty string so a missing-tfvars mistake fails fast
# at plan time instead of midway through apply.
# =============================================================================
variable "project_id" {
  description = "GCP project ID that owns the GKE cluster, GCS bucket, and Artifact Registry repo."
  type        = string

  validation {
    condition     = length(var.project_id) > 0
    error_message = "project_id must be set — pass it via terraform.tfvars, -var, or TF_VAR_project_id."
  }
}


# =============================================================================
# region
# -----------------------------------------------------------------------------
# Default region for all regional resources (cluster, bucket, registry).
# us-central1 chosen for cheap egress and broad service availability.
# =============================================================================
variable "region" {
  description = "GCP region for the GKE cluster, GCS bucket, and Artifact Registry repo."
  type        = string
  default     = "us-central1"
}


# =============================================================================
# zone
# -----------------------------------------------------------------------------
# Used only if zonal resources are added later (e.g. a Filestore instance).
# Autopilot clusters are regional, so this is unused today but kept for
# forward compatibility.
# =============================================================================
variable "zone" {
  description = "GCP zone — reserved for future zonal resources. Unused by the current Autopilot cluster."
  type        = string
  default     = "us-central1-a"
}


# =============================================================================
# name
# -----------------------------------------------------------------------------
# Naming prefix applied to every resource so a single GCP project can host
# multiple environments side by side (e.g. -dev, -staging). Constrained to
# the lowercase / hyphen alphabet that GCS and Artifact Registry both accept.
# =============================================================================
variable "name" {
  description = "Naming prefix for all created resources. Lowercase letters, digits, and hyphens only."
  type        = string
  default     = "stock-chart-predictor"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}[a-z0-9]$", var.name))
    error_message = "name must be 3-32 chars, lowercase letters/digits/hyphens, and must start with a letter."
  }
}