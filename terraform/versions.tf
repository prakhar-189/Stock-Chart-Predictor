# =============================================================================
# File        : terraform/versions.tf
# Author      : Prakhar Srivastava
# Date        : 2026-06-21
# Description : -> Terraform CLI and provider version pins.
#               -> Lives in its own file (rather than inlined in main.tf)
#                  so the version contract is the first thing reviewers see
#                  and so version bumps produce diffs against a stable,
#                  resource-free file.
#
#               -> Why pin both:
#                    required_version : guarantees CLI features the config
#                                       relies on (e.g. moved blocks, validation).
#                    required_providers: guarantees provider attribute schemas
#                                        — minor provider bumps can rename
#                                        resource arguments without warning.
# =============================================================================

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}