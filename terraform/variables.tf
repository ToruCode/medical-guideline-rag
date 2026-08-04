variable "aws_region" {
  description = "AWS region the stack is deployed into."
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "Short name used as a prefix for every resource this stack creates."
  type        = string
  default     = "medical-guideline-rag"
}

variable "domain_name" {
  description = <<-EOT
    Optional custom domain name (must already have a Route53 hosted
    zone in this account). When null (the default), the ALB is
    HTTP-only - no ACM certificate, HTTPS listener, or Route53 record
    is created. Set this once a domain is available to add HTTPS. See
    docs/adr/0029-aws-ecs-fargate-deployment.md.
  EOT
  type        = string
  default     = null
}

variable "container_image_tag" {
  description = "Image tag in the ECR repository that the app/ui task definitions run. Updated by the GitHub Actions deploy job (PR3); Terraform itself never builds or pushes an image."
  type        = string
  default     = "latest"
}

variable "github_repository" {
  description = "GitHub \"owner/repo\" allowed to assume the GitHub Actions OIDC deploy role, restricted to the main branch. See iam.tf."
  type        = string
  default     = "ToruCode/medical-guideline-rag"
}

variable "llm_provider" {
  description = <<-EOT
    MEDICAL_RAG_LLM_PROVIDER for the app task: "fake" (default - the
    Secrets Manager secret is created with a placeholder value, so
    "openai" would fail authentication until a real key is set) or
    "openai" (set this only after putting a real key into the
    llm_api_key secret - see docs/deployment-guide.md).
  EOT
  type        = string
  default     = "fake"

  validation {
    condition     = contains(["fake", "openai"], var.llm_provider)
    error_message = "llm_provider must be \"fake\" or \"openai\"."
  }
}

variable "embedding_provider" {
  description = "MEDICAL_RAG_EMBEDDING_PROVIDER for the app task. \"fake\" (default, no model download, fits the low-cost app_cpu/app_memory defaults) or \"sentence_transformers\" (downloads a real model on first use - consider raising app_cpu/app_memory if you switch to this)."
  type        = string
  default     = "fake"

  validation {
    condition     = contains(["fake", "sentence_transformers"], var.embedding_provider)
    error_message = "embedding_provider must be \"fake\" or \"sentence_transformers\"."
  }
}

variable "app_cpu" {
  description = "Fargate CPU units for the app (FastAPI) task. See AWS's Fargate task size table for valid cpu/memory pairs."
  type        = number
  default     = 512
}

variable "app_memory" {
  description = "Fargate memory (MiB) for the app (FastAPI) task."
  type        = number
  default     = 1024
}

variable "ui_cpu" {
  description = "Fargate CPU units for the ui (Streamlit) task."
  type        = number
  default     = 256
}

variable "ui_memory" {
  description = "Fargate memory (MiB) for the ui (Streamlit) task."
  type        = number
  default     = 512
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention for both ECS log groups."
  type        = number
  default     = 14
}
