output "alb_dns_name" {
  description = "Public DNS name of the ALB. The ui is reachable at http://<this>/ and the API at http://<this>/api/v1/... (or the custom domain, once domain_name is configured)."
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB's own hosted zone ID - needed for a Route53 alias record if domain_name was not set through this Terraform config."
  value       = aws_lb.main.zone_id
}

output "ecr_repository_url" {
  description = "ECR repository URL the CI/CD pipeline (PR3) pushes images to."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_app_service_name" {
  value = aws_ecs_service.app.name
}

output "ecs_ui_service_name" {
  value = aws_ecs_service.ui.name
}

output "github_actions_deploy_role_arn" {
  description = "IAM role ARN the CI/CD pipeline (PR3) assumes via OIDC to deploy."
  value       = aws_iam_role.github_actions_deploy.arn
}

output "s3_bucket_name" {
  value = aws_s3_bucket.guidelines.bucket
}

output "llm_api_key_secret_arn" {
  description = "Secrets Manager ARN to put the real OpenAI API key into (see docs/deployment-guide.md) - Terraform only creates it with a placeholder value."
  value       = aws_secretsmanager_secret.llm_api_key.arn
}
