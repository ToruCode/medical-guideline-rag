data "aws_caller_identity" "current" {}

# --- ECS task execution role: used by both app and ui to pull the
# image from ECR and write to CloudWatch Logs. Also resolves the
# llm_api_key secret into a container env var at task startup for app
# (see ecs.tf's "secrets" block) - this is the only code path involved
# in Secrets Manager integration; Settings (app/core/config.py) reads
# it as a plain environment variable and never calls AWS itself. ---
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project_name}-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_task_execution_secrets" {
  name = "${var.project_name}-read-llm-api-key-secret"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "secretsmanager:GetSecretValue"
      Resource = aws_secretsmanager_secret.llm_api_key.arn
    }]
  })
}

# --- app task role: read/write the guideline-PDF S3 bucket
# (app/infrastructure/storage/s3_object_storage.py), and mount/write
# its EFS-backed persistent Qdrant volume (app/infrastructure/vector_store/qdrant_vector_store.py)
# via IAM authorization, not POSIX permissions - see efs.tf. ---
resource "aws_iam_role" "app_task" {
  name = "${var.project_name}-app-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "app_task_s3" {
  name = "${var.project_name}-app-s3-access"
  role = aws_iam_role.app_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.guidelines.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.guidelines.arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "app_task_efs" {
  name = "${var.project_name}-app-efs-access"
  role = aws_iam_role.app_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"]
      Resource = aws_efs_file_system.qdrant.arn
      Condition = {
        StringEquals = {
          "elasticfilesystem:AccessPointArn" = aws_efs_access_point.qdrant.arn
        }
      }
    }]
  })
}

# --- ui task role: no extra permissions attached - the ui only calls
# app over HTTP through the ALB (see ecs.tf's ui_environment), never
# touches S3/EFS/Secrets Manager directly. ---
resource "aws_iam_role" "ui_task" {
  name = "${var.project_name}-ui-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# --- GitHub Actions OIDC deploy role. Not referenced by any workflow
# in this PR (PR2 is Terraform only) - PR3 wires .github/workflows/ci.yml's
# deploy job to assume this role via aws-actions/configure-aws-credentials. ---
resource "aws_iam_openid_connect_provider" "github_actions" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # AWS validates GitHub's OIDC certificate chain directly for this
  # well-known provider URL; this is the documented fallback
  # thumbprint from AWS's GitHub OIDC setup guide and is not itself
  # security-critical.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions_deploy" {
  name = "${var.project_name}-github-actions-deploy"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github_actions.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        # Restricted to pushes to main only - matches
        # .github/workflows/ci.yml's "deploy only from main" rule
        # (CLAUDE.md Git Rules / Issue #21 requirement 9).
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repository}:ref:refs/heads/main"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name = "${var.project_name}-github-actions-deploy"
  role = aws_iam_role.github_actions_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "EcrPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = aws_ecr_repository.app.arn
      },
      {
        # ECS's RegisterTaskDefinition/DescribeTaskDefinition/
        # UpdateService/DescribeServices actions do not support
        # resource-level ARN restriction in all cases; scoping this
        # further (e.g. to just this cluster's two services) is a
        # documented possible future tightening, not done here to
        # keep this policy simple for a single-cluster deployment.
        Sid    = "EcsDeploy"
        Effect = "Allow"
        Action = [
          "ecs:RegisterTaskDefinition",
          "ecs:DescribeTaskDefinition",
          "ecs:UpdateService",
          "ecs:DescribeServices",
        ]
        Resource = "*"
      },
      {
        Sid    = "PassTaskRoles"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          aws_iam_role.ecs_task_execution.arn,
          aws_iam_role.app_task.arn,
          aws_iam_role.ui_task.arn,
        ]
      }
    ]
  })
}
