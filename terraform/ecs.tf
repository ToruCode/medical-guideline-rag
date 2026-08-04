resource "aws_ecs_cluster" "main" {
  name = var.project_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.project_name}/app"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${var.project_name}/ui"
  retention_in_days = var.log_retention_days
}

locals {
  ecr_image = "${aws_ecr_repository.app.repository_url}:${var.container_image_tag}"

  # Shared MEDICAL_RAG_* environment - app_environment/ui_environment
  # below extend this with service-specific values. Mirrors
  # .env.example field-for-field; see docs/adr/0029-aws-ecs-fargate-deployment.md.
  common_environment = [
    { name = "MEDICAL_RAG_APP_NAME", value = "Medical Guideline RAG" },
    { name = "MEDICAL_RAG_ENVIRONMENT", value = "production" },
    { name = "MEDICAL_RAG_LOG_LEVEL", value = "INFO" },
    { name = "MEDICAL_RAG_API_V1_PREFIX", value = "/api/v1" },
  ]

  app_environment = concat(local.common_environment, [
    { name = "MEDICAL_RAG_CHUNK_SIZE", value = "1000" },
    { name = "MEDICAL_RAG_CHUNK_OVERLAP", value = "200" },
    { name = "MEDICAL_RAG_PDF_EXTRACTOR", value = "pymupdf" },
    { name = "MEDICAL_RAG_EMBEDDING_PROVIDER", value = var.embedding_provider },
    { name = "MEDICAL_RAG_EMBEDDING_MODEL_NAME", value = "intfloat/multilingual-e5-base" },
    { name = "MEDICAL_RAG_LLM_PROVIDER", value = var.llm_provider },
    { name = "MEDICAL_RAG_LLM_MODEL_NAME", value = "gpt-4o-mini" },
    { name = "MEDICAL_RAG_LLM_TIMEOUT_SECONDS", value = "30" },
    { name = "MEDICAL_RAG_LLM_CONTEXT_MAX_CHARS", value = "6000" },
    # Forced to "qdrant"/"s3" regardless of any local-dev default,
    # mirroring compose.yaml's app service (ADR 0027) - see
    # docs/adr/0029-aws-ecs-fargate-deployment.md.
    { name = "MEDICAL_RAG_VECTOR_STORE_PROVIDER", value = "qdrant" },
    { name = "MEDICAL_RAG_VECTOR_STORE_PATH", value = "/app/qdrant_storage" },
    { name = "MEDICAL_RAG_VECTOR_STORE_COLLECTION_NAME", value = "guideline_chunks" },
    { name = "MEDICAL_RAG_STORAGE_PROVIDER", value = "s3" },
    { name = "MEDICAL_RAG_S3_BUCKET_NAME", value = aws_s3_bucket.guidelines.bucket },
  ])

  ui_environment = concat(local.common_environment, [
    # Reaches "app" through the ALB's public path routing (/api/*),
    # not a private service-discovery name - no ECS Service
    # Connect/Cloud Map namespace was added, to keep this stack
    # simpler. See docs/adr/0029-aws-ecs-fargate-deployment.md.
    { name = "MEDICAL_RAG_UI_API_BASE_URL", value = "http://${aws_lb.main.dns_name}" },
  ])
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.app_cpu
  memory                   = var.app_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.app_task.arn

  volume {
    name = "qdrant_storage"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.qdrant.id
      transit_encryption = "ENABLED"

      authorization_config {
        access_point_id = aws_efs_access_point.qdrant.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "app"
      image     = local.ecr_image
      essential = true

      portMappings = [{ containerPort = 8000, protocol = "tcp" }]

      environment = local.app_environment
      secrets = [
        { name = "MEDICAL_RAG_LLM_API_KEY", valueFrom = aws_secretsmanager_secret.llm_api_key.arn },
      ]

      mountPoints = [
        { sourceVolume = "qdrant_storage", containerPath = "/app/qdrant_storage", readOnly = false },
      ]

      # Same health check the Dockerfile's own HEALTHCHECK runs
      # locally/in Docker Compose - ECS uses this (and separately, the
      # ALB target group's own health check in alb.tf) instead of
      # Docker's HEALTHCHECK instruction, which ECS does not consult.
      healthCheck = {
        command     = ["CMD-SHELL", "uv run --frozen --no-dev python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 10
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "app"
        }
      }
    }
  ])
}

resource "aws_ecs_task_definition" "ui" {
  family                   = "${var.project_name}-ui"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ui_cpu
  memory                   = var.ui_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ui_task.arn

  container_definitions = jsonencode([
    {
      name      = "ui"
      image     = local.ecr_image
      essential = true
      # Overrides the image's default CMD (uvicorn) - the ECS
      # equivalent of compose.yaml's ui service `command:` override
      # (ADR 0027); both run the exact same image.
      command = [
        "uv", "run", "--frozen", "--no-dev",
        "streamlit", "run", "app/ui/streamlit_app.py",
        "--server.address", "0.0.0.0",
      ]

      portMappings = [{ containerPort = 8501, protocol = "tcp" }]

      environment = local.ui_environment

      # ui's own health check, not the shared image's HEALTHCHECK
      # (which polls :8000/api/v1/health - not served by this
      # container). Mirrors compose.yaml's ui healthcheck override.
      healthCheck = {
        command     = ["CMD-SHELL", "uv run --frozen --no-dev python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 10
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ui.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ui"
        }
      }
    }
  ])
}

# "Recreate" deployment, not ECS's default rolling deployment: the
# embedded/local-mode Qdrant store (ADR 0026) only allows one process
# to hold vector_store_path open at a time, so a normal rolling
# deployment (new task healthy before the old one stops) would have
# two tasks briefly holding the same EFS-mounted path open
# simultaneously and crash with VectorStoreUnavailableError.
# deployment_minimum_healthy_percent=0 lets ECS fully stop the old task
# before starting the new one - a deliberate, accepted brief outage on
# every app deploy. See docs/adr/0029-aws-ecs-fargate-deployment.md.
resource "aws_ecs_service" "app" {
  name             = "${var.project_name}-app"
  cluster          = aws_ecs_cluster.main.id
  task_definition  = aws_ecs_task_definition.app.arn
  desired_count    = 1
  launch_type      = "FARGATE"
  platform_version = "LATEST" # required for EFS volumes on Fargate

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}

# Ordinary rolling deployment - the ui task is stateless, unlike app.
resource "aws_ecs_service" "ui" {
  name            = "${var.project_name}-ui"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ui.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener.http]
}
