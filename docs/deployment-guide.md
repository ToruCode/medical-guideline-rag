# AWS deployment guide

Deploys the Medical Guideline RAG stack (FastAPI `app` + Streamlit
`ui`, backed by a persistent, EFS-mounted Qdrant store) to AWS ECS
Fargate via Terraform. See
`docs/adr/0029-aws-ecs-fargate-deployment.md` for the full design
reasoning behind everything below.

> **Status**: this guide currently covers Terraform setup only
> (PR2 of Issue #21). The GitHub Actions deploy job that builds and
> pushes the first image, and the corresponding CI/CD sections below,
> are completed in PR3.

## Cost warning

Running this stack continuously costs approximately **$40-70/month**
(two always-on minimal Fargate tasks, an Application Load Balancer,
EFS, CloudWatch Logs/Dashboard) even with zero traffic, **before**
adding a NAT Gateway (deliberately not used - see the ADR) or a custom
domain's own costs. Nothing in this repository runs `terraform apply`
automatically; you incur cost only once you choose to run it yourself.
Run `terraform destroy` (see below) to stop billing when you are done.

## Prerequisites

- An AWS account and credentials configured locally (e.g. `aws
  configure`, or an SSO profile) with permission to create the
  resources this stack defines (VPC, ECS, ALB, EFS, S3, IAM,
  Secrets Manager, CloudWatch, ECR).
- [Terraform](https://developer.hashicorp.com/terraform/downloads)
  `~> 1.7`.
- Docker (to build and push the first image manually - see
  "Bootstrap order" below; PR3 automates this for subsequent
  deploys).

## First-time setup

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars if you want to change project_name, aws_region,
# or set domain_name once you own one - see the file's own comments.

terraform init
terraform plan   # review every resource before creating anything
terraform apply  # creates real AWS resources and starts billing
```

## Bootstrap order (image-before-service problem)

The `app`/`ui` ECS services reference an image in the ECR repository
this same `terraform apply` creates - but ECS cannot start a healthy
task from a repository that has no image in it yet. Until PR3's
GitHub Actions deploy job exists, push an image manually the first
time:

```bash
# From the repository root, after `terraform apply` has created the
# ECR repository (see the ecr_repository_url output):
aws ecr get-login-password --region <aws_region> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<aws_region>.amazonaws.com

docker build -t <ecr_repository_url>:latest .
docker push <ecr_repository_url>:latest

# Then let ECS notice the new image:
aws ecs update-service --cluster medical-guideline-rag --service medical-guideline-rag-app --force-new-deployment
aws ecs update-service --cluster medical-guideline-rag --service medical-guideline-rag-ui --force-new-deployment
```

## Switching to a real OpenAI key

`terraform apply` creates the `llm_api_key` Secrets Manager secret with
a placeholder value (`REPLACE_ME`) - Terraform never receives or
stores your real key. Set it out-of-band:

```bash
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw llm_api_key_secret_arn)" \
  --secret-string "sk-..."
```

Then set `llm_provider = "openai"` in `terraform.tfvars` and re-apply,
and force a new `app` deployment (see the `aws ecs update-service ...
--force-new-deployment` command above) so the running task picks up
the new environment/secret value.

## Rollback

```bash
# List recent task definition revisions for the service you're rolling back:
aws ecs list-task-definitions --family-prefix medical-guideline-rag-app --sort DESC

# Point the service at a previous revision:
aws ecs update-service --cluster medical-guideline-rag --service medical-guideline-rag-app \
  --task-definition medical-guideline-rag-app:<previous-revision-number>
```

## Troubleshooting

- **EFS mount failures / app task stuck in `PENDING`**: confirm
  `platform_version = "LATEST"` on the `app` service (Fargate requires
  1.4.0+ for EFS) and that the EFS security group
  (`terraform/security_groups.tf`) allows inbound NFS (2049) from the
  `app` task's security group.
- **`VectorStoreUnavailableError` / `RuntimeError` about the vector
  store path already being open**: only one `app` task may hold the
  EFS-mounted `vector_store_path` open at a time (embedded Qdrant, see
  ADR 0026/0029). Do not manually raise `desired_count` above 1 for the
  `app` service.
- **ALB health check failures**: check `GET /api/v1/health` (app) or
  `/_stcore/health` (ui) return `200` from inside the task (CloudWatch
  Logs, `/ecs/medical-guideline-rag/app` or `/ui`); confirm the
  relevant security group allows the ALB's security group on the
  container's port.
- **Secrets Manager `AccessDeniedException`**: confirm the ECS task
  execution role (not the task role) has `secretsmanager:GetSecretValue`
  on the `llm_api_key` secret's exact ARN (`terraform/iam.tf`).
- **A deploy briefly returns errors from the `app` service**: expected
  - `app` uses a "recreate" deployment (ADR 0029), not a zero-downtime
    rolling one, because of the embedded Qdrant single-process
    constraint. `ui` deploys without downtime.

## Stopping / tearing down

```bash
cd terraform
terraform destroy  # removes every resource this stack created; stops billing
```

`terraform destroy` does **not** happen automatically as part of any
workflow in this repository - run it yourself when you want to stop
paying for this stack.
