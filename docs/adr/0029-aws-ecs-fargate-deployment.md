# 0029. AWS ECS Fargate deployment (Terraform infrastructure)

## Status

Accepted

## Context

Issue #21 (deploy to AWS) was split into three sequential PRs given its
scope: PR1 added S3 document storage at the application-code level
(`docs/adr/0028-s3-document-storage.md`) with no AWS resources
provisioned; this PR (PR2) adds the Terraform infrastructure itself,
with no `terraform apply` run and no CI/CD wiring yet (PR3); PR3 will
add the GitHub Actions deploy job that builds/pushes an image and
updates the ECS services this PR creates.

Two constraints from earlier issues shaped this design directly:

- **Embedded/local-mode Qdrant** (`app/infrastructure/vector_store/qdrant_vector_store.py`,
  `docs/adr/0026-persistent-vector-store.md`) only allows one process
  to hold `vector_store_path` open at a time - a second concurrent
  holder raises `RuntimeError` (mapped to `VectorStoreUnavailableError`).
  Issue #21 explicitly keeps this implementation (EFS-backed
  persistence, no Qdrant server container), so this constraint carries
  directly into how the `app` ECS service must be deployed.
- **Docker Compose (Issue #19, ADR 0027)** already established the
  pattern of one shared image for both `app` and `ui`, distinguished
  only by a container command override, and each service having its
  own health check pointed at its own port/path
  (`app`: `:8000/api/v1/health`, `ui`: `:8501/_stcore/health`, since
  the image's own `Dockerfile` `HEALTHCHECK` only covers `:8000`).

## Decision

- **`app`'s ECS service uses a "recreate" deployment, not ECS's default
  rolling deployment.** `desired_count = 1`,
  `deployment_minimum_healthy_percent = 0`,
  `deployment_maximum_percent = 100`, with
  `deployment_circuit_breaker { enable = true, rollback = true }`
  (`terraform/ecs.tf`). A normal rolling deployment starts the new task
  and waits for it to be healthy before stopping the old one - with
  both tasks mounting the same EFS-backed `vector_store_path`
  simultaneously, the second one to open it would crash. Setting
  `minimum_healthy_percent = 0` makes ECS fully stop the old task
  first, which means **every `app` deployment causes a brief (seconds
  to roughly a minute) outage** - a deliberate, accepted trade-off,
  not an oversight. `ui`'s service keeps ECS's ordinary rolling
  settings (`minimum_healthy_percent = 100`, `maximum_percent = 200`)
  since it is stateless.
- **No NAT Gateway; ECS tasks run directly in public subnets**
  (`terraform/network.tf`, `terraform/ecs.tf`'s
  `network_configuration { assign_public_ip = true }`). Isolation comes
  entirely from security groups (`terraform/security_groups.tf`): the
  `app`/`ui` task security groups permit inbound only from the ALB's
  own security group, never `0.0.0.0/0`. This was confirmed with the
  user as an explicit cost/isolation trade-off (a NAT Gateway costs
  roughly $32-70/month even at minimal traffic) given this project's
  stated "keep costs low for personal development" objective. Moving
  tasks into private subnets behind a NAT Gateway is a documented
  future hardening step, not implemented here.
- **One ECR repository, one image, for both `app` and `ui`.** The `ui`
  ECS task definition's container overrides `command` to run
  `streamlit run app/ui/streamlit_app.py --server.address 0.0.0.0`
  instead of the image's default `uvicorn` `CMD` - the exact ECS
  equivalent of `compose.yaml`'s `ui` service `command:` override
  (ADR 0027). No second Dockerfile, no second ECR repository.
- **`ui` reaches `app` through the public ALB
  (`MEDICAL_RAG_UI_API_BASE_URL = http://<alb-dns-name>`), not a
  private service-discovery name.** The ALB already has a path-based
  rule (`/api/*`, `/docs`, `/openapi.json`, `/redoc` -> the `app`
  target group; see below) that this reuses directly. This avoids
  adding an ECS Service Connect namespace or Cloud Map just to let two
  services on the same cluster reach each other, at the cost of `ui`'s
  traffic to `app` making an extra hop out to the ALB and back rather
  than staying on the cluster's internal network. Given this stack's
  "minimize operational complexity" priority and that the ALB is
  already always present and reachable (public subnets, no NAT), this
  was judged the simpler default; Service Connect remains a documented
  future option if internal-only traffic becomes a requirement.
- **Path-based routing on a single ALB, one listener pair (HTTP always,
  HTTPS conditionally)** (`terraform/alb.tf`): `/api/*`, `/docs`,
  `/openapi.json`, `/redoc` go to the `app` target group; everything
  else (default) goes to `ui`. One ALB and one DNS name for both
  services, rather than two ALBs (roughly double the fixed monthly ALB
  cost) or host-based routing (needs two subdomains).
- **HTTPS is entirely conditional on `var.domain_name`.** When null
  (the default - no domain owned yet, confirmed with the user), the
  ALB only has an HTTP listener on port 80; when set, Terraform also
  looks up an existing Route53 hosted zone for that domain (does not
  create one), requests and DNS-validates an ACM certificate, and adds
  an HTTPS listener (443) with the same path-based rule. There is
  **no automatic HTTP-to-HTTPS redirect** even once a domain is
  configured - implementing that would require the HTTP listener's
  `default_action` to conditionally take a different *shape*
  (`redirect` vs. `forward`) based on `var.domain_name`, which is
  awkward to express cleanly in Terraform; adding it is a documented
  future improvement, not done here to keep this listener's
  configuration simple.
- **Secrets Manager needs no application code at all** - confirmed
  during PR1's planning and unchanged here.
  `terraform/secrets.tf` creates `aws_secretsmanager_secret.llm_api_key`
  with a **placeholder** `"REPLACE_ME"` value
  (`lifecycle { ignore_changes = [secret_string] }`, so a later
  `terraform apply` never overwrites a real value an operator set
  manually); `terraform/ecs.tf`'s `app` container definition references
  it via the task-definition `secrets` block, which ECS resolves into
  a plain `MEDICAL_RAG_LLM_API_KEY` container environment variable at
  task startup - `Settings` (`app/core/config.py`) reads it exactly
  like any other environment variable, unaware of Secrets Manager's
  existence.
- **`var.llm_provider` defaults to `"fake"`, not `"openai"`.** Since
  the secret starts out as a placeholder, defaulting to `"openai"`
  would make the very first deployment fail OpenAI authentication at
  answer-generation time. Switching to real answers is a documented
  two-step operator action (put the real key into Secrets Manager, set
  `llm_provider = "openai"`, re-apply) - see `docs/deployment-guide.md`.
  `var.embedding_provider` defaults to `"fake"` for the same
  cost/simplicity reason (no model download, fits the default,
  modest `app_cpu`/`app_memory`).
- **EFS access via an access point + IAM authorization, not POSIX
  permissions.** The app container runs as root (no `USER` in
  `Dockerfile`), so the access point's POSIX user is root:root -
  meaningless as an access control on its own. Actual access control is
  the task role's `elasticfilesystem:ClientMount`/`ClientWrite` IAM
  policy, scoped to this one access point's ARN
  (`terraform/iam.tf`'s `app_task_efs` policy), combined with the task
  definition's `efs_volume_configuration { authorization_config { iam
  = "ENABLED" } }`. `platform_version = "LATEST"` is set on the `app`
  service because Fargate's EFS support requires platform version
  1.4.0 or later.
- **Terraform state is local for this PR** (no S3+DynamoDB remote
  backend). Bootstrapping a remote backend requires resources that
  themselves need to exist before `terraform init` can use them - a
  disproportionate amount of chicken-and-egg complexity for a personal
  project at this stage. `terraform/.gitignore` excludes
  `*.tfstate*`; `.terraform.lock.hcl` is committed (like `uv.lock` at
  the repo root) for reproducible provider versions. Migrating to a
  remote backend is a documented future improvement once/if
  multi-machine or team access to this state becomes necessary.
- **No AWS resources were created while implementing this PR.** Only
  `terraform fmt` and `terraform validate` (against a local,
  backend-disabled `terraform init`) were run - both require no AWS
  credentials and create nothing. `terraform plan`/`terraform apply`
  need real AWS credentials and are explicitly left to the user to run
  themselves, per this project's own workflow rules around
  cost-incurring actions.

## Consequences

- Every `app` deployment (a new image pushed, or any Terraform change
  to the `app` task definition) causes a brief outage while the old
  task stops and the new one starts - by design, not a bug. `ui`
  deploys without downtime.
- The whole stack runs in public subnets with no NAT Gateway; security
  depends entirely on the security groups in
  `terraform/security_groups.tf` being correct, since there is no
  network-layer isolation as a second line of defense. This is
  reasonable for personal-project scale and was an explicit,
  cost-driven choice - it should be revisited (private subnets + NAT,
  or fully removing public IPs in favor of an interface VPC endpoint
  set) before any production or multi-tenant use beyond personal
  development.
- `ui`'s traffic to `app` is public-internet-routed (through the ALB)
  even though both run in the same VPC/cluster; this is simpler than
  the alternative (ECS Service Connect/Cloud Map) but not the lowest-
  latency or most private option.
- Without an HTTP->HTTPS redirect, a client that only knows the HTTP
  URL keeps working over HTTP even after `domain_name`/HTTPS is
  configured - both listeners stay live side by side.
- `terraform apply` bootstrap order matters: the `app`/`ui` ECS
  services reference `aws_ecr_repository.app`'s URL in their task
  definitions, but the image itself must be pushed to that repository
  before the services can start healthy tasks - documented as an
  explicit bootstrap step in `docs/deployment-guide.md` (this PR adds
  the skeleton; PR3 completes it once the CI/CD pipeline exists to
  push that first image).
- Local Terraform state means only one operator/machine can safely run
  `terraform apply` at a time, and `terraform.tfstate` (which contains
  sensitive values such as resource ARNs) must be backed up and never
  committed - both called out in `docs/deployment-guide.md`.
