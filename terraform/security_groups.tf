resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb"
  description = "Allows inbound HTTP/HTTPS from the internet to the ALB."
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS (only used once domain_name/ACM are configured)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# Tasks run in public subnets with a public IP (no NAT Gateway - see
# network.tf), so this security group is the only thing preventing
# direct internet access to the app container: it allows inbound only
# from the ALB's own security group, never from 0.0.0.0/0.
resource "aws_security_group" "app" {
  name        = "${var.project_name}-app"
  description = "Allows inbound to the app (FastAPI) task only from the ALB."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "FastAPI from the ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-app"
  }
}

resource "aws_security_group" "ui" {
  name        = "${var.project_name}-ui"
  description = "Allows inbound to the ui (Streamlit) task only from the ALB."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Streamlit from the ALB"
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ui"
  }
}

resource "aws_security_group" "efs" {
  name        = "${var.project_name}-efs"
  description = "Allows inbound NFS only from the app task (the only service that mounts this EFS)."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "NFS from the app task"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-efs"
  }
}
