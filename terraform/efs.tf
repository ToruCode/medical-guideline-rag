resource "aws_efs_file_system" "qdrant" {
  creation_token = "${var.project_name}-qdrant-storage"
  encrypted      = true

  tags = {
    Name = "${var.project_name}-qdrant-storage"
  }
}

resource "aws_efs_mount_target" "qdrant" {
  count = length(aws_subnet.public)

  file_system_id  = aws_efs_file_system.qdrant.id
  subnet_id       = aws_subnet.public[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# The app container runs as root (Dockerfile has no USER directive),
# so this access point's POSIX user is root:root - access is actually
# restricted by aws_iam_role_policy.app_task_efs's AccessPointArn
# condition (IAM authorization, "iam = ENABLED" on the task
# definition's efs_volume_configuration in ecs.tf), not POSIX
# permissions.
resource "aws_efs_access_point" "qdrant" {
  file_system_id = aws_efs_file_system.qdrant.id

  posix_user {
    uid = 0
    gid = 0
  }

  root_directory {
    path = "/qdrant_storage"

    creation_info {
      owner_uid   = 0
      owner_gid   = 0
      permissions = "755"
    }
  }

  tags = {
    Name = "${var.project_name}-qdrant-storage"
  }
}
