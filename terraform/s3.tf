# Bucket name includes the account ID for global-uniqueness (S3 bucket
# names are unique across all of AWS, not just this account).
resource "aws_s3_bucket" "guidelines" {
  bucket = "${var.project_name}-guidelines-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "guidelines" {
  bucket = aws_s3_bucket.guidelines.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Old versions are expired (not the current/live version) after 90
# days, rather than kept indefinitely, to bound storage cost - see
# docs/adr/0028-s3-document-storage.md for why POST /documents/upload
# relies on versioning (not a generated key) to make same-name
# re-uploads non-destructive.
resource "aws_s3_bucket_lifecycle_configuration" "guidelines" {
  bucket = aws_s3_bucket.guidelines.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    # Empty filter = applies to every object in the bucket (no
    # prefix/tag scoping needed - there is only one kind of object
    # here, guideline PDFs). Required by the provider even when empty;
    # omitting it is deprecated and becomes an error in a future
    # provider version.
    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_public_access_block" "guidelines" {
  bucket = aws_s3_bucket.guidelines.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
