# Created with a placeholder value; the real OpenAI API key is set
# out-of-band (see docs/deployment-guide.md), never via Terraform -
# lifecycle.ignore_changes keeps a later `apply` from overwriting a
# manually-set real value with this placeholder again.
resource "aws_secretsmanager_secret" "llm_api_key" {
  name        = "${var.project_name}/llm-api-key"
  description = "MEDICAL_RAG_LLM_API_KEY - the OpenAI API key used when llm_provider=\"openai\"."
}

resource "aws_secretsmanager_secret_version" "llm_api_key" {
  secret_id     = aws_secretsmanager_secret.llm_api_key.id
  secret_string = "REPLACE_ME"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
