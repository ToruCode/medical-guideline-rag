"""Response schemas for the health check endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health."""

    status: str
    service: str
    version: str
    environment: str
