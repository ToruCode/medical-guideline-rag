"""Health check endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Return service liveness status."""
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
