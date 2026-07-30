"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(api_router)
