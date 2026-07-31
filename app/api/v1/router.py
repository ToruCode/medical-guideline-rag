"""Aggregates all API v1 endpoint routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import documents, health, questions
from app.core.config import get_settings

api_router = APIRouter(prefix=get_settings().api_v1_prefix)
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(questions.router, tags=["questions"])
