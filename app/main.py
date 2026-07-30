"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.v1.router import api_router

app = FastAPI(title="Medical Guideline RAG")
app.include_router(api_router)
