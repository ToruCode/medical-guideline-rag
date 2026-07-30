"""API layer.

Receives HTTP requests, validates request data, calls application use
cases, and converts results into HTTP responses.

Allowed dependencies: app.application, app.schemas, app.core.
Must not depend on Qdrant, PostgreSQL, LangChain, or any LLM directly.
"""
