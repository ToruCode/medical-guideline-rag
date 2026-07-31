from collections.abc import Iterator

import pytest
from app.api.dependencies import get_embedder, get_llm, get_vector_store
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Ensure get_settings() picks up fresh env vars in every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_shared_singleton_caches() -> Iterator[None]:
    """Ensure the process-wide Embedder/VectorStore/Llm singletons used by
    the API's dependency providers (app/api/dependencies.py) don't leak
    state between tests.
    """
    get_embedder.cache_clear()
    get_vector_store.cache_clear()
    get_llm.cache_clear()
    yield
    get_embedder.cache_clear()
    get_vector_store.cache_clear()
    get_llm.cache_clear()
