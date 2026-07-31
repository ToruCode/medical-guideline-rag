from collections.abc import Iterator

import pytest
from app.api import dependencies
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Ensure get_settings() picks up fresh env vars in every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _clear_singleton_caches() -> None:
    dependencies.get_passage_embedder.cache_clear()
    dependencies.get_query_embedder.cache_clear()
    dependencies._get_sentence_transformer_model.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm.cache_clear()


@pytest.fixture(autouse=True)
def _clear_shared_singleton_caches() -> Iterator[None]:
    """Ensure the process-wide Embedder/VectorStore/Llm singletons used by
    the API's dependency providers (app/api/dependencies.py) don't leak
    state between tests.
    """
    _clear_singleton_caches()
    yield
    _clear_singleton_caches()
