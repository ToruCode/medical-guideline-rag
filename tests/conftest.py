from collections.abc import Iterator

import pytest
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Ensure get_settings() picks up fresh env vars in every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
