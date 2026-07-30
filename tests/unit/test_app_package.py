import importlib

import pytest

APP_SUBMODULES = (
    "app",
    "app.api",
    "app.application",
    "app.domain",
    "app.infrastructure",
    "app.core",
    "app.schemas",
)


@pytest.mark.parametrize("module_name", APP_SUBMODULES)
def test_app_subpackage_is_importable(module_name: str) -> None:
    importlib.import_module(module_name)
