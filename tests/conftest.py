from collections.abc import Iterator

import pytest
from app.api import dependencies
from app.core.config import Settings, get_settings
from dotenv import load_dotenv

# Populates os.environ from .env (if present) before any test module is
# collected. Live tests (tests/integration/test_live_*.py) decide
# whether to skip by reading os.environ directly at collection time, not
# via Settings()/get_settings() (which already reads .env on its own,
# but only once constructed) - without this, filling in MEDICAL_RAG_*
# values in .env would have no effect on those tests unless the same
# values were also exported into the shell. load_dotenv() never
# overrides a variable that is already set in the real environment, so
# real exported values still take precedence.
load_dotenv()

# A developer's real .env sets these for local `make dev`/`make ui` use
# (e.g. MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers,
# MEDICAL_RAG_LLM_PROVIDER=openai, plus a real MEDICAL_RAG_LLM_API_KEY).
# Ordinary tests must not see those values - they rely on Settings' fake/
# no-key provider defaults, same as CI (which never has a .env file). See
# docs/adr/0030-isolate-pytest-provider-settings-from-dotenv.md.
_DOTENV_PROVIDER_ENV_KEYS = (
    "MEDICAL_RAG_EMBEDDING_PROVIDER",
    "MEDICAL_RAG_LLM_PROVIDER",
    "MEDICAL_RAG_LLM_API_KEY",
)


def _clear_settings_and_provider_caches() -> None:
    get_settings.cache_clear()
    dependencies.get_passage_embedder.cache_clear()
    dependencies.get_query_embedder.cache_clear()
    dependencies._get_sentence_transformer_model.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_llm.cache_clear()
    dependencies.get_object_storage.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_provider_settings_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip any real .env-sourced provider settings for the duration of
    each test, so Settings()/get_settings() falls back to the fake/no-key
    defaults regardless of the developer's local .env.

    Two separate things read the real .env, and both must be neutralised:

    1. os.environ, populated once for the whole session by this module's
       load_dotenv() (needed so live tests' collection-time
       os.environ.get(...) skip checks still work - see below).
       monkeypatch.delenv removes the three keys from it for this test.
    2. Settings itself (pydantic-settings' `model_config["env_file"]`),
       which re-reads the .env *file* directly on every `Settings()`
       call, independent of os.environ - so merely deleting the env var
       is not enough; Settings() would still pick the real value straight
       from the file. Pointing `env_file` at a nonexistent path for this
       test removes that second source too, leaving only real OS
       environment variables (e.g. ones a test sets itself via
       monkeypatch.setenv) and Settings' class defaults.

    monkeypatch restores both to their pre-test state once the test ends,
    so tests.conftest's module-level load_dotenv() effect on live tests
    (which read os.environ directly at collection time, before this
    fixture ever runs) is unaffected. Tests that need a real provider opt
    back in explicitly via their own monkeypatch.setenv(...) (e.g.
    tests/integration/test_live_rag_e2e.py), which runs after this
    fixture's setup and simply overrides it for that test - env vars set
    directly still take precedence over env_file when it is a real path.
    """
    for key in _DOTENV_PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    _clear_settings_and_provider_caches()
    yield
    _clear_settings_and_provider_caches()
