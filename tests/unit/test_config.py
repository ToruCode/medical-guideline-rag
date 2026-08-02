import pytest
from app.core.config import Settings, get_settings
from pydantic import ValidationError


def test_default_settings_values() -> None:
    settings = Settings()

    assert settings.app_name == "Medical Guideline RAG"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "local"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.pdf_extractor == "pymupdf"


def test_settings_can_be_overridden_by_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDICAL_RAG_APP_NAME", "Custom Name")
    monkeypatch.setenv("MEDICAL_RAG_ENVIRONMENT", "staging")
    monkeypatch.setenv("MEDICAL_RAG_DEBUG", "true")
    monkeypatch.setenv("MEDICAL_RAG_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.app_name == "Custom Name"
    assert settings.environment == "staging"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"


def test_settings_env_var_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("medical_rag_app_name", "Lowercase Name")

    settings = Settings()

    assert settings.app_name == "Lowercase Name"


def test_unknown_env_vars_do_not_break_settings_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDICAL_RAG_SOME_UNKNOWN_OPTION", "anything")

    settings = Settings()

    assert settings.app_name == "Medical Guideline RAG"


def test_invalid_environment_value_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDICAL_RAG_ENVIRONMENT", "not-a-real-environment")

    with pytest.raises(ValidationError):
        Settings()


def test_pdf_extractor_can_be_overridden_by_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDICAL_RAG_PDF_EXTRACTOR", "pypdf")

    settings = Settings()

    assert settings.pdf_extractor == "pypdf"


def test_invalid_pdf_extractor_value_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_RAG_PDF_EXTRACTOR", "not-a-real-extractor")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_returns_cached_instance() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second
