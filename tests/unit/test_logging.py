import logging

import pytest
from app.core.config import Settings
from app.core.logging import setup_logging


def test_setup_logging_runs_without_exception() -> None:
    setup_logging(Settings())


def test_setup_logging_applies_configured_log_level() -> None:
    setup_logging(Settings(log_level="DEBUG"))

    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_emits_formatted_record(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(Settings())
    logger = logging.getLogger("app.test_logging")

    logger.info("test message")

    output = capsys.readouterr().err
    assert "INFO" in output
    assert "app.test_logging" in output
    assert "test message" in output
