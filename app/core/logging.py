"""Logging configuration for the application."""

import logging.config

from app.core.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure standard library logging for the application."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
            },
            "root": {
                "level": settings.log_level,
                "handlers": ["console"],
            },
        }
    )
