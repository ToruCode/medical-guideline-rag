"""Shared constants used across the application."""

from typing import Final

# Maximum length of CitationSchema.text_preview before truncation, so a
# citation's full chunk text (which may be long) never leaks wholesale
# into an API response meant only for debugging retrieval quality.
CITATION_TEXT_PREVIEW_MAX_CHARS: Final[int] = 200
