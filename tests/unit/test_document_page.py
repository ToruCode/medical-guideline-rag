import dataclasses

import pytest
from app.domain.models.document import DocumentPage


def _make_page() -> DocumentPage:
    return DocumentPage(
        document_id="abc123",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=1,
        text="hello",
        title="Sample",
    )


def test_document_page_is_frozen() -> None:
    page = _make_page()

    with pytest.raises(dataclasses.FrozenInstanceError):
        page.text = "changed"  # type: ignore[misc]


def test_document_page_allows_none_title() -> None:
    page = dataclasses.replace(_make_page(), title=None)

    assert page.title is None
