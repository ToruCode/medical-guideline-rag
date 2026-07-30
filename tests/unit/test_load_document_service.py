from pathlib import Path

from app.application.services.load_document import LoadDocumentService
from app.domain.models.document import DocumentPage


class FakePdfLoader:
    def __init__(self, pages: list[DocumentPage]) -> None:
        self._pages = pages
        self.received_path: Path | None = None

    def load(self, file_path: Path) -> list[DocumentPage]:
        self.received_path = file_path
        return self._pages


def test_execute_delegates_to_injected_loader() -> None:
    expected_pages = [
        DocumentPage(
            document_id="abc123",
            source_name="sample.pdf",
            source_path="/tmp/sample.pdf",
            page_number=1,
            text="hello",
            title=None,
        )
    ]
    fake_loader = FakePdfLoader(expected_pages)
    service = LoadDocumentService(fake_loader)
    file_path = Path("/tmp/sample.pdf")

    result = service.execute(file_path)

    assert result == expected_pages
    assert fake_loader.received_path == file_path
