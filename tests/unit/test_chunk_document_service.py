from app.application.services.chunk_document import ChunkDocumentService
from app.domain.models.document import DocumentPage


class FakeTextSplitter:
    def __init__(self, fragments_by_text: dict[str, list[str]]) -> None:
        self._fragments_by_text = fragments_by_text

    def split(self, text: str) -> list[str]:
        return self._fragments_by_text.get(text, [])


def _make_page(
    page_number: int, text: str, document_id: str = "doc-1", title: str | None = "Guideline"
) -> DocumentPage:
    return DocumentPage(
        document_id=document_id,
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=page_number,
        text=text,
        title=title,
    )


def test_carries_document_metadata_into_each_chunk() -> None:
    page = _make_page(1, "full page text")
    splitter = FakeTextSplitter({"full page text": ["part one", "part two"]})
    service = ChunkDocumentService(splitter)

    chunks = service.execute([page])

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.document_id == "doc-1"
        assert chunk.source_name == "sample.pdf"
        assert chunk.source_path == "/tmp/sample.pdf"
        assert chunk.page_number == 1
        assert chunk.title == "Guideline"


def test_chunk_index_is_zero_based_and_sequential_per_page() -> None:
    page = _make_page(1, "full page text")
    splitter = FakeTextSplitter({"full page text": ["a", "b", "c"]})
    service = ChunkDocumentService(splitter)

    chunks = service.execute([page])

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.text for chunk in chunks] == ["a", "b", "c"]


def test_multiple_pages_produce_chunks_with_correct_page_numbers() -> None:
    pages = [
        _make_page(1, "page one text"),
        _make_page(2, "page two text"),
    ]
    splitter = FakeTextSplitter(
        {
            "page one text": ["p1-a"],
            "page two text": ["p2-a", "p2-b"],
        }
    )
    service = ChunkDocumentService(splitter)

    chunks = service.execute(pages)

    assert [(chunk.page_number, chunk.chunk_index, chunk.text) for chunk in chunks] == [
        (1, 0, "p1-a"),
        (2, 0, "p2-a"),
        (2, 1, "p2-b"),
    ]


def test_empty_page_produces_no_chunks() -> None:
    page = _make_page(1, "")
    splitter = FakeTextSplitter({"": []})
    service = ChunkDocumentService(splitter)

    chunks = service.execute([page])

    assert chunks == []
