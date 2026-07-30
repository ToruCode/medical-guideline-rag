"""Use case for splitting loaded document pages into chunks."""

from app.domain.models.chunk import Chunk
from app.domain.models.document import DocumentPage
from app.domain.ports.text_splitter import TextSplitter


class ChunkDocumentService:
    """Splits each DocumentPage's text via an injected TextSplitter."""

    def __init__(self, text_splitter: TextSplitter) -> None:
        self._text_splitter = text_splitter

    def execute(self, pages: list[DocumentPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            fragments = self._text_splitter.split(page.text)
            for index, fragment in enumerate(fragments):
                chunks.append(
                    Chunk(
                        document_id=page.document_id,
                        source_name=page.source_name,
                        source_path=page.source_path,
                        page_number=page.page_number,
                        chunk_index=index,
                        text=fragment,
                        title=page.title,
                    )
                )
        return chunks
