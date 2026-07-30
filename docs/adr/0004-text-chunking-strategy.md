# 0004. Use a dependency-free, character-count based text chunker

## Status

Accepted

## Context

`DocumentPage`s produced by the PDF loading foundation (Issue #4,
`docs/adr/0003-pdf-extraction-library.md`) need to be split into
smaller `Chunk`s before they can be embedded and indexed for search.
No embedding model has been selected yet, so its tokenizer and token
limits are unknown at this point in the project.

## Decision

- Split text using a fixed-size, overlapping character window
  (`app/infrastructure/chunking/fixed_size_text_splitter.py`),
  configured by two settings: `chunk_size` and `chunk_overlap`
  (in characters, default 1000/200). Character count is used instead
  of token count because no embedding model/tokenizer has been chosen
  yet; token-aware limits would be guesswork at this stage.
- Do not use LangChain's `TextSplitter` implementations. The domain
  defines its own `TextSplitter` Protocol
  (`app/domain/ports/text_splitter.py`, `split(text: str) -> list[str]`)
  that infrastructure implementations satisfy, keeping the chunking
  algorithm's choice of library out of the domain and application
  layers, consistent with the LangChain-independence decision in ADR
  0003.
- Keep the `TextSplitter` Protocol's contract limited to raw text in,
  text fragments out. Carrying `DocumentPage` metadata
  (`document_id`, `page_number`, `source_name`, `source_path`,
  `title`) into the resulting `Chunk`s is handled separately by
  `ChunkDocumentService` in the application layer, so the splitting
  algorithm can be swapped without touching how metadata is attached.
- An empty page (`text == ""`) produces zero chunks, not a chunk with
  empty text.
- `chunk_size` and `chunk_overlap` are validated once, in
  `FixedSizeTextSplitter.__init__` (`chunk_size > 0`,
  `0 <= chunk_overlap < chunk_size`), raising
  `InvalidChunkConfigError` on invalid values. `Settings` itself does
  not duplicate this validation.

## Consequences

- Splitting text purely by character count does not respect sentence
  or paragraph boundaries; a chunk may end mid-sentence. This is
  acceptable for this foundational issue, and can be improved later by
  adding a sentence/paragraph-aware or tokenizer-aware implementation
  of the same `TextSplitter` Protocol, without changing the domain
  model or `ChunkDocumentService`.
- Because validation happens in `FixedSizeTextSplitter.__init__`
  rather than in `Settings`, an invalid `chunk_size`/`chunk_overlap`
  combination set via environment variables is only detected the first
  time a splitter is constructed from those settings, not at
  application startup. This trade-off keeps the invariant in a single
  place; it can be revisited if fail-fast startup validation becomes
  necessary.
- Once an embedding model is selected, `chunk_size`/`chunk_overlap`
  may need to move from characters to tokens; the setting names are
  unit-agnostic so this does not require a breaking rename.
