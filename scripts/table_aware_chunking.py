"""Rule-based table-aware chunking for Issue #30's chunking comparison.

Not a general table-recognition algorithm. This module targets one
specific, empirically observed pattern in this project's PyMuPDF text
extraction of Japanese guideline tables: each table cell tends to be
extracted as its own line (a table title line, several short
column-header lines, then many short data-cell lines, in sequence).
Detection here is a heuristic tuned to that pattern - it does not use
PDF layout/coordinate information, and it will both miss real tables
that don't match this line pattern and occasionally misclassify
unusually terse prose as a table. See
docs/adr/0021-table-aware-chunking-comparison.md for the full
rationale and known limitations, and
scripts/compare_chunking_strategies.py for how this is compared against
the existing fixed-size chunking.

Scope: like the production FixedSizeTextSplitter, this operates
strictly within one page's text - it never merges content across a
page boundary. A table (or an answer) that continues onto the next
page is a known, explicitly out-of-scope limitation, not something
this module attempts to bridge.
"""

import re
from dataclasses import dataclass

from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter

_TABLE_TITLE_PATTERN = re.compile(r"^(表|別表|補足表|付記表)\s*\d+")
_ANNOTATION_PATTERN = re.compile(r"^(注|※|備考|＊)\s*\d*[）)]?")
_HEADING_MAX_CHARS = 30
_SHORT_LINE_MAX_CHARS = 20
_NUMERIC_SYMBOL_RATIO_THRESHOLD = 0.3
_TABLE_ROW_RUN_MIN_LENGTH = 3


def _is_table_title_line(line: str) -> bool:
    return bool(_TABLE_TITLE_PATTERN.match(line.strip()))


def _is_annotation_line(line: str) -> bool:
    return bool(_ANNOTATION_PATTERN.match(line.strip()))


def _is_heading_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    return not stripped.endswith("。")


def _numeric_symbol_ratio(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0
    numeric_symbol_chars = sum(1 for ch in stripped if ch.isdigit() or not ch.isalnum())
    return numeric_symbol_chars / len(stripped)


def _is_table_row_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith("。"):
        return False
    if len(stripped) <= _SHORT_LINE_MAX_CHARS:
        return True
    return _numeric_symbol_ratio(line) >= _NUMERIC_SYMBOL_RATIO_THRESHOLD


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A run of ordinary prose lines - chunked by the existing
    FixedSizeTextSplitter, unchanged from the "fixed" strategy.
    """

    lines: list[str]


@dataclass(frozen=True, slots=True)
class TableBlock:
    """A heuristically detected table: an optional title line, an
    optional short heading line found immediately before it, the row
    lines themselves, and any annotation lines immediately following.
    """

    heading_context: str | None
    title: str | None
    rows: list[str]
    annotation_lines: list[str]


Block = TextBlock | TableBlock


class TableBlockDetector:
    """Heuristically splits one page's lines into TextBlocks and
    TableBlocks. See module docstring for the detection rules and their
    limitations - this is a comparison-only heuristic, not a table
    recognizer.
    """

    def detect(self, lines: list[str]) -> list[Block]:
        blocks: list[Block] = []
        pending_text: list[str] = []
        index = 0
        total = len(lines)

        def flush_text() -> None:
            if pending_text:
                blocks.append(TextBlock(lines=list(pending_text)))
                pending_text.clear()

        def heading_from_pending() -> str | None:
            for candidate in reversed(pending_text):
                if candidate.strip():
                    return candidate.strip() if _is_heading_like_line(candidate) else None
            return None

        while index < total:
            line = lines[index]
            if not line.strip():
                pending_text.append(line)
                index += 1
                continue

            is_title = _is_table_title_line(line)

            if not is_title:
                # A title line found within what would otherwise be a
                # contiguous row-like run takes priority: everything
                # before it (e.g. a heading line) is left as pending
                # text/heading context, and detection restarts at the
                # title itself, rather than being swallowed as the
                # first "row" of an untitled table.
                scan = index
                found_title_at: int | None = None
                while scan < total and (
                    _is_table_row_like_line(lines[scan]) or _is_table_title_line(lines[scan])
                ):
                    if _is_table_title_line(lines[scan]):
                        found_title_at = scan
                        break
                    scan += 1
                if found_title_at is not None:
                    pending_text.extend(lines[index:found_title_at])
                    index = found_title_at
                    continue

            row_start = index + 1 if is_title else index
            run_end = row_start
            while run_end < total and _is_table_row_like_line(lines[run_end]):
                run_end += 1
            run_length = run_end - row_start

            if is_title or run_length >= _TABLE_ROW_RUN_MIN_LENGTH:
                heading_context = heading_from_pending()
                flush_text()

                title = line.strip() if is_title else None
                rows = [lines[k].strip() for k in range(row_start, run_end) if lines[k].strip()]

                annotation: list[str] = []
                lookahead = run_end
                while lookahead < total and (
                    _is_annotation_line(lines[lookahead]) or not lines[lookahead].strip()
                ):
                    if lines[lookahead].strip():
                        annotation.append(lines[lookahead].strip())
                    lookahead += 1

                blocks.append(
                    TableBlock(
                        heading_context=heading_context,
                        title=title,
                        rows=rows,
                        annotation_lines=annotation,
                    )
                )
                index = lookahead
                continue

            pending_text.append(line)
            index += 1

        flush_text()
        return blocks


@dataclass(frozen=True, slots=True)
class TableAwareChunk:
    """One chunk produced by TableAwareTextSplitter, carrying the
    comparison-only metadata that the production Chunk model does not
    (and, per this issue's scope, must not) carry.

    split_trigger is None for non-table chunks; for table chunks it is
    "single_chunk" (the whole table fit in one chunk), "row_group_size"
    (cut because table_row_group_size rows was reached before
    table_max_chars), or "max_chars" (cut because table_max_chars was
    reached before table_row_group_size rows).
    """

    text: str
    is_table_chunk: bool
    heading_context: str | None
    table_title: str | None
    row_count: int | None
    split_trigger: str | None
    has_header_lines: bool = False
    is_header_duplicate: bool = False
    exceeded_max_chars_after_header: bool = False


class TableAwareTextSplitter:
    """Splits one page's text into TableAwareChunks: ordinary prose runs
    are delegated to the existing FixedSizeTextSplitter unchanged (so
    "fixed" and "table_aware" behave identically outside detected
    tables), and detected TableBlocks are chunked to keep heading/title/
    column-header context attached to every part of a split table (see
    module docstring; docs/adr/0021-table-aware-chunking-comparison.md
    for why table_max_chars takes priority over table_row_group_size).
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        table_max_chars: int,
        table_row_group_size: int,
    ) -> None:
        self._prose_splitter = FixedSizeTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._table_max_chars = table_max_chars
        self._table_row_group_size = table_row_group_size
        self._detector = TableBlockDetector()

    def split_page(self, text: str) -> list[TableAwareChunk]:
        if not text:
            return []

        chunks: list[TableAwareChunk] = []
        for block in self._detector.detect(text.split("\n")):
            if isinstance(block, TextBlock):
                block_text = "\n".join(block.lines).strip()
                for fragment in self._prose_splitter.split(block_text):
                    chunks.append(
                        TableAwareChunk(
                            text=fragment,
                            is_table_chunk=False,
                            heading_context=None,
                            table_title=None,
                            row_count=None,
                            split_trigger=None,
                        )
                    )
            else:
                chunks.extend(self._build_table_chunks(block))
        return chunks

    def _build_table_chunks(self, block: TableBlock) -> list[TableAwareChunk]:
        header_lines, data_rows = _split_header_and_data(block.rows)
        prefix_parts = [part for part in (block.heading_context, block.title) if part]
        prefix_parts.extend(header_lines)
        prefix_text = "\n".join(prefix_parts)
        annotation_text = "\n".join(block.annotation_lines)

        if not data_rows:
            full_text = "\n".join(part for part in (prefix_text, annotation_text) if part)
            if not full_text:
                return []
            return [
                TableAwareChunk(
                    text=full_text,
                    is_table_chunk=True,
                    heading_context=block.heading_context,
                    table_title=block.title,
                    row_count=0,
                    split_trigger="single_chunk",
                    has_header_lines=bool(header_lines),
                )
            ]

        groups = self._group_rows(data_rows, prefix_len=len(prefix_text))

        chunks: list[TableAwareChunk] = []
        for group_index, group_rows in enumerate(groups):
            is_last = group_index == len(groups) - 1
            parts = [prefix_text] if prefix_text else []
            parts.append("\n".join(group_rows))
            if is_last and annotation_text:
                parts.append(annotation_text)
            text = "\n".join(parts)

            if len(groups) == 1:
                trigger = "single_chunk"
            elif len(group_rows) >= self._table_row_group_size:
                trigger = "row_group_size"
            else:
                trigger = "max_chars"

            chunks.append(
                TableAwareChunk(
                    text=text,
                    is_table_chunk=True,
                    heading_context=block.heading_context,
                    table_title=block.title,
                    row_count=len(group_rows),
                    split_trigger=trigger,
                    has_header_lines=bool(header_lines),
                    is_header_duplicate=group_index > 0,
                    exceeded_max_chars_after_header=len(text) > self._table_max_chars,
                )
            )
        return chunks

    def _group_rows(self, data_rows: list[str], *, prefix_len: int) -> list[list[str]]:
        # table_max_chars takes priority over table_row_group_size: a
        # group is cut as soon as either limit would be exceeded by the
        # next row, whichever comes first.
        groups: list[list[str]] = []
        current: list[str] = []
        current_len = prefix_len
        for row in data_rows:
            row_len = len(row) + 1
            exceeds_chars = current and (current_len + row_len > self._table_max_chars)
            exceeds_rows = len(current) >= self._table_row_group_size
            if current and (exceeds_chars or exceeds_rows):
                groups.append(current)
                current = []
                current_len = prefix_len
            current.append(row)
            current_len += row_len
        if current:
            groups.append(current)
        return groups


def _split_header_and_data(rows: list[str]) -> tuple[list[str], list[str]]:
    """The leading run of rows that don't look numeric/symbol-heavy is
    treated as column-header lines (e.g. "グループ", "最大濃度（mg/L）");
    the first row that does look numeric/symbol-heavy starts the actual
    data. Heuristic only - see module docstring.
    """
    header: list[str] = []
    for row in rows:
        if _numeric_symbol_ratio(row) >= _NUMERIC_SYMBOL_RATIO_THRESHOLD:
            break
        header.append(row)
    return header, rows[len(header) :]
