"""Unit tests for scripts/table_aware_chunking.py.

Uses only self-authored, fictional synthetic text - no real guideline
PDF content and no text_preview from any real document.
"""

from scripts.table_aware_chunking import (
    TableAwareTextSplitter,
    TableBlock,
    TableBlockDetector,
    TextBlock,
)


def test_detects_table_title() -> None:
    text = "表1　サンプル基準表\n項目\n基準値\n0.01\n0.02\n0.03"
    blocks = TableBlockDetector().detect(text.split("\n"))

    table_blocks = [b for b in blocks if isinstance(b, TableBlock)]
    assert len(table_blocks) == 1
    assert table_blocks[0].title == "表1　サンプル基準表"


def test_detects_consecutive_short_lines_as_table_without_title() -> None:
    text = "説明文はここまでです。\n項目A\n項目B\n0.1\n0.2\n0.3\nここから通常の説明文に戻ります。"
    blocks = TableBlockDetector().detect(text.split("\n"))

    table_blocks = [b for b in blocks if isinstance(b, TableBlock)]
    assert len(table_blocks) == 1
    assert table_blocks[0].title is None
    assert table_blocks[0].rows == ["項目A", "項目B", "0.1", "0.2", "0.3"]


def test_normal_prose_is_not_misclassified_as_table() -> None:
    text = (
        "これはテスト用の自己作成された説明文です。表ではない通常の文章を示しています。\n"
        "複数の文からなる段落で、短い行や数値の羅列は含まれていません。\n"
        "そのため表として誤検出されないことを期待します。"
    )
    blocks = TableBlockDetector().detect(text.split("\n"))

    assert all(isinstance(b, TextBlock) for b in blocks)


def test_heading_is_carried_into_table_chunk() -> None:
    text = "小見出し\n表1　サンプル基準表\n項目\n基準値\n0.01\n0.02"
    splitter = TableAwareTextSplitter(
        chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    chunks = splitter.split_page(text)

    table_chunks = [c for c in chunks if c.is_table_chunk]
    assert len(table_chunks) == 1
    assert table_chunks[0].heading_context == "小見出し"
    assert table_chunks[0].table_title == "表1　サンプル基準表"


def test_annotation_is_included_in_table_chunk() -> None:
    text = "表1　サンプル基準表\n項目\n基準値\n0.01\n0.02\n注） これはテスト用の注釈です．"
    splitter = TableAwareTextSplitter(
        chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    chunks = splitter.split_page(text)

    table_chunks = [c for c in chunks if c.is_table_chunk]
    assert len(table_chunks) == 1
    assert "注） これはテスト用の注釈です．" in table_chunks[0].text


def test_column_header_is_duplicated_across_split_chunks() -> None:
    rows = "\n".join(f"{i:03d}.0" for i in range(30))  # 30 numeric data rows
    text = f"表1　サンプル基準表\n項目\n基準値\n{rows}"
    splitter = TableAwareTextSplitter(
        chunk_size=1000, chunk_overlap=200, table_max_chars=100000, table_row_group_size=10
    )

    chunks = splitter.split_page(text)
    table_chunks = [c for c in chunks if c.is_table_chunk]

    assert len(table_chunks) == 3  # 30 rows / 10 per group
    assert table_chunks[0].is_header_duplicate is False
    assert all(c.is_header_duplicate for c in table_chunks[1:])
    for chunk in table_chunks:
        assert "表1　サンプル基準表" in chunk.text
        assert "項目" in chunk.text
        assert "基準値" in chunk.text


def test_long_table_split_stays_within_table_max_chars() -> None:
    rows = "\n".join(f"物質{i}　基準値{i}0.01mg/L" for i in range(50))
    text = f"表1　サンプル基準表\n項目\n基準値\n{rows}"
    splitter = TableAwareTextSplitter(
        chunk_size=1000, chunk_overlap=200, table_max_chars=200, table_row_group_size=1000
    )

    chunks = splitter.split_page(text)
    table_chunks = [c for c in chunks if c.is_table_chunk]

    assert len(table_chunks) > 1
    # Any part exceeding the budget is explicitly flagged, not silently produced.
    for chunk in table_chunks:
        if len(chunk.text) > 200:
            assert chunk.exceeded_max_chars_after_header is True


def test_empty_page_is_handled_safely() -> None:
    splitter = TableAwareTextSplitter(
        chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    assert splitter.split_page("") == []


def test_prose_blocks_are_split_like_fixed_size_splitter() -> None:
    long_prose = "これはテスト用の説明文です。" * 100  # long enough to force a split at chunk_size
    splitter = TableAwareTextSplitter(
        chunk_size=100, chunk_overlap=20, table_max_chars=1000, table_row_group_size=20
    )

    chunks = splitter.split_page(long_prose)

    assert all(not c.is_table_chunk for c in chunks)
    assert len(chunks) > 1
