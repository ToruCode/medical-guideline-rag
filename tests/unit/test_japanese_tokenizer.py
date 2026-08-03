"""Unit tests for scripts/japanese_tokenizer.py.

Uses only inline, self-authored strings - no real guideline content.
"""

from scripts.japanese_tokenizer import tokenize_japanese_text


def test_returns_empty_list_for_empty_string() -> None:
    assert tokenize_japanese_text("") == []


def test_returns_empty_list_for_whitespace_and_punctuation_only() -> None:
    assert tokenize_japanese_text("   、。！？  ") == []


def test_plain_japanese_sentence_is_not_empty() -> None:
    tokens = tokenize_japanese_text("これは自己作成のサンプル文書です。")
    assert tokens != []


def test_ascii_alnum_run_becomes_one_lowercased_token() -> None:
    tokens = tokenize_japanese_text("SpO2")
    assert tokens == ["spo2"]


def test_ascii_and_japanese_runs_are_tokenized_independently() -> None:
    tokens = tokenize_japanese_text("SpO2 90%以上")
    assert "spo2" in tokens
    assert "90" in tokens
    # "以上" (2 chars) -> a single bigram token.
    assert "以上" in tokens


def test_japanese_run_is_split_into_overlapping_bigrams() -> None:
    tokens = tokenize_japanese_text("医療機関")
    assert tokens == ["医療", "療機", "機関"]


def test_single_japanese_character_becomes_one_unigram_token() -> None:
    tokens = tokenize_japanese_text("薬")
    assert tokens == ["薬"]


def test_punctuation_ends_a_run_without_bridging_across_it() -> None:
    # "薬剤" then a full-width comma then "投与" must not bigram across the
    # comma (e.g. must not produce "剤投").
    tokens = tokenize_japanese_text("薬剤、投与")
    assert "剤投" not in tokens
    assert tokens == ["薬剤", "投与"]


def test_fullwidth_ascii_is_normalized_before_tokenizing() -> None:
    # NFKC-normalizes fullwidth "Ａ１" to halfwidth "A1" before tokenizing,
    # so it is treated as a single ASCII token like its halfwidth form.
    assert tokenize_japanese_text("Ａ１") == tokenize_japanese_text("A1")
