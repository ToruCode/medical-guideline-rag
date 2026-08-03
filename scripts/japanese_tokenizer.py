"""Tokenizer for BM25 over Japanese medical guideline text (Issue #24).

A hybrid, dependency-free scheme rather than a morphological analyzer
(SudachiPy/Janome/fugashi): this issue is a Dense-vs-Hybrid comparison
tool, not a production search feature, so reproducibility and test
simplicity across machines (no dictionary download, no native
extension) outweigh linguistic precision. See
docs/adr/0019-hybrid-search-comparison.md for the comparison against
morphological analyzers.

Two character classes are tokenized differently:

- ASCII alnum runs (drug names in Roman letters, device model
  numbers, abbreviations, numeric values - exactly the "exact term
  match matters" cases this issue is motivated by) are kept as single,
  lowercased tokens, so e.g. "SpO2" stays one token instead of being
  fragmented.
- Any other letter/digit run (hiragana/katakana/kanji, and any other
  non-ASCII script) is split into overlapping character bigrams (a
  single-character run becomes one unigram token), since whitespace
  does not mark word boundaries in Japanese and no dictionary is used
  to find them.

Punctuation and whitespace are never included in a token and simply
end whatever run was in progress.
"""

import unicodedata


def tokenize_japanese_text(text: str) -> list[str]:
    """Tokenizes text for BM25 indexing/querying. See module docstring.

    Returns [] only when text contains no letter/digit characters at
    all (e.g. "", whitespace, or punctuation only).
    """
    normalized = unicodedata.normalize("NFKC", text)

    tokens: list[str] = []
    ascii_run: list[str] = []
    other_run: list[str] = []

    for char in normalized:
        if char.isascii() and char.isalnum():
            _flush_other_run(other_run, tokens)
            ascii_run.append(char)
        elif char.isalpha() or char.isdigit():
            _flush_ascii_run(ascii_run, tokens)
            other_run.append(char)
        else:
            _flush_ascii_run(ascii_run, tokens)
            _flush_other_run(other_run, tokens)

    _flush_ascii_run(ascii_run, tokens)
    _flush_other_run(other_run, tokens)
    return tokens


def _flush_ascii_run(run: list[str], tokens: list[str]) -> None:
    if run:
        tokens.append("".join(run).lower())
        run.clear()


def _flush_other_run(run: list[str], tokens: list[str]) -> None:
    if not run:
        return
    if len(run) == 1:
        tokens.append(run[0])
    else:
        tokens.extend("".join(run[i : i + 2]) for i in range(len(run) - 1))
    run.clear()
