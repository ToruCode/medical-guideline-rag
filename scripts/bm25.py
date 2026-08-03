"""Self-contained Okapi BM25 scorer over a fixed, pre-tokenized corpus
(Issue #24's Hybrid Search comparison).

Implemented directly rather than via the `rank-bm25` package: the
formula is small and fully explainable, tokenization must be supplied
by our own Japanese tokenizer (scripts/japanese_tokenizer.py)
regardless of which BM25 implementation is used, and this project's
existing convention (InMemoryVectorStore's cosine similarity,
tests/support/evaluation/metrics.py's Recall/MRR) is to implement
small, well-understood algorithms directly rather than add a
dependency for them. See docs/adr/0019-hybrid-search-comparison.md.

k1/b use the conventional Okapi BM25 defaults (also rank-bm25's
defaults). idf uses the "+1" variant
(log((N - df + 0.5) / (df + 0.5) + 1)), which is always non-negative
for 0 <= df <= N, unlike the classic Robertson-Sparck Jones formula
(which can go negative for very common terms).
"""

import math
from collections import Counter

_K1 = 1.5
_B = 0.75


class Bm25Index:
    """Indexes a fixed corpus of (doc_id, tokens) pairs for BM25 scoring.

    doc_id values must be unique. tokens should come from the same
    tokenizer used to tokenize queries (tokenize_japanese_text), so
    query and document tokens are drawn from the same vocabulary.
    """

    def __init__(self, documents: list[tuple[str, list[str]]]) -> None:
        self._doc_ids = [doc_id for doc_id, _ in documents]
        self._term_frequencies: dict[str, Counter[str]] = {
            doc_id: Counter(tokens) for doc_id, tokens in documents
        }
        self._doc_lengths: dict[str, int] = {doc_id: len(tokens) for doc_id, tokens in documents}

        self._document_count = len(documents)
        self._average_doc_length = (
            sum(self._doc_lengths.values()) / self._document_count if self._document_count else 0.0
        )

        self._document_frequency: Counter[str] = Counter()
        for _, tokens in documents:
            for term in set(tokens):
                self._document_frequency[term] += 1

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        """BM25 score of doc_id against query_tokens.

        Raises KeyError for a doc_id not present in this index.
        """
        term_frequencies = self._term_frequencies[doc_id]
        length_norm = self._length_norm(self._doc_lengths[doc_id])

        score = 0.0
        for term in query_tokens:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = self._idf(term)
            numerator = frequency * (_K1 + 1)
            denominator = frequency + _K1 * length_norm
            score += idf * numerator / denominator
        return score

    def score_all(self, query_tokens: list[str]) -> dict[str, float]:
        """BM25 score of every indexed document against query_tokens."""
        return {doc_id: self.score(query_tokens, doc_id) for doc_id in self._doc_ids}

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
        """The k highest-scoring (doc_id, score) pairs, descending by
        score with a deterministic tie-break (ascending doc_id),
        matching InMemoryVectorStore.search()'s tie-break convention.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ranked = sorted(self.score_all(query_tokens).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def _length_norm(self, doc_length: int) -> float:
        if self._average_doc_length == 0:
            return 1.0
        return 1 - _B + _B * (doc_length / self._average_doc_length)

    def _idf(self, term: str) -> float:
        df = self._document_frequency.get(term, 0)
        return math.log((self._document_count - df + 0.5) / (df + 0.5) + 1)
