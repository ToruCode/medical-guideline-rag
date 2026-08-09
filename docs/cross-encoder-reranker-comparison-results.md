# Cross Encoder reranking comparison results

Aggregate Recall@1/3/5/MRR@5/latency measurements from
`scripts/compare_reranking_strategies.py`, run manually against real
guideline documents kept entirely locally, comparing Dense-only,
Hybrid (Dense+BM25), and Hybrid+Cross-Encoder-Rerank search under a
fixed retrieval configuration (`chunk_size=1000`, `chunk_overlap=200`,
`top_k=5`, `dense_candidate_k=20`, `bm25_candidate_k=20`, `alpha=0.7`,
`intfloat/multilingual-e5-base`, PDF extractor: PyMuPDF; see each entry
for the `reranker_candidate_k` it used - `scripts/compare_reranking_
strategies.py`'s default is `10`, chosen per the sweep recorded below).
See
`docs/evaluation-dataset-format.md` for the dataset format and
`docs/adr/0020-cross-encoder-reranker-comparison.md` for the full
design reasoning (candidate/rerank separation, Cross Encoder model
choice, latency measurement, per-question categorization). For
Dense-vs-Hybrid only, see
`docs/hybrid-search-comparison-results.md`/
`scripts/compare_retrieval_strategies.py`.

This is a comparison/measurement result only - it does not reflect a
production adoption decision. Production retrieval remains Dense-only
until a separate issue changes it.

Only aggregate numbers and measurement configuration are recorded
here. Document titles/publishers are anonymized (e.g. `"Guideline
A"`); the underlying PDF, dataset, questions, expected pages/chunks,
and any extracted text (including `text_preview`) are never committed -
see `CLAUDE.md`'s data/copyright rules.

To add an entry: run `scripts/compare_reranking_strategies.py`, review
the Markdown table it prints for anything identifying, then paste it
below (most recent first).

---

## reranker_candidate_k sweep: 5 / 10 / 20 / 30 (2026-08-09)

- Document: Guideline A
- Cases: 30 (same corrected dataset as the entry below)
- Configuration: `chunk_size=1000`, `chunk_overlap=200`, `top_k=5`,
  `dense_candidate_k=20`, `bm25_candidate_k=20`, `alpha=0.7`,
  `intfloat/multilingual-e5-base`,
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `device=cpu` - only
  `reranker_candidate_k` varies below. `candidate_k=5` is the baseline
  (same run as the entry below); 10/20/30 are additional runs.

| reranker_candidate_k | Recall@1 | Recall@3 | Recall@5 | MRR@5 | avg_reranking_latency_ms | avg_total_latency_ms |
|---:|---:|---:|---:|---:|---:|---:|
| 5 (baseline) | 0.78 | 0.95 | 0.95 | 0.87 | 712.2 | 723.9 |
| **10 (adopted)** | 0.78 | 0.93 | **0.97** | 0.87 | 1251.6 | 1262.5 |
| 20 | 0.78 | 0.93 | 0.93 | 0.86 | 2533.6 | 2544.9 |
| 30 | 0.78 | 0.93 | 0.93 | 0.86 | 3426.7 | 3438.6 |

**Adopted: `reranker_candidate_k=10`** (this repo's comparison tooling
default as of this entry). It is the only tested value with a net
Recall@5 gain over the `candidate_k=5` baseline: one previously-missing
case (its correct chunk ranked 6th-10th by Hybrid's fused score,
outside the `candidate_k=5` window fed to the reranker) is recovered.
`candidate_k=20`/`30` recover the exact same case but no more, while
exposing the reranker to enough extra distractor candidates that a
*different*, previously-fully-correct case gets pushed out of the
final top-5 - a net regression versus both the baseline and
`candidate_k=10`, for roughly 2-2.7x the latency. See
`docs/adr/0020-cross-encoder-reranker-comparison.md` for the reranking
design this sweep tunes.

One further case remains unresolved at every tested `candidate_k`
(recall@5 = 0 at 5/10/20/30 alike): its correct chunk's Hybrid-fused
rank is within reach of `candidate_k=10`+, so it does enter the
reranker's candidate set, but the Cross Encoder itself never scores it
into the top-5 - increasing `candidate_k` further is not expected to
resolve this one on its own. Left as a follow-up.

---

## Cross Encoder reranking comparison (2026-08-09)

- Document: Guideline A
- Cases: 30
- Configuration: `chunk_size=1000`, `chunk_overlap=200`, `top_k=5`,
  `dense_candidate_k=20`, `bm25_candidate_k=20`, `reranker_candidate_k=5`,
  `alpha=0.7`, `intfloat/multilingual-e5-base`,
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `device=cpu`,
  PDF extractor: PyMuPDF
- Note: this run uses a corrected evaluation dataset - one case's gold
  label was found, by inspecting the extracted PDF text directly, to
  point at the wrong page of a multi-page table (the question asked
  about one specific table row; the neighboring page did not contain
  that row at all). The label was corrected to the page that actually
  contains the row before this run. See the case-level detail (never
  committed) for the full before/after.

| strategy | alpha | reranker | Recall@1 | Recall@3 | Recall@5 | MRR@5 | avg_latency_ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dense |  | none | 0.65 | 0.88 | 0.95 | 0.78 | 87.1 |
| hybrid | 0.7 | none | 0.68 | 0.88 | 0.95 | 0.80 | 87.3 |
| hybrid_rerank | 0.7 | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 0.78 | 0.95 | 0.95 | 0.87 | 723.9 |

Hybrid -> Hybrid+Rerank per-question breakdown: rank improved (both
correct) 7, rank worsened (both correct) 3, unchanged/still correct 19,
unchanged/still missing 1, hybrid-correct-rerank-worse 0,
hybrid-incorrect-rerank-better 0.

Two known unresolved Recall@5 gaps remain across all three strategies
(pre-existing, unrelated to the gold-label fix above): one case whose
correct page ranks just outside every strategy's top-5 (a retrieval-depth/
candidate-window question, not yet resolved), and one case whose gold
answer spans two adjacent pages of the source PDF, of which only one
page is consistently retrieved.
