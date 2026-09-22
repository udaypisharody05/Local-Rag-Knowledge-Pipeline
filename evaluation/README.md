# Evaluation

This directory provides a small deterministic evaluation for the running Local RAG API. It measures retrieval and response structure; it does not train models, use an LLM judge, or claim to prove factual correctness.

## Corpus preparation

Upload every file in `corpus/` through the normal document API. To exercise stale-snapshot deletion safety, include `deleted-content.txt` in a rebuild, then logically delete that document without rebuilding. All other corpus documents should remain active. Rebuild once before running the evaluator unless intentionally testing the stale deletion case.

The dataset contains 18 cases across semantic retrieval, exact identifiers, multi-source questions, citation-required answers, insufficient-context refusals, deleted-content exclusion, and prompt-injection resistance. Expected source names match the files in `corpus/`.

## Running against a local deployment

Set `RAG_API_KEY` in the process environment and optionally set `RAG_API_BASE_URL` (default `http://localhost:8000`). Then run:

```bash
python -m evaluation.evaluate
```

The evaluator calls `/search/hybrid` and `/query`, prints a concise summary, and writes `evaluation/results/latest.json`. Generated result JSON is ignored because it is deployment- and model-specific.

Metrics include Hit@K, MRR, Recall@K, forbidden-source exclusion, citation-label/metadata agreement, expected-source citations, expected answer terms, refusal-without-citation rate, and request latency. Retrieval scores from FAISS and BM25 are never compared directly.
