"""Deterministic BM25Okapi wrapper with explicit tokenized corpus storage."""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.retrieval.sparse.tokenizer import tokenize


class SparseStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SparseMatch:
    position: int
    score: float


class BM25Store:
    index_type = "BM25"

    def __init__(self, tokenized_corpus: list[list[str]]) -> None:
        if not tokenized_corpus:
            raise SparseStoreError("At least one corpus entry is required")
        if any(not isinstance(token, str) or not token for row in tokenized_corpus for token in row):
            raise SparseStoreError("BM25 corpus contains an invalid token")
        self.tokenized_corpus = tuple(tuple(row) for row in tokenized_corpus)
        self._token_sets = tuple(frozenset(row) for row in tokenized_corpus)
        self._index = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Store":
        return cls([tokenize(text) for text in texts])

    @property
    def count(self) -> int:
        return len(self.tokenized_corpus)

    def search(self, query: str, k: int) -> list[SparseMatch]:
        query_tokens = tokenize(query)
        if not query_tokens or self._index is None or k < 1:
            return []
        scores = self._index.get_scores(query_tokens)
        query_set = set(query_tokens)
        candidates = [
            SparseMatch(position=position, score=score)
            for position, corpus_tokens in enumerate(self._token_sets)
            if query_set.intersection(corpus_tokens)
            and (score := float(scores[position])) != 0.0
        ]
        candidates.sort(key=lambda result: (-result.score, result.position))
        return candidates[: min(k, self.count)]
