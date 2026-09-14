"""Deterministic tokenizer and real BM25 behavior tests."""

from app.retrieval.sparse import BM25Store, TOKENIZER_VERSION, tokenize


def test_tokenizer_is_deterministic_and_preserves_technical_identifiers() -> None:
    text = "FastAPI, IndexFlatIP! nomic-embed-text document_id /api/embed PostgreSQL."
    expected = [
        "fastapi",
        "indexflatip",
        "nomic-embed-text",
        "document_id",
        "/api/embed",
        "postgresql",
    ]
    assert tokenize(text) == expected
    assert tokenize(text) == tokenize(text)
    assert TOKENIZER_VERSION


def test_tokenizer_handles_punctuation_without_empty_tokens() -> None:
    assert tokenize("...hello; (world) C++") == ["hello", "world", "c"]
    assert tokenize("!!!") == []


def test_bm25_builds_expected_corpus_and_ranks_exact_identifier() -> None:
    store = BM25Store.build(
        ["ordinary ocean water", "FAISS IndexFlatIP vector search", "PostgreSQL metadata"]
    )
    assert store.count == 3
    assert store.tokenized_corpus[1] == ("faiss", "indexflatip", "vector", "search")
    matches = store.search("IndexFlatIP", 3)
    assert [match.position for match in matches] == [1]
    assert matches[0].score > 0


def test_bm25_excludes_lexical_candidate_with_zero_score() -> None:
    # With two documents and a term in exactly one, BM25Okapi assigns IDF 0.
    store = BM25Store.build(["system PostgreSQL", "meaning vectors"])
    assert store.search("system", 2) == []


def test_bm25_handles_absent_repeated_and_tokenless_terms() -> None:
    store = BM25Store.build(["apple apple fruit", "ocean water", "database metadata"])
    assert store.search("missing", 2) == []
    assert store.search("!!!", 2) == []
    assert store.search("apple", 2)[0].position == 0


def test_bm25_handles_single_and_tokenless_corpus_entries() -> None:
    single = BM25Store.build(["document_id"])
    assert single.search("document_id", 1)[0].position == 0
    tokenless = BM25Store.build(["!!!"])
    assert tokenless.search("anything", 1) == []
