"""Real FAISS exact-search and snapshot persistence tests."""

import json
from uuid import uuid4

import pytest

from app.retrieval.dense import FaissStore
from app.retrieval.sparse import BM25Store
from app.retrieval.snapshot import DenseSnapshot, SnapshotError, SnapshotStorage


def _snapshot() -> DenseSnapshot:
    chunk_ids = (uuid4(), uuid4())
    return DenseSnapshot(
        version=1,
        store=FaissStore.build([[10.0, 0.0], [0.0, 2.0]]),
        chunk_ids=chunk_ids,
        embedding_model="deterministic-test-model",
        embedding_dimension=2,
        sparse_store=BM25Store.build(["apple fruit", "IndexFlatIP PostgreSQL"]),
        sparse_chunk_ids=chunk_ids,
        rrf_k=60,
    )


def test_faiss_build_mapping_and_cosine_normalization() -> None:
    snapshot = _snapshot()
    matches = snapshot.store.search([3.0, 0.0], 2)
    assert snapshot.store.count == len(snapshot.chunk_ids) == 2
    assert snapshot.store.dimension == 2
    assert matches[0].position == 0
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].position == 1
    assert matches[1].score == pytest.approx(0.0)


def test_snapshot_persists_reloads_and_preserves_results(tmp_path) -> None:
    snapshot = _snapshot()
    storage = SnapshotStorage(tmp_path / "versions")
    directory = storage.save_atomic(snapshot, chunking_config_hashes=["hash-b", "hash-a"])
    loaded = storage.load(1)

    assert directory.name == "000001"
    assert loaded.chunk_ids == snapshot.chunk_ids
    assert loaded.store.search([1.0, 0.0], 2) == snapshot.store.search([1.0, 0.0], 2)
    assert loaded.sparse_store.search("IndexFlatIP", 2) == snapshot.sparse_store.search(
        "IndexFlatIP", 2
    )
    assert (directory / "bm25_corpus.jsonl").is_file()
    assert (directory / "bm25_mapping.json").is_file()
    assert not list(directory.glob("*.pkl"))
    assert not list(directory.glob("*.pickle"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "deterministic-test-model"
    assert manifest["embedding_dimension"] == 2
    assert manifest["chunk_count"] == 2
    assert manifest["dense_index_type"] == "IndexFlatIP"
    assert manifest["dense_similarity_metric"] == "cosine"
    assert manifest["sparse_index_type"] == "BM25"
    assert manifest["fusion_strategy"] == "RRF"
    assert manifest["rrf_k"] == 60
    assert manifest["chunking_config_hashes"] == ["hash-a", "hash-b"]


def test_snapshot_rejects_mapping_count_mismatch(tmp_path) -> None:
    snapshot = _snapshot()
    storage = SnapshotStorage(tmp_path / "versions")
    directory = storage.save_atomic(snapshot, chunking_config_hashes=[])
    (directory / "faiss_mapping.json").write_text(
        json.dumps([str(snapshot.chunk_ids[0])]), encoding="utf-8"
    )
    with pytest.raises(SnapshotError, match="counts"):
        storage.load(1)


def test_snapshot_rejects_invalid_chunk_uuid(tmp_path) -> None:
    snapshot = _snapshot()
    storage = SnapshotStorage(tmp_path / "versions")
    directory = storage.save_atomic(snapshot, chunking_config_hashes=[])
    (directory / "faiss_mapping.json").write_text(json.dumps(["invalid", "uuid"]), encoding="utf-8")
    with pytest.raises(SnapshotError, match="missing, corrupt, or invalid"):
        storage.load(1)


def test_snapshot_rejects_dense_sparse_mapping_set_mismatch(tmp_path) -> None:
    snapshot = _snapshot()
    storage = SnapshotStorage(tmp_path / "versions")
    directory = storage.save_atomic(snapshot, chunking_config_hashes=[])
    replacement = uuid4()
    (directory / "bm25_mapping.json").write_text(
        json.dumps([str(snapshot.chunk_ids[0]), str(replacement)]), encoding="utf-8"
    )
    lines = (directory / "bm25_corpus.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["chunk_id"] = str(replacement)
    lines[1] = json.dumps(entry)
    (directory / "bm25_corpus.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="different chunks"):
        storage.load(1)


def test_snapshot_rejects_bm25_count_mismatch(tmp_path) -> None:
    snapshot = _snapshot()
    storage = SnapshotStorage(tmp_path / "versions")
    directory = storage.save_atomic(snapshot, chunking_config_hashes=[])
    first_line = (directory / "bm25_corpus.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (directory / "bm25_corpus.jsonl").write_text(first_line + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="missing, corrupt, or invalid"):
        storage.load(1)
