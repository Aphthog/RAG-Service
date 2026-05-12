import os
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from rag_service.indexer import FAISSIndexer
from rag_service.types import Embeddings
from rag_service.errors import IndexLockedError


def make_embeddings(n: int) -> Embeddings:
    rng = np.random.default_rng(42)
    dense = rng.random((n, 1024)).astype(np.float32)
    data = rng.random(n * 5).astype(np.float32)
    indices = rng.integers(0, 65536, n * 5)
    indptr = np.arange(0, n * 5 + 1, 5)
    sparse = csr_matrix((data, indices, indptr), shape=(n, 65536))
    return Embeddings(dense=dense, sparse=sparse)


def make_chunks(n: int) -> list[dict]:
    return [
        {"text": f"chunk {i}", "metadata": {"src": f"doc{i//3}.md"}, "chunk_index": i, "start_char": i*100, "end_char": (i+1)*100}
        for i in range(n)
    ]


class TestFAISSIndexer:
    def test_build_creates_index_files(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        chunks = make_chunks(10)
        emb = make_embeddings(10)
        idx.build(chunks, emb)
        assert os.path.exists(os.path.join(tmp_index_dir, "test_tenant", "dense.faiss"))
        assert os.path.exists(os.path.join(tmp_index_dir, "test_tenant", "sparse.npz"))
        assert os.path.exists(os.path.join(tmp_index_dir, "test_tenant", "chunks.jsonl"))

    def test_chunk_count_after_build(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        idx.build(make_chunks(15), make_embeddings(15))
        assert idx.chunk_count() == 15

    def test_add_increments_count(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        idx.build(make_chunks(5), make_embeddings(5))
        assert idx.chunk_count() == 5
        idx.add(make_chunks(3), make_embeddings(3))
        assert idx.chunk_count() == 8

    def test_load_restores_index(self, tmp_index_dir):
        idx1 = FAISSIndexer(tmp_index_dir, "test_tenant")
        idx1.build(make_chunks(10), make_embeddings(10))
        idx2 = FAISSIndexer(tmp_index_dir, "test_tenant")
        idx2.load()
        assert idx2.chunk_count() == 10

    def test_search_dense_returns_results(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        n = 20
        idx.build(make_chunks(n), make_embeddings(n))
        q = np.random.default_rng(99).random((1, 1024)).astype(np.float32)
        results = idx.search_dense(q, k=5)
        assert len(results) == 5
        for chunk_id, score in results:
            assert 0 <= chunk_id < n
            assert isinstance(score, float)

    def test_search_sparse_returns_results(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        n = 20
        chunks = make_chunks(n)
        emb = make_embeddings(n)
        idx.build(chunks, emb)
        q = emb.sparse[0]
        results = idx.search_sparse(q, k=5)
        assert len(results) >= 1
        for chunk_id, score in results:
            assert 0 <= chunk_id < n

    def test_load_nonexistent_raises(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "nonexistent")
        with pytest.raises(FileNotFoundError):
            idx.load()

    def test_build_overwrites_previous_index(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        idx.build(make_chunks(10), make_embeddings(10))
        assert idx.chunk_count() == 10
        idx.build(make_chunks(3), make_embeddings(3))
        assert idx.chunk_count() == 3

    def test_chunks_metadata_preserved(self, tmp_index_dir):
        idx = FAISSIndexer(tmp_index_dir, "test_tenant")
        chunks = [
            {"text": "t1", "metadata": {"source": "a.md"}, "chunk_index": 0, "start_char": 0, "end_char": 2},
            {"text": "t2", "metadata": {"source": "b.md"}, "chunk_index": 1, "start_char": 0, "end_char": 2},
        ]
        idx.build(chunks, make_embeddings(2))
        chunk_path = os.path.join(tmp_index_dir, "test_tenant", "chunks.jsonl")
        with open(chunk_path, "r", encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        assert lines[0]["metadata"] == {"source": "a.md"}
        assert lines[1]["metadata"] == {"source": "b.md"}

    def test_lock_contention_raises_index_locked_error(self, tmp_index_dir, monkeypatch):
        idx1 = FAISSIndexer(tmp_index_dir, "locked_tenant")
        idx1.build(make_chunks(5), make_embeddings(5))
        from filelock import FileLock
        lock_path = os.path.join(tmp_index_dir, "locked_tenant", ".lock")
        blocker = FileLock(lock_path, timeout=0.1)
        blocker.acquire()
        try:
            with pytest.raises(IndexLockedError, match="locked"):
                idx1.build(make_chunks(3), make_embeddings(3))
        finally:
            blocker.release()
