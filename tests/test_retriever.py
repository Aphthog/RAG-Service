import numpy as np
import pytest
from scipy.sparse import csr_matrix
from rag_service.retriever import Retriever
from rag_service.embedder import MockEmbedder
from rag_service.types import Embeddings, SearchResult


class MockIndexer:
    """In-memory indexer for retriever testing."""

    def __init__(self, chunks: list[dict], emb: Embeddings):
        self._chunks = list(chunks)
        self._dense = emb.dense.copy()
        self._sparse = emb.sparse.copy()

    def search_dense(self, q, k):
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        v_norm = self._dense / (np.linalg.norm(self._dense, axis=1, keepdims=True) + 1e-8)
        scores = (q_norm @ v_norm.T)[0]
        k = min(k, len(scores))
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    def search_sparse(self, q, k):
        scores = (q @ self._sparse.T).toarray()[0]
        k = min(k, len(scores))
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def get_chunks(self):
        return list(self._chunks)

    def chunk_count(self):
        return len(self._chunks)


@pytest.fixture
def mock_embedder():
    return MockEmbedder(seed=42)


@pytest.fixture
def sample_chunks():
    return [
        {"text": f"Document chunk {i}", "metadata": {"source": f"doc{i//3}.md"}, "chunk_index": i, "start_char": i*100, "end_char": (i+1)*100}
        for i in range(10)
    ]


@pytest.fixture
def mock_indexer(sample_chunks, mock_embedder):
    emb = mock_embedder.encode([c["text"] for c in sample_chunks])
    return MockIndexer(sample_chunks, emb)


class TestRetriever:
    def test_retrieve_returns_search_results(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False)
        results = retriever.retrieve_sync("test query", top_k=5)
        assert 0 < len(results) <= 5
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.content
            assert isinstance(r.metadata, dict)
            assert 0 <= r.score <= 1
            assert isinstance(r.chunk_index, int)

    def test_retrieve_respects_top_k(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False)
        results = retriever.retrieve_sync("test query", top_k=3)
        assert len(results) == 3

    def test_empty_index_returns_empty(self, mock_embedder):
        empty_emb = Embeddings(
            dense=np.empty((0, 1024), dtype=np.float32),
            sparse=csr_matrix((0, 65536), dtype=np.float32),
        )
        empty_idx = MockIndexer([], empty_emb)
        retriever = Retriever(mock_embedder, empty_idx, enable_rerank=False)
        results = retriever.retrieve_sync("query", top_k=5)
        assert results == []

    def test_hybrid_scores_sorted_descending(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False, recall_top_k=10)
        results = retriever.retrieve_sync("test query", top_k=5)
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_rerank_disabled_skips_flagembedding(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False)
        results = retriever.retrieve_sync("test", top_k=3)
        assert len(results) <= 3

    def test_query_with_rewritten_queries(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False, recall_top_k=10)
        results = retriever.retrieve_sync("original query", rewritten_queries=["rewritten q1", "rewritten q2"], top_k=5)
        assert 0 < len(results) <= 5
        for r in results:
            assert isinstance(r, SearchResult)
