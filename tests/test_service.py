import pytest
from rag_service.service import RAGService
from rag_service.embedder import MockEmbedder
from rag_service.errors import EmptyInputError, TenantNotFoundError


@pytest.fixture
def mock_rag(tmp_index_dir):
    return RAGService(index_dir=tmp_index_dir, embedder=MockEmbedder(seed=42), enable_rerank=False)


class TestRAGService:
    def test_index_creates_tenant(self, mock_rag):
        count = mock_rag.index_sync("kb1", texts=["hello world", "foo bar"], metadatas=[{"s": "a.md"}, {"s": "b.md"}])
        assert count > 0
        assert "kb1" in mock_rag.list_tenants()

    def test_search_returns_results(self, mock_rag):
        mock_rag.index_sync("kb1", texts=["Python is a programming language.", "FAISS is for vector search.", "Today is a nice day."])
        results = mock_rag.search_sync("programming", tenant="kb1", top_k=2)
        assert 0 < len(results) <= 2
        for r in results:
            assert r.content
            assert isinstance(r.score, float)
            assert isinstance(r.chunk_index, int)

    def test_multi_tenant_isolation(self, mock_rag):
        mock_rag.index_sync("tenant_a", texts=["content A1", "content A2"])
        mock_rag.index_sync("tenant_b", texts=["content B1", "content B2"])
        assert set(mock_rag.list_tenants()) == {"tenant_a", "tenant_b"}
        stats_a = mock_rag.index_stats("tenant_a")
        stats_b = mock_rag.index_stats("tenant_b")
        assert stats_a["chunk_count"] >= 1
        assert stats_b["chunk_count"] >= 1

    def test_incremental_index(self, mock_rag):
        count1 = mock_rag.index_sync("kb1", texts=["text one", "text two"])
        count2 = mock_rag.index_sync("kb1", texts=["text three"])
        assert count2 > count1

    def test_delete_index(self, mock_rag):
        mock_rag.index_sync("kb1", texts=["hello"])
        assert "kb1" in mock_rag.list_tenants()
        mock_rag.delete_index_sync("kb1")
        assert "kb1" not in mock_rag.list_tenants()

    def test_search_nonexistent_tenant_raises(self, mock_rag):
        with pytest.raises(TenantNotFoundError) as exc:
            mock_rag.search_sync("query", tenant="nonexistent")
        assert "nonexistent" in str(exc.value)

    def test_index_empty_texts_raises(self, mock_rag):
        with pytest.raises(EmptyInputError):
            mock_rag.index_sync("kb1", texts=[])

    def test_index_mismatched_lengths_raises(self, mock_rag):
        with pytest.raises(ValueError, match="length"):
            mock_rag.index_sync("kb1", texts=["a", "b"], metadatas=[{"s": "x.md"}])

    def test_index_without_metadatas(self, mock_rag):
        count = mock_rag.index_sync("kb1", texts=["hello world"])
        assert count > 0

    def test_index_stats(self, mock_rag):
        mock_rag.index_sync("kb1", texts=["a", "b", "c"])
        stats = mock_rag.index_stats("kb1")
        assert stats["chunk_count"] >= 2
        assert "index_dir" in stats
