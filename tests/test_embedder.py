import numpy as np
from rag_service.embedder import MockEmbedder
from rag_service.types import Embeddings


class TestMockEmbedder:
    def test_encode_returns_correct_shapes(self):
        embedder = MockEmbedder(dim=1024, sparse_dim=65536)
        texts = ["hello world", "foo bar baz"]
        result = embedder.encode(texts)
        assert isinstance(result, Embeddings)
        assert result.dense.shape == (2, 1024)
        assert result.dense.dtype == np.float32
        assert result.sparse.shape == (2, 65536)

    def test_encode_empty_list(self):
        embedder = MockEmbedder()
        result = embedder.encode([])
        assert result.dense.shape == (0, 1024)
        assert result.sparse.shape == (0, 65536)

    def test_encode_single_text(self):
        embedder = MockEmbedder()
        result = embedder.encode(["hello"])
        assert result.dense.shape == (1, 1024)
        assert result.sparse.shape == (1, 65536)

    def test_dim_property(self):
        embedder = MockEmbedder(dim=512)
        assert embedder.dim == 512

    def test_deterministic_with_seed(self):
        embedder = MockEmbedder(seed=42)
        r1 = embedder.encode(["test"])
        r2 = embedder.encode(["test"])
        assert np.allclose(r1.dense, r2.dense)

    def test_different_texts_different_vectors(self):
        embedder = MockEmbedder(seed=42)
        r1 = embedder.encode(["hello"])
        r2 = embedder.encode(["world"])
        assert not np.allclose(r1.dense, r2.dense)
