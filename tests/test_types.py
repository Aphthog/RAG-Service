import numpy as np
from scipy.sparse import csr_matrix
from rag_service.types import Chunk, SearchResult, Embeddings, ChunkConfig


def test_chunk_creation():
    chunk = Chunk(
        text="hello world",
        metadata={"source": "test.md"},
        chunk_index=0,
        start_char=0,
        end_char=11,
    )
    assert chunk.text == "hello world"
    assert chunk.metadata == {"source": "test.md"}
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 11


def test_search_result_creation():
    sr = SearchResult(
        content="result text",
        metadata={"source": "test.md"},
        score=0.95,
        chunk_index=3,
    )
    assert sr.content == "result text"
    assert sr.score == 0.95
    assert sr.chunk_index == 3


def test_embeddings_creation():
    dense = np.random.random((3, 1024)).astype(np.float32)
    sparse = csr_matrix((3, 65536), dtype=np.float32)
    emb = Embeddings(dense=dense, sparse=sparse)
    assert emb.dense.shape == (3, 1024)
    assert emb.sparse.shape == (3, 65536)


def test_chunk_config_defaults():
    cfg = ChunkConfig()
    assert cfg.min_chars == 200
    assert cfg.max_chars == 800
    assert cfg.overlap == 80
    assert cfg.semantic_threshold == 0.6
    assert cfg.batch_size == 32


def test_chunk_config_custom():
    cfg = ChunkConfig(min_chars=100, max_chars=500, overlap=50, batch_size=16)
    assert cfg.min_chars == 100
    assert cfg.max_chars == 500
    assert cfg.overlap == 50
    assert cfg.batch_size == 16
