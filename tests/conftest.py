import tempfile
import shutil
import pytest
import numpy as np
from scipy.sparse import csr_matrix


@pytest.fixture
def tmp_index_dir():
    """Temporary index directory, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="rag_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_texts():
    return [
        "Python is a high-level programming language.",
        "Python was created by Guido van Rossum in 1991.",
        "The language has a strong focus on code readability.",
        "FAISS is a library for efficient similarity search.",
        "RAG combines retrieval with generation for better answers.",
    ]


@pytest.fixture
def sample_metadatas():
    return [
        {"source": "doc1.md"},
        {"source": "doc1.md"},
        {"source": "doc1.md"},
        {"source": "doc2.md"},
        {"source": "doc3.md"},
    ]


@pytest.fixture
def mock_dense_vectors():
    """Return fixed (N, 1024) dense vectors."""
    rng = np.random.default_rng(42)
    return rng.random((5, 1024)).astype(np.float32)


@pytest.fixture
def mock_sparse_vectors():
    """Return fixed (N, 65536) scipy CSR sparse vectors."""
    rng = np.random.default_rng(42)
    data = rng.random(25).astype(np.float32)
    indices = rng.integers(0, 65536, 25)
    indptr = np.array([0, 5, 10, 15, 20, 25])
    return csr_matrix((data, indices, indptr), shape=(5, 65536))
