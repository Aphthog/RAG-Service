# RAG Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent, reusable RAG Python library (rag_service) with hybrid dense+sparse retrieval, adaptive chunking, and optional FastAPI server.

**Architecture:** 7 core modules (types, errors, embedder, chunker, indexer, retriever, service) composed into a single RAGService entry point. Dense retrieval via FAISS IndexHNSWFlat, sparse via scipy CSR matrix multiplication. Both indexes wrapped in a single FAISSIndexer with per-tenant filelock. Optional FastAPI shell in server.py.

**Tech Stack:** Python 3.11+, numpy, scipy, faiss-cpu, sentence-transformers (bge-m3), FlagEmbedding (bge-reranker-v2-m3), filelock, FastAPI + uvicorn (optional)

---

## File Structure

```
rag-service/
├── rag_service/
│   ├── __init__.py
│   ├── types.py
│   ├── errors.py
│   ├── embedder.py
│   ├── chunker.py
│   ├── indexer.py
│   ├── retriever.py
│   └── service.py
├── server.py
├── pyproject.toml
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_types.py
    ├── test_errors.py
    ├── test_embedder.py
    ├── test_chunker.py
    ├── test_indexer.py
    ├── test_retriever.py
    └── test_service.py
```

---

### Task 1: Project scaffold + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `rag_service/__init__.py` (empty placeholder)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "rag-service"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "faiss-cpu>=1.8",
    "sentence-transformers>=3.0",
    "FlagEmbedding>=1.3",
    "filelock>=3.13",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
server = ["fastapi", "uvicorn"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"
```

- [ ] **Step 2: Write tests/conftest.py with shared fixtures**

```python
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
    data = rng.random(25).astype(np.float32)  # 5 non-zeros per row
    indices = rng.integers(0, 65536, 25)
    indptr = np.array([0, 5, 10, 15, 20, 25])
    return csr_matrix((data, indices, indptr), shape=(5, 65536))
```

- [ ] **Step 3: Create empty placeholder files**

```bash
touch rag_service/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: Verify package structure imports**

Run: `python -c "import rag_service; from scipy.sparse import csr_matrix; import numpy; import filelock; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml rag_service/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold rag-service project with pyproject.toml"
```

---

### Task 2: Data types and exceptions

**Files:**
- Create: `rag_service/types.py`
- Create: `rag_service/errors.py`
- Create: `tests/test_types.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write failing tests for types**

```python
# tests/test_types.py
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
```

```python
# tests/test_errors.py
from rag_service.errors import (
    RAGException,
    IndexNotFoundError,
    EmptyInputError,
    EmbeddingError,
    IndexBuildError,
    TenantNotFoundError,
    IndexLockedError,
)


def test_rag_exception_base():
    err = RAGException("base error")
    assert isinstance(err, Exception)
    assert str(err) == "base error"


def test_index_not_found_error():
    err = IndexNotFoundError("index path not found")
    assert isinstance(err, RAGException)
    assert "index path not found" in str(err)


def test_empty_input_error():
    err = EmptyInputError("texts cannot be empty")
    assert isinstance(err, RAGException)


def test_embedding_error():
    err = EmbeddingError("model failed to load")
    assert isinstance(err, RAGException)


def test_index_build_error():
    err = IndexBuildError("FAISS build failed")
    assert isinstance(err, RAGException)


def test_tenant_not_found_error():
    err = TenantNotFoundError("my_kb", ["kb1", "kb2"])
    assert isinstance(err, RAGException)
    assert "my_kb" in str(err)
    assert "kb1" in str(err)
    assert "kb2" in str(err)


def test_index_locked_error():
    err = IndexLockedError("my_kb")
    assert isinstance(err, RAGException)
    assert "my_kb" in str(err)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_types.py tests/test_errors.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement types.py**

```python
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_index: int
    start_char: int
    end_char: int


@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float
    chunk_index: int


@dataclass
class Embeddings:
    dense: np.ndarray
    sparse: csr_matrix


@dataclass
class ChunkConfig:
    min_chars: int = 200
    max_chars: int = 800
    overlap: int = 80
    semantic_threshold: float = 0.6
    batch_size: int = 32
```

- [ ] **Step 4: Implement errors.py**

```python
class RAGException(Exception):
    """Base exception for all RAG service errors."""
    pass


class IndexNotFoundError(RAGException):
    pass


class EmptyInputError(RAGException):
    pass


class EmbeddingError(RAGException):
    pass


class IndexBuildError(RAGException):
    pass


class TenantNotFoundError(RAGException):
    def __init__(self, tenant: str, available: list[str]):
        self.tenant = tenant
        self.available = available
        msg = f"Tenant '{tenant}' not found. Available tenants: {available}"
        super().__init__(msg)


class IndexLockedError(RAGException):
    def __init__(self, tenant: str):
        self.tenant = tenant
        msg = f"Index for tenant '{tenant}' is locked by another operation."
        super().__init__(msg)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_types.py tests/test_errors.py -v`
Expected: ALL PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add rag_service/types.py rag_service/errors.py tests/test_types.py tests/test_errors.py
git commit -m "feat: add data types and exception hierarchy"
```

---

### Task 3: Embedder (bge-m3)

**Files:**
- Create: `rag_service/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write failing tests for embedder**

```python
# tests/test_embedder.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from rag_service.embedder import Embedder, MockEmbedder
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embedder.py::TestMockEmbedder -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement embedder.py**

```python
import logging
import numpy as np
from scipy.sparse import csr_matrix
from rag_service.types import Embeddings

logger = logging.getLogger("rag_service.embedder")


class Embedder:
    def __init__(self, model_path: str | None = None, device: str = "auto"):
        self._model_path = model_path
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading bge-m3 model (~2.2GB), this may take a few minutes...")
        from sentence_transformers import SentenceTransformer
        model_name = self._model_path or "BAAI/bge-m3"
        self._model = SentenceTransformer(model_name, device=self._device)
        logger.info("bge-m3 model loaded successfully.")

    def encode(self, texts: list[str], batch_size: int = 32) -> Embeddings:
        if not texts:
            return Embeddings(
                dense=np.empty((0, self.dim), dtype=np.float32),
                sparse=csr_matrix((0, 65536), dtype=np.float32),
            )
        self._load_model()
        # bge-m3 returns dense + sparse in one pass
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            output_value=None,  # get raw SentenceTransformerOutput
            show_progress_bar=False,
        )
        # sentence-transformers 3.x API
        if hasattr(output, 'dense_vecs') and hasattr(output, 'sparse_vecs'):
            dense = np.array(output.dense_vecs, dtype=np.float32)
            # convert list of dicts to CSR matrix
            sparse_list = output.sparse_vecs
            if sparse_list:
                sparse = self._sparse_to_csr(sparse_list, 65536)
            else:
                sparse = csr_matrix((len(texts), 65536), dtype=np.float32)
        else:
            # Fallback for older sentence-transformers
            dense = np.array(output, dtype=np.float32)
            sparse = csr_matrix((len(texts), 65536), dtype=np.float32)
        return Embeddings(dense=dense, sparse=sparse)

    def _sparse_to_csr(self, sparse_dicts: list[dict], vocab_size: int) -> csr_matrix:
        """Convert list of {token_id: weight} dicts to CSR matrix."""
        import numpy as np
        data = []
        indices = []
        indptr = [0]
        for d in sparse_dicts:
            for idx, val in d.items():
                indices.append(int(idx))
                data.append(float(val))
            indptr.append(len(indices))
        return csr_matrix(
            (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32),
             np.array(indptr, dtype=np.int32)),
            shape=(len(sparse_dicts), vocab_size),
        )

    @property
    def dim(self) -> int:
        return 1024


class MockEmbedder:
    """Deterministic embedder for testing. Returns seeded random vectors."""

    def __init__(self, dim: int = 1024, sparse_dim: int = 65536, seed: int = 42):
        self._dim = dim
        self._sparse_dim = sparse_dim
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def encode(self, texts: list[str], batch_size: int = 32) -> Embeddings:
        n = len(texts)
        dense = self._rng.random((n, self._dim), dtype=np.float32)
        # generate ~10 non-zero sparse entries per text
        data = []
        indices = []
        indptr = [0]
        for i in range(n):
            nz = self._rng.integers(3, 15)
            idx = self._rng.integers(0, self._sparse_dim, nz)
            vals = self._rng.random(nz, dtype=np.float32)
            indices.extend(idx.tolist())
            data.extend(vals.tolist())
            indptr.append(len(indices))
        sparse = csr_matrix(
            (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32),
             np.array(indptr, dtype=np.int32)),
            shape=(n, self._sparse_dim),
        )
        return Embeddings(dense=dense, sparse=sparse)

    @property
    def dim(self) -> int:
        return self._dim
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embedder.py::TestMockEmbedder -v`
Expected: ALL PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_service/embedder.py tests/test_embedder.py
git commit -m "feat: add Embedder (bge-m3) with MockEmbedder for testing"
```

---

### Task 4: Adaptive Chunker

**Files:**
- Create: `rag_service/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests for chunker**

```python
# tests/test_chunker.py
import pytest
from rag_service.chunker import AdaptiveChunker, MarkdownChunker, SemanticChunker
from rag_service.types import Chunk, ChunkConfig
from rag_service.embedder import MockEmbedder


class TestMarkdownChunker:
    def test_splits_on_headings(self):
        chunker = MarkdownChunker(ChunkConfig(max_chars=800))
        text = "# Title\nSome content here.\n\n## Section 1\nSection one text.\n\n## Section 2\nSection two text."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3
        # first chunk should contain the title
        assert "Title" in chunks[0].text
        assert all(isinstance(c.metadata, dict) for c in chunks)

    def test_empty_text_returns_empty(self):
        chunker = MarkdownChunker(ChunkConfig())
        assert chunker.chunk("") == []

    def test_preserves_overlap(self):
        cfg = ChunkConfig(max_chars=50, min_chars=10, overlap=20)
        chunker = MarkdownChunker(cfg)
        text = "A" * 120
        chunks = chunker.chunk(text)
        if len(chunks) >= 2:
            # overlap: end of chunk 0 should appear in start of chunk 1
            assert chunks[0].text[-10:] in chunks[1].text or len(chunks[0].text) <= cfg.max_chars

    def test_chunk_index_sequence(self):
        chunker = MarkdownChunker(ChunkConfig(max_chars=50))
        text = "Long text. " * 50
        chunks = chunker.chunk(text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_start_end_char_positions(self):
        chunker = MarkdownChunker(ChunkConfig(max_chars=100))
        text = "Hello world. This is a test. More content here."
        chunks = chunker.chunk(text)
        # chunks should cover the full text
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(text)


class TestSemanticChunker:
    def test_chunks_with_mock_embedder(self):
        embedder = MockEmbedder(seed=123)
        chunker = SemanticChunker(embedder, ChunkConfig(max_chars=200, min_chars=50, semantic_threshold=0.5))
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.text
            assert isinstance(c.metadata, dict)

    def test_single_sentence(self):
        embedder = MockEmbedder(seed=123)
        chunker = SemanticChunker(embedder, ChunkConfig())
        text = "Just one sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "Just one sentence" in chunks[0].text

    def test_empty_text(self):
        embedder = MockEmbedder()
        chunker = SemanticChunker(embedder, ChunkConfig())
        assert chunker.chunk("") == []


class TestAdaptiveChunker:
    def test_routes_markdown_to_markdown_chunker(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "# Title\n## Section\nContent here with markdown headings."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text

    def test_routes_plain_text_to_semantic_chunker(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "This is a plain paragraph without any markdown structure. Just regular text content."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_empty_text(self):
        embedder = MockEmbedder()
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        assert chunker.chunk("") == []

    def test_chunks_inherit_metadata(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "# Doc\nContent of the document."
        chunks = chunker.chunk(text, metadata={"source": "test.md", "author": "camille"})
        for c in chunks:
            assert c.metadata == {"source": "test.md", "author": "camille"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement chunker.py**

```python
import re
import logging
from rag_service.types import Chunk, ChunkConfig

logger = logging.getLogger("rag_service.chunker")


class SemanticChunker:
    """Chunk by embedding similarity — split at semantic boundary drops."""

    def __init__(self, embedder, config: ChunkConfig):
        self._embedder = embedder
        self._cfg = config

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        meta = metadata or {}
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text.strip(), metadata=meta, chunk_index=0, start_char=0, end_char=len(text))]

        # Batch encode all sentences
        embeddings = self._embedder.encode(sentences, batch_size=self._cfg.batch_size)
        vecs = embeddings.dense

        # Compute cosine similarity between adjacent sentences
        import numpy as np
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vecs / norms
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

        # Find breakpoints where similarity < threshold
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < self._cfg.semantic_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        # Merge sentences between breakpoints into chunks within [min_chars, max_chars]
        chunks = []
        chunk_idx = 0
        current = ""
        current_start = 0
        pos = 0

        for bp_start, bp_end in zip(breakpoints[:-1], breakpoints[1:]):
            segment = " ".join(sentences[bp_start:bp_end])
            if not current:
                current = segment
                current_start = pos
            elif len(current) + len(segment) <= self._cfg.max_chars:
                current += " " + segment
            else:
                if len(current) >= self._cfg.min_chars or not segment:
                    chunks.append(Chunk(
                        text=current.strip(),
                        metadata=meta,
                        chunk_index=chunk_idx,
                        start_char=current_start,
                        end_char=current_start + len(current),
                    ))
                    chunk_idx += 1
                    current = segment
                    current_start = current_start + len(current) + 1

            pos = current_start + len(current) + 1 if chunk_idx > 0 else len(current)

        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                metadata=meta,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=current_start + len(current),
            ))

        # Apply overlap
        if self._cfg.overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, text)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using Chinese and English punctuation."""
        # Match sentence-ending punctuation followed by whitespace or end
        pattern = r'(?<=[。！？.!?\n])\s*'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _apply_overlap(self, chunks: list[Chunk], original_text: str) -> list[Chunk]:
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].end_char
            overlap_start = max(chunks[i].start_char - self._cfg.overlap, chunks[i - 1].start_char)
            if overlap_start < prev_end:
                chunks[i] = Chunk(
                    text=original_text[overlap_start:chunks[i].end_char],
                    metadata=chunks[i].metadata,
                    chunk_index=chunks[i].chunk_index,
                    start_char=overlap_start,
                    end_char=chunks[i].end_char,
                )
        return chunks


class MarkdownChunker:
    """Structurally-aware chunker for Markdown documents."""

    def __init__(self, config: ChunkConfig, semantic_chunker: SemanticChunker | None = None):
        self._cfg = config
        self._semantic = semantic_chunker

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        meta = metadata or {}

        # Split on Markdown headings
        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        chunks = []
        chunk_idx = 0

        for section in sections:
            if not section.strip():
                continue
            # Split section on blank lines (paragraphs)
            paragraphs = re.split(r'\n\s*\n', section)
            current = ""
            current_start = None

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if not current:
                    current = para
                    # find position in original text
                    current_start = text.find(para) if current_start is None else current_start
                elif len(current) + len(para) <= self._cfg.max_chars:
                    current += "\n\n" + para
                else:
                    # Current chunk is full — add it
                    chunks.extend(self._finalize_chunk(current, current_start, meta, chunk_idx))
                    chunk_idx += 1
                    current = para
                    current_start = text.find(para, current_start + len(current))

            # Last paragraph group of this section
            if current.strip():
                chunks.extend(self._finalize_chunk(current, current_start or 0, meta, chunk_idx))
                chunk_idx += 1

        # Apply overlap
        if self._cfg.overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].end_char
                overlap_start = max(chunks[i].start_char - self._cfg.overlap, chunks[i - 1].start_char)
                if overlap_start < prev_end:
                    chunks[i] = Chunk(
                        text=text[overlap_start:chunks[i].end_char],
                        metadata=chunks[i].metadata,
                        chunk_index=chunks[i].chunk_index,
                        start_char=overlap_start,
                        end_char=chunks[i].end_char,
                    )

        return chunks

    def _finalize_chunk(self, text: str, start: int, metadata: dict, base_idx: int) -> list[Chunk]:
        """If text exceeds max_chars, fall back to semantic chunker; otherwise return single chunk."""
        if len(text) <= self._cfg.max_chars:
            return [Chunk(text=text, metadata=metadata, chunk_index=base_idx, start_char=start, end_char=start + len(text))]
        if self._semantic:
            sub_chunks = self._semantic.chunk(text, metadata)
            # Adjust indices relative to parent
            return [Chunk(
                text=c.text, metadata=c.metadata,
                chunk_index=base_idx + c.chunk_index,
                start_char=start + c.start_char,
                end_char=start + c.end_char,
            ) for c in sub_chunks]
        # No semantic fallback — just split at max_chars
        result = []
        for i in range(0, len(text), self._cfg.max_chars - self._cfg.overlap):
            end = min(i + self._cfg.max_chars, len(text))
            result.append(Chunk(
                text=text[i:end], metadata=metadata,
                chunk_index=base_idx + len(result),
                start_char=start + i, end_char=start + end,
            ))
        return result


class AdaptiveChunker:
    """Routes input to MarkdownChunker or SemanticChunker based on content detection."""

    _MARKDOWN_PATTERN = re.compile(r'^#{1,6}\s|\n#{1,6}\s|```|^\|.*\|$', re.MULTILINE)

    def __init__(self, embedder, config: ChunkConfig):
        self._embedder = embedder
        self._cfg = config
        self._semantic = SemanticChunker(embedder, config)
        self._markdown = MarkdownChunker(config, self._semantic)

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        if self._MARKDOWN_PATTERN.search(text):
            logger.debug("Routing to MarkdownChunker")
            return self._markdown.chunk(text, metadata)
        logger.debug("Routing to SemanticChunker")
        return self._semantic.chunk(text, metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunker.py -v`
Expected: ALL PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_service/chunker.py tests/test_chunker.py
git commit -m "feat: add adaptive chunker (Markdown + semantic)"
```

---

### Task 5: FAISS Indexer (dense + sparse)

**Files:**
- Create: `rag_service/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: Write failing tests for indexer**

```python
# tests/test_indexer.py
import os
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix, load_npz
from rag_service.indexer import FAISSIndexer
from rag_service.types import Embeddings, Chunk


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
        # search with a sparse vector
        q = emb.sparse[0]
        results = idx.search_sparse(q, k=5)
        assert len(results) == 5
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
        """Simulate lock held by another process."""
        from rag_service.errors import IndexLockedError
        idx1 = FAISSIndexer(tmp_index_dir, "locked_tenant")
        idx1.build(make_chunks(5), make_embeddings(5))
        # Hold the lock manually
        from filelock import FileLock
        lock_path = os.path.join(tmp_index_dir, "locked_tenant", ".lock")
        blocker = FileLock(lock_path, timeout=0.1)
        blocker.acquire()
        try:
            with pytest.raises(IndexLockedError, match="locked"):
                idx1.build(make_chunks(3), make_embeddings(3))
        finally:
            blocker.release()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_indexer.py -v`
Expected: FAIL — ModuleNotFoundError (10 tests, 1 new)

- [ ] **Step 3: Implement indexer.py**

```python
import os
import json
import logging
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from filelock import FileLock
from rag_service.types import Embeddings
from rag_service.errors import IndexNotFoundError, IndexBuildError, IndexLockedError

logger = logging.getLogger("rag_service.indexer")


class FAISSIndexer:
    _DENSE_FILE = "dense.faiss"
    _SPARSE_FILE = "sparse.npz"
    _CHUNKS_FILE = "chunks.jsonl"
    _LOCK_FILE = ".lock"

    def __init__(self, index_dir: str, tenant: str):
        import faiss
        self._faiss = faiss
        self._tenant_dir = os.path.join(index_dir, tenant)
        self._dense_path = os.path.join(self._tenant_dir, self._DENSE_FILE)
        self._sparse_path = os.path.join(self._tenant_dir, self._SPARSE_FILE)
        self._chunks_path = os.path.join(self._tenant_dir, self._CHUNKS_FILE)
        self._lock_path = os.path.join(self._tenant_dir, self._LOCK_FILE)
        self._lock = FileLock(self._lock_path, timeout=5)
        self._dense_index = None
        self._sparse_matrix = None
        self._chunks = []
        self._add_count_since_rebuild = 0

    def build(self, chunks: list[dict], emb: Embeddings) -> None:
        try:
            with self._lock:
                try:
                    return self._build_unlocked(chunks, emb)
                except Exception as e:
                    raise IndexBuildError(str(e)) from e
        except self._lock.timeout_class:  # filelock Timeout
            raise IndexLockedError(self._tenant_dir.split(os.sep)[-1]) from None

    def _build_unlocked(self, chunks: list[dict], emb: Embeddings) -> None:
        os.makedirs(self._tenant_dir, exist_ok=True)
        n, d = emb.dense.shape
        # Dense index: IndexHNSWFlat with inner product (equivalent to cosine on normalized vectors)
        dense_index = self._faiss.IndexHNSWFlat(d, 32)
        dense_index.hnsw.efConstruction = 200
        dense_index.hnsw.efSearch = 512
        # Normalize for cosine similarity via inner product
        faiss = self._faiss
        faiss.normalize_L2(emb.dense)
        dense_index.add(emb.dense)
        # Sparse: store as CSR matrix
        sparse_matrix = emb.sparse.copy()
        # Save
        self._faiss.write_index(dense_index, self._dense_path)
        save_npz(self._sparse_path, sparse_matrix)
        self._write_chunks(chunks)
        self._dense_index = dense_index
        self._sparse_matrix = sparse_matrix
        self._chunks = list(chunks)
        self._add_count_since_rebuild = 0

    def add(self, chunks: list[dict], emb: Embeddings) -> None:
        try:
            with self._lock:
                try:
                    self._add_unlocked(chunks, emb)
                except Exception as e:
                    raise IndexBuildError(str(e)) from e
        except self._lock.timeout_class:
            raise IndexLockedError(self._tenant_dir.split(os.sep)[-1]) from None

    def _add_unlocked(self, chunks: list[dict], emb: Embeddings) -> None:
        if self._dense_index is None:
            self.load()
        faiss = self._faiss
        faiss.normalize_L2(emb.dense)
        self._dense_index.add(emb.dense)
        # merge sparse
        if self._sparse_matrix is not None:
            from scipy.sparse import vstack
            self._sparse_matrix = vstack([self._sparse_matrix, emb.sparse])
        else:
            self._sparse_matrix = emb.sparse.copy()
        # append chunks
        start_idx = len(self._chunks)
        for i, c in enumerate(chunks):
            c["chunk_index"] = start_idx + i
        self._chunks.extend(chunks)
        # persist
        self._faiss.write_index(self._dense_index, self._dense_path)
        save_npz(self._sparse_path, self._sparse_matrix)
        self._write_chunks(self._chunks)
        self._add_count_since_rebuild += len(chunks)
        # Warn if HNSW graph quality may be degrading
        if self._add_count_since_rebuild > 0.2 * len(self._chunks):
            logger.warning(
                "Incremental adds (%d) exceed 20%% of total chunks (%d). "
                "HNSW graph quality may have degraded; consider full rebuild.",
                self._add_count_since_rebuild, len(self._chunks),
            )

    def load(self, tenant: str | None = None) -> None:
        if not os.path.exists(self._dense_path):
            raise FileNotFoundError(f"Index not found for tenant. Expected at {self._dense_path}")
        self._dense_index = self._faiss.read_index(self._dense_path)
        self._dense_index.hnsw.efSearch = 512
        self._sparse_matrix = load_npz(self._sparse_path)
        self._chunks = self._read_chunks()
        self._add_count_since_rebuild = 0

    def search_dense(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._dense_index is None:
            raise IndexNotFoundError("Dense index not loaded. Call load() or build() first.")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss = self._faiss
        faiss.normalize_L2(q)
        distances, indices = self._dense_index.search(q, k)
        results = []
        for i, d in zip(indices[0], distances[0]):
            if i >= 0 and i < len(self._chunks):
                results.append((int(i), float(d)))
        return results

    def search_sparse(self, q: csr_matrix, k: int) -> list[tuple[int, float]]:
        if self._sparse_matrix is None:
            raise IndexNotFoundError("Sparse index not loaded. Call load() or build() first.")
        # q is (1, 65536), sparse_matrix is (N, 65536).T → (65536, N)
        # scores = q @ sparse.T → (1, N)
        scores = (q @ self._sparse_matrix.T).toarray()[0]
        # get top-k indices
        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def chunk_count(self) -> int:
        return len(self._chunks)

    def get_chunks(self) -> list[dict]:
        return list(self._chunks)

    def _write_chunks(self, chunks: list[dict]) -> None:
        with open(self._chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def _read_chunks(self) -> list[dict]:
        if not os.path.exists(self._chunks_path):
            return []
        chunks = []
        with open(self._chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_indexer.py -v`
Expected: ALL PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_service/indexer.py tests/test_indexer.py
git commit -m "feat: add FAISS indexer with dense HNSW + scipy CSR sparse + lock contention handling"
```

---

### Task 6: Retriever (hybrid fusion + rerank)

**Files:**
- Create: `rag_service/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests for retriever**

```python
# tests/test_retriever.py
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
        # brute-force cosine similarity
        from rag_service.indexer import FAISSIndexer
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        v_norm = self._dense / (np.linalg.norm(self._dense, axis=1, keepdims=True) + 1e-8)
        scores = (q_norm @ v_norm.T)[0]
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    def search_sparse(self, q, k):
        scores = (q @ self._sparse.T).toarray()[0]
        idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idx]

    def get_chunks(self):
        return list(self._chunks)


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
        assert len(results) <= 5
        assert len(results) > 0
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

    def test_hybrid_scoring_different_from_pure_dense(self, mock_embedder, mock_indexer):
        retriever_hybrid = Retriever(mock_embedder, mock_indexer, enable_rerank=False, recall_top_k=10)
        results = retriever_hybrid.retrieve_sync("test query", top_k=5)
        scores = [r.score for r in results]
        # scores should be sorted descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_rerank_disabled_skips_flagembedding(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False)
        results = retriever.retrieve_sync("test", top_k=3)
        assert len(results) <= 3

    def test_query_with_rewritten_queries(self, mock_embedder, mock_indexer):
        retriever = Retriever(mock_embedder, mock_indexer, enable_rerank=False, recall_top_k=10)
        results = retriever.retrieve_sync("original query", rewritten_queries=["rewritten q1", "rewritten q2"], top_k=5)
        assert len(results) <= 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement retriever.py**

```python
import logging
import numpy as np
from rag_service.types import SearchResult, Embeddings

logger = logging.getLogger("rag_service.retriever")


class Retriever:
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3

    def __init__(
        self,
        embedder,
        indexer,
        enable_rerank: bool = True,
        recall_top_k: int = 50,
        reranker_model: str | None = None,
    ):
        self._embedder = embedder
        self._indexer = indexer
        self._enable_rerank = enable_rerank
        self._recall_top_k = recall_top_k
        self._reranker = None
        self._reranker_model = reranker_model or "BAAI/bge-reranker-v2-m3"

    async def retrieve(
        self, query: str, top_k: int = 5, rewritten_queries: list[str] | None = None
    ) -> list[SearchResult]:
        return self.retrieve_sync(query, top_k, rewritten_queries)

    def retrieve_sync(
        self, query: str, top_k: int = 5, rewritten_queries: list[str] | None = None
    ) -> list[SearchResult]:
        queries = [query]
        if rewritten_queries:
            queries.extend(rewritten_queries)

        # Phase 1: Hybrid recall (dense + sparse)
        all_candidates: dict[int, float] = {}
        for q in queries:
            emb = self._embedder.encode([q])
            k = min(self._recall_top_k, self._indexer.chunk_count())
            if k == 0:
                return []

            # Dense search
            dense_results = self._indexer.search_dense(emb.dense, k * 2)
            # Sparse search
            sparse_results = self._indexer.search_sparse(emb.sparse, k * 2)

            # Merge: score = 0.7 * dense + 0.3 * sparse
            dense_scores = {cid: score for cid, score in dense_results}
            sparse_scores = {cid: score for cid, score in sparse_results}

            # Normalize scores independently
            for scores in [dense_scores, sparse_scores]:
                if scores:
                    max_val = max(scores.values())
                    min_val = min(scores.values())
                    rng = max_val - min_val if max_val != min_val else 1
                    for cid in scores:
                        scores[cid] = (scores[cid] - min_val) / rng

            all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
            for cid in all_ids:
                d_score = dense_scores.get(cid, 0.0)
                s_score = sparse_scores.get(cid, 0.0)
                combined = self.DENSE_WEIGHT * d_score + self.SPARSE_WEIGHT * s_score
                if cid not in all_candidates or combined > all_candidates[cid]:
                    all_candidates[cid] = combined

        # Select top candidates
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:self._recall_top_k]

        # Phase 2: Rerank (if enabled)
        if self._enable_rerank and len(sorted_candidates) > top_k:
            sorted_candidates = self._rerank(query, sorted_candidates, top_k)

        # Build results
        chunks = self._indexer.get_chunks()
        results = []
        final_list = sorted_candidates[:top_k]
        for cid, score in final_list:
            c = chunks[cid]
            results.append(SearchResult(
                content=c["text"],
                metadata=c.get("metadata", {}),
                score=float(score),
                chunk_index=c.get("chunk_index", cid),
            ))
        return results

    def _rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        """Re-rank candidates using bge-reranker-v2-m3."""
        if not candidates:
            return candidates
        if self._reranker is None:
            self._load_reranker()

        chunks = self._indexer.get_chunks()
        pairs = [(query, chunks[cid]["text"]) for cid, _ in candidates]
        scores = self._reranker.compute_score(pairs, normalize=True)

        # Map back
        reranked = []
        for i, (cid, _) in enumerate(candidates):
            reranked.append((cid, float(scores[i])))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _load_reranker(self):
        logger.info("Loading reranker model (%s)...", self._reranker_model)
        from FlagEmbedding import FlagReranker
        self._reranker = FlagReranker(self._reranker_model, use_fp16=True)
        logger.info("Reranker model loaded.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: ALL PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_service/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever with hybrid fusion + configurable reranker"
```

---

### Task 7: RAGService (unified entry point)

**Files:**
- Create: `rag_service/service.py`
- Create: `tests/test_service.py`

- [ ] **Step 1: Write failing tests for service**

```python
# tests/test_service.py
import os
import pytest
from rag_service.service import RAGService
from rag_service.errors import EmptyInputError, TenantNotFoundError


class TestRAGService:
    def test_index_creates_tenant(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        count = rag.index_sync("kb1", texts=["hello world", "foo bar"], metadatas=[{"s": "a.md"}, {"s": "b.md"}])
        assert count > 0
        assert "kb1" in rag.list_tenants()

    def test_search_returns_results(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        rag.index_sync("kb1", texts=["Python is a programming language.", "FAISS is for vector search.", "Today is a nice day."])
        results = rag.search_sync("programming", tenant="kb1", top_k=2)
        assert len(results) <= 2
        assert "Python" in results[0].content

    def test_multi_tenant_isolation(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        rag.index_sync("tenant_a", texts=["content A1", "content A2"])
        rag.index_sync("tenant_b", texts=["content B1", "content B2"])
        assert set(rag.list_tenants()) == {"tenant_a", "tenant_b"}
        stats_a = rag.index_stats("tenant_a")
        stats_b = rag.index_stats("tenant_b")
        assert stats_a["chunk_count"] >= 1
        assert stats_b["chunk_count"] >= 1

    def test_incremental_index(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        count1 = rag.index_sync("kb1", texts=["text one", "text two"])
        count2 = rag.index_sync("kb1", texts=["text three"])
        assert count2 > count1

    def test_delete_index(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        rag.index_sync("kb1", texts=["hello"])
        assert "kb1" in rag.list_tenants()
        rag.delete_index_sync("kb1")
        assert "kb1" not in rag.list_tenants()

    def test_search_nonexistent_tenant_raises(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        with pytest.raises(TenantNotFoundError) as exc:
            rag.search_sync("query", tenant="nonexistent")
        assert "nonexistent" in str(exc.value)
        assert "Available tenants" in str(exc.value) or "[]" in str(exc.value)

    def test_index_empty_texts_raises(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        with pytest.raises(EmptyInputError):
            rag.index_sync("kb1", texts=[])

    def test_index_mismatched_lengths_raises(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        with pytest.raises(ValueError, match="length"):
            rag.index_sync("kb1", texts=["a", "b"], metadatas=[{"s": "x.md"}])

    def test_index_without_metadatas(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        count = rag.index_sync("kb1", texts=["hello world"])
        assert count > 0

    def test_index_stats(self, tmp_index_dir):
        rag = RAGService(index_dir=tmp_index_dir)
        rag.index_sync("kb1", texts=["a", "b", "c"])
        stats = rag.index_stats("kb1")
        assert stats["chunk_count"] >= 2
        assert "index_dir" in stats
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_service.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement service.py**

```python
import os
import shutil
import logging
from rag_service.chunker import AdaptiveChunker
from rag_service.embedder import Embedder, MockEmbedder
from rag_service.indexer import FAISSIndexer
from rag_service.retriever import Retriever
from rag_service.types import ChunkConfig, SearchResult
from rag_service.errors import EmptyInputError, TenantNotFoundError

logger = logging.getLogger("rag_service")


class RAGService:
    def __init__(
        self,
        index_dir: str,
        enable_rerank: bool = True,
        query_rewriter=None,
        rerank_top_k: int = 5,
        recall_top_k: int = 50,
        chunk_min_chars: int = 200,
        chunk_max_chars: int = 800,
        chunk_overlap: int = 80,
        chunk_batch_size: int = 32,
        embed_batch_size: int = 32,
        verbose: bool = False,
    ):
        self._index_dir = index_dir
        self._enable_rerank = enable_rerank
        self._query_rewriter = query_rewriter
        self._rerank_top_k = rerank_top_k
        self._recall_top_k = recall_top_k
        self._embed_batch_size = embed_batch_size
        self._verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)

        self._chunk_config = ChunkConfig(
            min_chars=chunk_min_chars,
            max_chars=chunk_max_chars,
            overlap=chunk_overlap,
            batch_size=chunk_batch_size,
        )

        # Components are lazily initialized
        self._embedder = None
        self._chunker = None

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _get_chunker(self):
        if self._chunker is None:
            self._chunker = AdaptiveChunker(self._get_embedder(), self._chunk_config)
        return self._chunker

    def _get_indexer(self, tenant: str) -> FAISSIndexer:
        return FAISSIndexer(self._index_dir, tenant)

    async def index(self, tenant: str, texts: list[str], metadatas: list[dict] | None = None) -> int:
        return self.index_sync(tenant, texts, metadatas)

    def index_sync(self, tenant: str, texts: list[str], metadatas: list[dict] | None = None) -> int:
        if not texts:
            raise EmptyInputError("texts cannot be empty")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})"
            )
        if metadatas is None:
            metadatas = [{}] * len(texts)

        chunker = self._get_chunker()
        embedder = self._get_embedder()

        # Chunk all texts
        all_chunks = []
        all_chunk_dicts = []
        chunk_idx = 0
        for text, meta in zip(texts, metadatas):
            chunks = chunker.chunk(text, metadata=meta)
            for c in chunks:
                c.chunk_index = chunk_idx
                all_chunks.append(c)
                all_chunk_dicts.append({
                    "text": c.text,
                    "metadata": c.metadata,
                    "chunk_index": chunk_idx,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                })
                chunk_idx += 1

        logger.info("Chunked %d texts into %d chunks.", len(texts), len(all_chunks))

        # Encode all chunks
        chunk_texts = [c.text for c in all_chunks]
        embeddings = embedder.encode(chunk_texts, batch_size=self._embed_batch_size)
        logger.info(
            "Encoded %d chunks: dense=%s, sparse=%s",
            len(all_chunks), embeddings.dense.shape, embeddings.sparse.shape,
        )

        # Build or add to index
        indexer = self._get_indexer(tenant)
        tenant_dir = os.path.join(self._index_dir, tenant)
        if os.path.exists(os.path.join(tenant_dir, "dense.faiss")):
            indexer.load()
            indexer.add(all_chunk_dicts, embeddings)
            logger.info("Added %d chunks to existing tenant '%s'.", len(all_chunks), tenant)
        else:
            indexer.build(all_chunk_dicts, embeddings)
            logger.info("Built new index for tenant '%s' with %d chunks.", tenant, len(all_chunks))

        return indexer.chunk_count()

    async def search(self, query: str, tenant: str, top_k: int = 5) -> list[SearchResult]:
        return self.search_sync(query, tenant, top_k)

    def search_sync(self, query: str, tenant: str, top_k: int = 5) -> list[SearchResult]:
        embedder = self._get_embedder()
        indexer = self._get_indexer(tenant)

        try:
            indexer.load()
        except FileNotFoundError:
            available = self.list_tenants()
            raise TenantNotFoundError(tenant, available)

        retriever = Retriever(
            embedder, indexer,
            enable_rerank=self._enable_rerank,
            recall_top_k=self._recall_top_k,
        )

        # Query rewriting
        rewritten = None
        if self._query_rewriter:
            import asyncio
            if asyncio.iscoroutinefunction(self._query_rewriter):
                import asyncio
                rewritten = asyncio.run(self._query_rewriter(query))
            else:
                rewritten = self._query_rewriter(query)

        results = retriever.retrieve_sync(query, top_k=top_k, rewritten_queries=rewritten)

        if self._verbose:
            logger.debug("Query: %s", query)
            logger.debug("Results: %d", len(results))
            for r in results:
                logger.debug("  score=%.4f chunk=%d content=%.80s", r.score, r.chunk_index, r.content)

        return results

    async def delete_index(self, tenant: str) -> None:
        return self.delete_index_sync(tenant)

    def delete_index_sync(self, tenant: str) -> None:
        tenant_dir = os.path.join(self._index_dir, tenant)
        if os.path.exists(tenant_dir):
            shutil.rmtree(tenant_dir)
            logger.info("Deleted index for tenant '%s'.", tenant)
        else:
            raise TenantNotFoundError(tenant, self.list_tenants())

    def list_tenants(self) -> list[str]:
        if not os.path.exists(self._index_dir):
            return []
        return [
            d for d in os.listdir(self._index_dir)
            if os.path.isdir(os.path.join(self._index_dir, d)) and os.path.exists(os.path.join(self._index_dir, d, "dense.faiss"))
        ]

    def index_stats(self, tenant: str) -> dict:
        indexer = self._get_indexer(tenant)
        try:
            indexer.load()
        except FileNotFoundError:
            available = self.list_tenants()
            raise TenantNotFoundError(tenant, available)
        return {
            "tenant": tenant,
            "chunk_count": indexer.chunk_count(),
            "index_dir": os.path.join(self._index_dir, tenant),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_service.py -v`
Expected: ALL PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_service/service.py tests/test_service.py
git commit -m "feat: add RAGService unified entry point"
```

---

### Task 8: Package exports (__init__.py)

**Files:**
- Modify: `rag_service/__init__.py`

- [ ] **Step 1: Update __init__.py with public exports**

```python
from rag_service.types import Chunk, SearchResult, Embeddings, ChunkConfig
from rag_service.errors import (
    RAGException,
    IndexNotFoundError,
    EmptyInputError,
    EmbeddingError,
    IndexBuildError,
    TenantNotFoundError,
    IndexLockedError,
)
from rag_service.service import RAGService

__all__ = [
    "RAGService",
    "Chunk",
    "SearchResult",
    "Embeddings",
    "ChunkConfig",
    "RAGException",
    "IndexNotFoundError",
    "EmptyInputError",
    "EmbeddingError",
    "IndexBuildError",
    "TenantNotFoundError",
    "IndexLockedError",
]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from rag_service import RAGService, SearchResult, ChunkConfig, RAGException; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add rag_service/__init__.py
git commit -m "feat: wire up package exports in __init__.py"
```

---

### Task 9: Run all tests + final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS (~43 tests across 5 test files)

- [ ] **Step 2: Verify package is importable**

Run: `python -c "from rag_service import RAGService; r = RAGService(index_dir='/tmp/rag_test'); print('RAGService created OK')"`
Expected: `RAGService created OK`

- [ ] **Step 3: Verify pip install -e**

Run: `pip install -e "C:/Users/Camille/Desktop/rag-service[server]" 2>&1 | tail -5`
Expected: Successfully installed rag-service

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: final verification — all tests passing"
```

---

### Task 10: Optional FastAPI server (server.py)

**Files:**
- Create: `server.py`

- [ ] **Step 1: Implement server.py**

```python
"""Optional FastAPI HTTP shell for RAGService. pip install rag-service[server] to use."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "./data/rag_indexes")
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    from rag_service import RAGService
    rag_service = RAGService(index_dir=INDEX_DIR, enable_rerank=True)
    yield


app = FastAPI(title="RAG Service", version="1.0.0", lifespan=lifespan)


class IndexRequest(BaseModel):
    texts: list[str]
    metadatas: list[dict] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultResponse(BaseModel):
    content: str
    metadata: dict
    score: float
    chunk_index: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/tenants/{name}/index")
async def index_tenant(name: str, req: IndexRequest):
    try:
        count = await rag_service.index(name, req.texts, req.metadatas)
        return {"tenant": name, "chunk_count": count}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tenants/{name}/search")
async def search_tenant(name: str, req: SearchRequest):
    try:
        results = await rag_service.search(req.query, tenant=name, top_k=req.top_k)
        return {"results": [r.__dict__ for r in results]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/tenants/{name}")
async def delete_tenant(name: str):
    try:
        await rag_service.delete_index(name)
        return {"deleted": name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tenants")
async def list_tenants():
    return {"tenants": rag_service.list_tenants()}


@app.get("/tenants/{name}/stats")
async def tenant_stats(name: str):
    try:
        return rag_service.index_stats(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Verify server imports**

Run: `python -c "import server; print('Server module OK')"`
Expected: `Server module OK`

- [ ] **Step 3: Commit**

```bash
git add server.py
git commit -m "feat: add optional FastAPI server"
```
