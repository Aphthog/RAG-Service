# RAG Service

Reusable Python RAG (Retrieval-Augmented Generation) library with hybrid dense+sparse retrieval, adaptive chunking, and multi-tenant isolation.

## Features

- **Hybrid retrieval** — bge-m3 model: 1024-dim dense (FAISS HNSW) + 65536-dim sparse (scipy CSR), fused 0.7/0.3
- **Adaptive chunking** — Markdown heading-aware + semantic similarity boundary detection
- **Re-ranking** — bge-reranker-v2-m3 for result refinement
- **Multi-tenant** — Per-tenant index isolation with filelock-based concurrency control
- **Optional HTTP API** — FastAPI server included

## Installation

```bash
pip install git+https://github.com/Aphthog/rag-service.git

# With HTTP server support:
pip install "rag-service[server] @ git+https://github.com/Aphthog/rag-service.git"
```

## Quick Start

```python
from rag_service import RAGService

rag = RAGService(index_dir="./data/rag_indexes")

# Index documents (auto-chunked)
rag.index_sync(
    tenant="my_kb",
    texts=["Python is a programming language...", "FAISS is a vector search library..."],
    metadatas=[{"source": "wiki1.md"}, {"source": "wiki2.md"}],
)

# Search
results = rag.search_sync("vector search", tenant="my_kb", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.content}")
```

## HTTP Server

```bash
# Start server
python server.py

# Index
curl -X POST http://localhost:8000/tenants/my_kb/index \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "foo bar"]}'

# Search
curl -X POST http://localhost:8000/tenants/my_kb/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "top_k": 3}'

# List tenants
curl http://localhost:8000/tenants

# Health check
curl http://localhost:8000/health
```

## Model Download

Models download automatically on first use from HuggingFace. Set these env vars to use a custom cache path:

```bash
set HF_HOME=E:/ai_models/huggingface
set TRANSFORMERS_CACHE=E:/ai_models/transformers
set SENTENCE_TRANSFORMERS_HOME=E:/ai_models/sentence-transformers
```

## API

### RAGService

| Method | Description |
|--------|-------------|
| `index_sync(tenant, texts, metadatas?)` | Index documents (auto-chunk + encode + build) |
| `search_sync(query, tenant, top_k=5)` | Search across indexed chunks |
| `delete_index_sync(tenant)` | Remove a tenant's index |
| `list_tenants()` | List all indexed tenants |
| `index_stats(tenant)` | Get chunk count and index location |

### Configuration

```python
rag = RAGService(
    index_dir="./data",
    enable_rerank=True,       # Use bge-reranker for re-ranking
    chunk_max_chars=800,      # Max characters per chunk
    chunk_overlap=80,         # Overlap between chunks
    recall_top_k=50,          # Candidates from hybrid retrieval
    embedder=my_embedder,     # Inject custom embedder (for testing)
)
```
