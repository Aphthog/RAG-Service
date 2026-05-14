import os
import json
import logging
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from filelock import FileLock, Timeout as FileLockTimeout
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
        except FileLockTimeout:
            raise IndexLockedError(self._tenant_dir.split(os.sep)[-1]) from None

    def _build_unlocked(self, chunks: list[dict], emb: Embeddings) -> None:
        os.makedirs(self._tenant_dir, exist_ok=True)
        n, d = emb.dense.shape
        logger.info("    - 构建 HNSW 索引（维度=%d, 连接数=32, efConstruction=200）", d)
        dense_index = self._faiss.IndexHNSWFlat(d, 32)
        dense_index.hnsw.efConstruction = 200
        dense_index.hnsw.efSearch = 512
        faiss = self._faiss
        faiss.normalize_L2(emb.dense)
        dense_index.add(emb.dense)
        sparse_matrix = emb.sparse.copy()
        self._faiss.write_index(dense_index, self._dense_path)
        save_npz(self._sparse_path, sparse_matrix)
        self._write_chunks(chunks)
        logger.info("    - 索引文件写入完成: %s, %s, %s",
                     self._dense_path, self._sparse_path, self._chunks_path)
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
        except FileLockTimeout:
            raise IndexLockedError(self._tenant_dir.split(os.sep)[-1]) from None

    def _add_unlocked(self, chunks: list[dict], emb: Embeddings) -> None:
        if self._dense_index is None:
            self.load()
        faiss = self._faiss
        faiss.normalize_L2(emb.dense)
        self._dense_index.add(emb.dense)
        if self._sparse_matrix is not None:
            from scipy.sparse import vstack
            self._sparse_matrix = vstack([self._sparse_matrix, emb.sparse])
        else:
            self._sparse_matrix = emb.sparse.copy()
        start_idx = len(self._chunks)
        for i, c in enumerate(chunks):
            c["chunk_index"] = start_idx + i
        self._chunks.extend(chunks)
        self._faiss.write_index(self._dense_index, self._dense_path)
        save_npz(self._sparse_path, self._sparse_matrix)
        self._write_chunks(self._chunks)
        self._add_count_since_rebuild += len(chunks)
        if self._add_count_since_rebuild > 0.2 * len(self._chunks):
            logger.warning(
                "Incremental adds (%d) exceed 20%% of total chunks (%d). "
                "HNSW graph quality may have degraded; consider full rebuild.",
                self._add_count_since_rebuild, len(self._chunks),
            )

    def load(self, tenant: str | None = None) -> None:
        if not os.path.exists(self._dense_path):
            raise FileNotFoundError(f"Index not found for tenant. Expected at {self._dense_path}")
        logger.info("  ▶ 加载索引: %s", self._tenant_dir)
        self._dense_index = self._faiss.read_index(self._dense_path)
        self._dense_index.hnsw.efSearch = 512
        self._sparse_matrix = load_npz(self._sparse_path)
        self._chunks = self._read_chunks()
        self._add_count_since_rebuild = 0
        logger.info("  ✓ 索引加载完成: %d 个块（稠密 + 稀疏）", len(self._chunks))

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
                results.append((int(i), -float(d)))
        return results

    def search_sparse(self, q: csr_matrix, k: int) -> list[tuple[int, float]]:
        if self._sparse_matrix is None:
            raise IndexNotFoundError("Sparse index not loaded. Call load() or build() first.")
        scores = (q @ self._sparse_matrix.T).toarray()[0]
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
