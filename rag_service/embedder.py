import logging
import numpy as np
from scipy.sparse import csr_matrix
from rag_service.types import Embeddings

# Suppress noisy httpx/huggingface hub logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger("rag_service.embedder")


class Embedder:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        self._model_path = model_path
        self._device = device or self._detect_device()
        self._model = None

    @staticmethod
    def _detect_device() -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

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
        logger.info("Encoding %d texts (batch_size=%d, dim=%d)...", len(texts), batch_size, self.dim)
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            output_value=None,
            show_progress_bar=True,
        )

        # Helper: convert potentially-CUDA tensor to numpy
        def _to_numpy(v):
            if hasattr(v, 'cpu'):
                return v.cpu().numpy()
            return np.asarray(v, dtype=np.float32)

        # SentenceTransformer 3.x returns list[dict] with 'sentence_embedding' key
        if isinstance(output, list) and output and isinstance(output[0], dict):
            dense = np.array([_to_numpy(o["sentence_embedding"]) for o in output], dtype=np.float32)
            sparse = csr_matrix((len(texts), 65536), dtype=np.float32)
        elif hasattr(output, 'dense_vecs') and hasattr(output, 'sparse_vecs'):
            dense = _to_numpy(output.dense_vecs)
            sparse_list = output.sparse_vecs
            if sparse_list:
                sparse = self._sparse_to_csr(sparse_list, 65536)
            else:
                sparse = csr_matrix((len(texts), 65536), dtype=np.float32)
        else:
            dense = _to_numpy(output)
            sparse = csr_matrix((len(texts), 65536), dtype=np.float32)
        return Embeddings(dense=dense, sparse=sparse)

    def _sparse_to_csr(self, sparse_dicts: list[dict], vocab_size: int) -> csr_matrix:
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
        # Derive seed offset from text hash for input-dependent determinism
        text_hash = abs(hash("".join(texts))) if texts else 0
        rng = np.random.default_rng(self._seed + text_hash)
        dense = rng.random((n, self._dim), dtype=np.float32)
        data = []
        indices = []
        indptr = [0]
        for i in range(n):
            nz = rng.integers(3, 15)
            idx = rng.integers(0, self._sparse_dim, nz)
            vals = rng.random(nz, dtype=np.float32)
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
