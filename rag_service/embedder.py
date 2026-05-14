import logging
import os
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
        self._device = device or os.environ.get("RAG_DEVICE") or self._detect_device()
        self._base_model = None
        self._tokenizer = None
        self._sparse_linear = None
        self._sparse_dim = None

    @staticmethod
    def _detect_device() -> str:
        """Probe CUDA in a subprocess with timeout to avoid hanging the main process."""
        import subprocess, sys, json, tempfile

        logger.info("  ⚡ 检测 GPU 状态...")
        probe_code = """\
import json, sys
try:
    import torch
    ok = torch.cuda.is_available()
    if ok:
        torch.cuda.synchronize()
    sys.stdout.write(json.dumps({"ok": ok}))
except Exception:
    sys.stdout.write(json.dumps({"ok": False}))
"""
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(probe_code)
                tmp = f.name
            proc = subprocess.run(
                [sys.executable, tmp],
                capture_output=True, text=True, timeout=15,
            )
            result = json.loads(proc.stdout)
            device = "cuda" if result.get("ok") else "cpu"
            logger.info("  ✓ GPU 探针结果: %s", device)
            return device
        except Exception:
            logger.info("  ⚠ GPU 探针超时或失败，降级到 CPU")
            return "cpu"
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _load_model(self):
        if self._base_model is not None:
            return
        import torch.nn as nn
        from transformers import XLMRobertaModel, AutoTokenizer

        model_name = self._model_path or "BAAI/bge-m3"

        logger.info("  📦 加载 bge-m3 模型 (~2.2GB) 到设备 '%s'，请稍候...", self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._base_model = XLMRobertaModel.from_pretrained(model_name)
        self._base_model = self._base_model.to(self._device)
        self._base_model.eval()

        # Load sparse_linear head from checkpoint
        try:
            import torch as _torch
            from huggingface_hub import snapshot_download
            import os as _os

            hidden_size = self._base_model.config.hidden_size  # 1024
            vocab_size = self._base_model.config.vocab_size   # 250002
            self._sparse_dim = vocab_size

            # bge-m3 sparse_linear: projects [1024] → [1] score per token,
            # then scores are placed at token-ID positions in the vocab vector
            self._sparse_linear = nn.Linear(hidden_size, 1)

            # Locate the sparse_linear.pt file in HF cache
            cache_base = _os.environ.get('TRANSFORMERS_CACHE') or _os.environ.get('HF_HOME', '')
            model_cache = _os.path.join(cache_base, 'models--BAAI--bge-m3')
            if not _os.path.exists(model_cache):
                # Fall back to snapshot_download to find it
                model_dir = snapshot_download(
                    repo_id=model_name,
                    ignore_patterns=['*.DS_Store', 'imgs/*'],
                )
                model_cache = _os.path.dirname(_os.path.dirname(model_dir))

            loaded = False
            snaps = _os.path.join(model_cache, 'snapshots')
            for d in _os.listdir(snaps) if _os.path.isdir(snaps) else []:
                sl_path = _os.path.join(snaps, d, 'sparse_linear.pt')
                if _os.path.exists(sl_path):
                    sd = _torch.load(sl_path, map_location='cpu', weights_only=True)
                    self._sparse_linear.weight.data = sd['weight'].float()  # [1, 1024]
                    if 'bias' in sd:
                        self._sparse_linear.bias.data = sd['bias'].float()  # [1]
                    loaded = True
                    logger.info("  ✅ 找到 sparse_linear.pt: %s", sl_path)
                    break

            if loaded:
                self._sparse_linear = self._sparse_linear.to(self._device)
                self._sparse_linear.eval()
                logger.info("  ✅ 稀疏线性层加载完成（vocab_size=%d）", self._sparse_dim)
            else:
                raise RuntimeError("sparse_linear.pt not found in any snapshot")
        except Exception as e:
            logger.warning("  ⚠ 稀疏线性层加载失败，降级为仅稠密检索: %s", e)
            self._sparse_linear = None
            self._sparse_dim = 65536  # fallback for empty matrices

        logger.info("  ✅ 模型加载完成（设备: %s）", self._device)

    def encode(self, texts: list[str], batch_size: int = 32) -> Embeddings:
        if not texts:
            logger.info("  ⚠ 编码请求为空，返回空向量")
            return Embeddings(
                dense=np.empty((0, self.dim), dtype=np.float32),
                sparse=csr_matrix((0, self._sparse_dim or 65536), dtype=np.float32),
            )
        self._load_model()

        import torch

        all_dense = []
        all_sparse_dicts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                return_tensors='pt', max_length=8192,
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._base_model(**encoded)

            hidden = outputs.last_hidden_state  # (batch, seq_len, 1024)

            # Dense: CLS token → L2 normalize
            dense_vec = hidden[:, 0]
            dense_vec = dense_vec / dense_vec.norm(dim=1, keepdim=True).clamp(min=1e-12)
            all_dense.append(dense_vec.cpu().numpy())

            # Sparse: linear([1024]→[1]) → ReLU → scatter-by-token-id → max-pool
            if self._sparse_linear is not None:
                sparse_scores = torch.relu(self._sparse_linear(hidden).squeeze(-1))  # (batch, seq_len)
                input_ids = encoded['input_ids']  # (batch, seq_len)
                for b in range(sparse_scores.shape[0]):
                    row_scores = sparse_scores[b]
                    row_ids = input_ids[b]
                    mask = row_scores > 0
                    if mask.any():
                        tok_ids = row_ids[mask].tolist()
                        weights = row_scores[mask].tolist()
                        # Max-pool over duplicate token IDs
                        vec = {}
                        for tid, w in zip(tok_ids, weights):
                            if w > vec.get(tid, 0):
                                vec[tid] = w
                        all_sparse_dicts.append(vec)
                    else:
                        all_sparse_dicts.append({})

        dense = np.vstack(all_dense) if all_dense else np.empty((0, self.dim), dtype=np.float32)
        sparse_dim = self._sparse_dim or 65536

        if all_sparse_dicts:
            sparse = self._sparse_to_csr(all_sparse_dicts, sparse_dim)
        else:
            sparse = csr_matrix((len(texts), sparse_dim), dtype=np.float32)

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
