# RAG Service 设计文档

> 独立、可复用的 RAG (Retrieval-Augmented Generation) Python 库。提供文本分块、向量化、索引、混合检索、重排序的完整链路，SDK + 可选 HTTP 服务两种使用方式。

**创建日期:** 2026-05-12

**更新日期:** 2026-05-12（autoplan 审查修订）

**目标:** 替换 qq-bot 中效果不佳的 RAG 模块，同时作为通用 RAG 组件供其他项目通过 `pip install git+...` 使用。

---

## 一、整体架构

```
rag-service/                  # C:/Users/Camille/Desktop/rag-service/
├── rag_service/              # 包名 rag_service（避免与 PyPI rag 冲突）
│   ├── __init__.py           # 导出 RAGService, SearchResult, RAGException
│   ├── types.py              # 数据类型: Chunk, SearchResult, Embeddings, ChunkConfig
│   ├── chunker.py            # 自适应分块（Markdown 结构 + 语义 fallback，批处理编码）
│   ├── embedder.py           # bge-m3 dense + sparse 双向量
│   ├── indexer.py            # FAISS 索引（dense HNSW） + scipy CSR 稀疏检索
│   ├── retriever.py          # 混合召回 + rerank
│   ├── service.py            # 统一入口 RAGService
│   └── errors.py             # 异常类型
├── server.py                 # 可选 FastAPI 壳（≤200 行）
├── pyproject.toml
└── tests/
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_indexer.py
    ├── test_retriever.py
    └── test_service.py
```

**组件依赖关系：**

```
RAGService
    ├──→ Chunker ──→ 复用 Embedder 实例（语义分块，批处理编码）
    ├──→ Embedder ──→ sentence-transformers + bge-m3
    ├──→ Indexer ──→ FAISS(dense) + scipy.sparse.csr_matrix(sparse)
    └──→ Retriever ──→ Embedder + Indexer + FlagEmbedding(reranker)
```

**调用方视角：**

```python
from rag_service import RAGService

rag = RAGService(index_dir="/data/rag_indexes")

# 索引
await rag.index("my_kb", texts=["文本1", "文本2"],
                metadatas=[{"source": "doc1.md"}, {"source": "doc2.md"}])

# 检索
results = await rag.search("怎么修复登录bug", tenant="my_kb", top_k=5)
# → [SearchResult(content="...", metadata={"source": "doc1.md"}, score=0.93, chunk_index=3), ...]
```

**多租户隔离：** 每个租户一个子目录 `{index_dir}/{tenant}/`，存独立的 FAISS 索引和 chunk 元数据。写操作受 per-tenant `threading.Lock` 保护（FAISS 非线程安全，并发写会损坏索引）。

**模型存储路径：** 默认使用 HuggingFace 标准缓存路径（`~/.cache/huggingface/`），通过 `HF_HOME`、`TRANSFORMERS_CACHE`、`SENTENCE_TRANSFORMERS_HOME` 环境变量可覆盖到自定义路径（如 `E:/ai_models/`）。代码不硬编码盘符，跨平台可用。

---

## 二、全流程

### 2.1 离线索引

```
PDF/Word/HTML/MD/纯文本          ← 调用方用对应工具转文本
        │
        ▼
  (texts, metadatas)              ← 调用方传入 RAGService.index()
        │                         ← 入口校验: len(texts) == len(metadatas)
        ▼
  AdaptiveChunker.chunk(text)     ← 按 Markdown 结构或语义边界切分
        │                           语义切分使用批处理编码（batch_size=32）
        ▼
  List[Chunk]                     ← {text, metadata, chunk_index, start_char, end_char}
        │                           metadata 从父文档继承，一文多块共享
        ▼
  Embedder.encode(texts)          ← bge-m3 输出（batch_size 可配置）
  ├── dense: (N, 1024)
  └── sparse: (N, 65536) scipy CSR 稀疏格式
        │
        ▼
  Indexer.build()                 ← 双索引并行建
  ├── dense_index (IndexHNSWFlat, M=32, efConstruction=200, efSearch=512)
  └── sparse: scipy.sparse.csr_matrix + 内积搜索（C 底层矩阵乘法）
        │
        ▼
  持久化到 {index_dir}/{tenant}/
  ├── dense.faiss
  ├── sparse.npz                  ← scipy sparse 格式，极小体积
  └── chunks.jsonl                ← 每行: {text, metadata, chunk_index, start_char, end_char}
```

### 2.2 增量追加

```
新文本 → AdaptiveChunker → Embedder.encode → Indexer.add() → 自动写盘
```

FAISS HNSW `add()` 支持增量追加，但图质量会随增量次数逐步下降。Indexer 内部追踪 `add_count_since_rebuild`，当增量超过当前 chunk 总量的 20% 时发出 `logging.warning` 建议全量重建（`build()`）。scipy CSR 稀疏矩阵重建是 O(N) 操作，增量开销可忽略。

### 2.3 在线检索

```
query: "那个登录bug怎么修的"
        │
        ▼
  [查询改写]（可选，调用方注入 async callback）
  ├── LLM → 2~3 个改写 query
  └── 未注入则直接用原始 query
        │
        ▼
  Embedder.encode(query) → dense_q + sparse_q（CSR 格式）
        │
        ▼
  混合召回（并行）
  ├── dense_q × dense_index (FAISS HNSW)  → top-K×2 (K=50)
  └── sparse_q × sparse_index (scipy CSR) → top-K×2
        │
        ▼
  混合打分 + 去重合并
  score = 0.7 × dense_cosine_sim + 0.3 × sparse_inner_product
  去重后取 top-K (K=50)
        │
        ▼
  [Reranker 精排]（默认开，可配置关闭）
  BAAI/bge-reranker-v2-m3 逐条打分
  50 → top-N (N=5, 可配置)
        │
        ▼
  List[SearchResult]
  [{content, metadata, score, chunk_index}, ...]
```

---

## 三、组件设计

### 3.1 数据类型 (`types.py`)

```python
@dataclass
class Chunk:
    text: str
    metadata: dict           # 从父文档继承，一文多块共享同一 metadata
    chunk_index: int
    start_char: int
    end_char: int

@dataclass
class SearchResult:
    content: str
    metadata: dict           # 调用方传入的元数据（source 等）
    score: float             # 0~1，越高越相关
    chunk_index: int

@dataclass
class Embeddings:
    dense: np.ndarray        # (N, 1024)
    sparse: csr_matrix       # (N, 65536) scipy CSR 稀疏格式

@dataclass
class ChunkConfig:
    min_chars: int = 200
    max_chars: int = 800
    overlap: int = 80
    semantic_threshold: float = 0.6   # 语义切分相似度断刀阈值
    batch_size: int = 32              # 语义切分编码批大小
```

### 3.2 分块器 (`chunker.py`)

**AdaptiveChunker** — 自适应分块入口：

```
输入文本
    │
    ▼
检测是否含 Markdown 结构？（标题/代码块/表格）
    │
    ├── 是 → MarkdownChunker
    │       按 # 标题切段 → 按空行切块 → 检查每块长度
    │       超过 max_chars(800) 的子块 → SemanticChunker 再次切分
    │
    └── 否 → SemanticChunker
              逐句打 embedding（批处理 batch_size=32）
              → 计算相邻句余弦相似度
              → 相似度 < semantic_threshold(0.6) 处断刀
              → 合并句子直到 chunk 长度在 [200, 800] 之间
              重叠 80 字符
```

批处理优化：所有句子打包一次 `model.encode(sentences, batch_size=32)`，避免逐句调用的额外开销。50 句文档编码耗时从 ~2s 降至 ~80ms。

### 3.3 Embedder (`embedder.py`)

封装 bge-m3 模型（`BAAI/bge-m3`）：

- **模型路径:** 默认 `None`，走 HuggingFace 标准缓存（`~/.cache/huggingface/hub/`），通过 `model_path` 参数或 `HF_HOME` 环境变量覆盖
- **首次加载:** 打印 `"Loading bge-m3 model (~2.2GB), this may take a few minutes..."`
- **单例加载:** 首次 `encode()` 调用时初始化，进程内唯一实例
- **Chunker 共用:** `AdaptiveChunker` 的语义分块复用同一个 Embedder 实例
- **输出:** `Embeddings(dense=ndarray, sparse=csr_matrix)` — dense 为 FAISS 用，sparse 为 scipy CSR 稀疏检索用
- **设备:** 自动检测 GPU，否则 CPU
- **批处理:** `encode(texts, batch_size=32)` 控制编码批大小

```python
class Embedder:
    def __init__(self, model_path: str | None = None, device: str = "auto"):
        ...

    def encode(self, texts: list[str], batch_size: int = 32) -> Embeddings:
        """批量编码，返回 dense + sparse 向量"""
        ...

    @property
    def dim(self) -> int:
        return 1024  # dense 维度
```

### 3.4 Indexer (`indexer.py`)

双索引，dense 用 FAISS HNSW，sparse 用 scipy CSR 矩阵：

```
{index_dir}/{tenant}/
├── dense.faiss     # IndexHNSWFlat, M=32, efConstruction=200, efSearch=512
├── sparse.npz      # scipy.sparse.csr_matrix, save_npz/load_npz
├── chunks.jsonl    # 每行一个 Chunk + metadata
└── .lock           # 写锁（filelock）
```

**并发安全：** 所有写操作（`build`/`add`）持有 per-tenant `filelock.FileLock`。读操作（`search_dense`/`search_sparse`）不加锁，与写并发时 FAISS 可能读到中间状态——如果调用方需要严格读写隔离，应在外部串行化。

**增量阈值：** `add()` 追踪增量次数，超过当前 chunk 量 20% 时 `logging.warning("HNSW graph quality may have degraded; consider full rebuild.")`

**核心方法：**

```python
class FAISSIndexer:
    def build(self, chunks: list[dict], emb: Embeddings) -> None: ...
    def add(self, chunks: list[dict], emb: Embeddings) -> None: ...
    def load(self, tenant: str) -> None: ...
    def search_dense(self, q: np.ndarray, k: int) -> list[tuple[int, float]]: ...
    def search_sparse(self, q: csr_matrix, k: int) -> list[tuple[int, float]]: ...
    def chunk_count(self) -> int: ...
```

`build()` 覆盖已有索引；`add()` 增量追加。两者都自动持久化。Sparse 检索通过 `csr_matrix @ sparse_query.T` 矩阵乘法实现，底层 C 实现，无需自定义倒排索引。

### 3.5 Retriever (`retriever.py`)

**混合召回：**

```
dense_scores = cosine_similarity(dense_q, dense_index_results)
sparse_scores = inner_product(sparse_q, sparse_index_results)  # CSR 矩阵乘法
combined = 0.7 * dense_scores + 0.3 * sparse_scores
去重 → top-K (默认 50)
```

**Rerank 精排（默认开启，可配置关闭）：**

- 模型: `BAAI/bge-reranker-v2-m3`，默认存储到 HuggingFace 缓存
- 框架: FlagEmbedding
- 候选 50 条 → 重排取 top-N (默认 5)

```python
class Retriever:
    def __init__(self, embedder, indexer, enable_rerank: bool, ...): ...

    async def retrieve(
        self, query: str, top_k: int = 5, rewritten_queries: list[str] | None = None
    ) -> list[SearchResult]: ...
```

### 3.6 RAGService (`service.py`)

统一入口，组合以上所有组件：

```python
class RAGService:
    def __init__(
        self,
        index_dir: str,
        enable_rerank: bool = True,
        query_rewriter: Callable[[str], Awaitable[list[str]]] | None = None,
        rerank_top_k: int = 5,
        recall_top_k: int = 50,
        chunk_min_chars: int = 200,
        chunk_max_chars: int = 800,
        chunk_overlap: int = 80,
        chunk_batch_size: int = 32,
        embed_batch_size: int = 32,
        verbose: bool = False,
    ): ...

    async def index(self, tenant: str, texts: list[str],
                    metadatas: list[dict] | None = None) -> int:
        """索引文本，返回 chunk 数量。
        若 metadatas 为 None，所有 chunk metadata 为空 dict。
        若 len(metadatas) != len(texts)，抛出 ValueError。
        若 tenant 索引已存在，增量追加。
        """
        ...

    async def search(self, query: str, tenant: str,
                     top_k: int = 5) -> list[SearchResult]:
        """检索。若配置了 query_rewriter 则先改写再检索。"""
        ...

    async def delete_index(self, tenant: str) -> None: ...
    def list_tenants(self) -> list[str]: ...
    def index_stats(self, tenant: str) -> dict: ...
```

**日志：** `verbose=True` 时输出结构化日志：chunk 切分边界、embedding shapes、dense/sparse 各阶段 top-K 得分、reranker 输入输出得分对比。默认使用 `logging.getLogger("rag_service")`，级别 INFO。

### 3.7 Server (`server.py`)

可选 FastAPI 壳，≤200 行。不修改核心逻辑，仅做 HTTP ↔ RAGService 转换。

```
POST   /tenants/{name}/index      # body: {texts: [...], metadatas: [...]}
POST   /tenants/{name}/search     # body: {query: "...", top_k: 5}
DELETE /tenants/{name}            # 删除租户
GET    /tenants                   # 列出所有租户
GET    /tenants/{name}/stats      # 索引统计
GET    /health                    # {"status": "ok"}
```

HTTP 模式下 `query_rewriter` 不可用（需要调用方注入 Python callable），其余功能一致。调用方可在发起 HTTP 请求前自行改写查询。

### 3.8 异常 (`errors.py`)

```python
class RAGException(Exception): ...
class IndexNotFoundError(RAGException): ...
class EmptyInputError(RAGException): ...
class EmbeddingError(RAGException): ...
class IndexBuildError(RAGException): ...
class TenantNotFoundError(RAGException):
    """消息格式: "Tenant 'xxx' not found. Available tenants: ['a', 'b']" """
...
class IndexLockedError(RAGException): ...
```

---

## 四、配置与依赖

### 4.1 依赖清单 (`pyproject.toml`)

```toml
[project]
name = "rag-service"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",              # CSR 稀疏矩阵检索
    "faiss-cpu>=1.8",
    "sentence-transformers>=3.0",
    "FlagEmbedding>=1.3",       # bge-reranker-v2-m3
    "filelock>=3.13",           # per-tenant 索引锁
    "fastapi>=0.110",           # server.py 需要
    "uvicorn[standard]>=0.30",  # server.py 需要
]

[project.optional-dependencies]
server = ["fastapi", "uvicorn"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### 4.2 模型文件

| 模型 | 用途 | 默认存储路径 | 大小 |
|------|------|-------------|------|
| BAAI/bge-m3 | Embedding | HF 缓存（`~/.cache/huggingface/hub/`） | ~2.2GB |
| BAAI/bge-reranker-v2-m3 | Rerank | HF 缓存（`~/.cache/huggingface/hub/`） | ~1.5GB |

**环境变量（可选，覆盖默认缓存路径）：**

```bash
HF_HOME=E:/ai_models/huggingface
TRANSFORMERS_CACHE=E:/ai_models/transformers
SENTENCE_TRANSFORMERS_HOME=E:/ai_models/sentence-transformers
```

不设这些变量时，模型自动下载到对应平台的 HuggingFace 默认缓存目录，跨平台工作。

### 4.3 安装方式

```bash
# 开发安装
cd C:/Users/Camille/Desktop/rag-service
pip install -e ".[server]"

# 其他项目使用（Git 直装）
pip install git+https://github.com/Aphthog/rag-service.git

# 后续可选：发布 PyPI 后
pip install rag-service
```

---

## 五、测试策略

| 层级 | 文件名 | 测试内容 |
|------|--------|---------|
| 单元 | test_chunker.py | MarkdownChunker 结构切分、SemanticChunker 相似度边界（mock Embedder 返回固定向量）、空文本/单句/超长句、overlap 正确性、批处理正确性 |
| 单元 | test_embedder.py | 输出 Embeddings shape、dense dim=1024、sparse CSR shape=(N,65536)、sparse 非零项 > 0、空列表返回空数组（CI 用 MockEmbedder） |
| 组件 | test_indexer.py | build 创建索引、load 恢复索引、add 增量追加、chunk_count 正确、不存在的路径抛 FileNotFoundError、per-tenant 锁互斥、sparse CSR 持久化/加载 |
| 组件 | test_retriever.py | 混合打分公式验证（MockEmbedder + MockIndexer）、去重逻辑、rerank 开启/关闭、空库返回空列表 |
| 集成 | test_service.py | 完整流程: index → search → add → search、多租户隔离、delete_index、index_stats、texts/metadatas 长度不匹配抛 ValueError、EmptyInputError |

**Mock 策略:**
- **MockEmbedder:** 返回固定维度 random 向量（dense + sparse CSR），所有单元测试默认使用，**不需要下载真实模型**
- **集成测试:** 使用 MockEmbedder + 临时目录，`pytest` 默认零模型下载
- **真实模型烟雾测试:** 通过 `pytest --real-models` marker 选跑，CI 默认跳过

**不做:** 真实 HTTP 接口测试（Server 层用 FastAPI TestClient 在手动验证阶段测试）。

---

## 六、与 qq-bot 的集成路径

1. 完成 `rag-service` 核心开发并通过全部测试
2. qq-bot 中添加依赖: `pip install git+https://github.com/Aphthog/rag-service.git`
3. 修改 `qq_bot/agent/tools.py` 中 `_handle_search_knowledge` 和 `_handle_add_to_knowledge`，将 `from qq_bot.rag.retriever import Retriever` 替换为 `from rag_service import RAGService`
4. 移除 `qq_bot/rag/` 目录及 `qq_bot.skills.memory` 中的旧 RAG 依赖
5. qq-bot `.env` 中新增 `RAG_INDEX_DIR`、`HF_HOME` 等环境变量
