# RAG Service

可复用的 Python RAG（检索增强生成）库，支持混合稠密+稀疏检索、自适应分块、多租户隔离。

## 功能特性

- **混合检索** — bge-m3 模型：1024 维稠密向量（FAISS HNSW）+ 65536 维稀疏向量（scipy CSR），融合权重 0.7/0.3
- **自适应分块** — Markdown 标题感知 + 语义相似度边界检测
- **重排序** — bge-reranker-v2-m3 对候选结果精排
- **多租户隔离** — 每个知识库独立索引目录，filelock 并发控制
- **可选 HTTP 服务** — 内置 FastAPI 服务器 + 前端演示页面

## 安装

```bash
pip install git+https://github.com/Aphthog/rag-service.git

# 带 HTTP 服务依赖：
pip install "rag-service[server] @ git+https://github.com/Aphthog/rag-service.git"
```

## 快速开始

```python
from rag_service import RAGService

rag = RAGService(index_dir="./data/rag_indexes")

# 建立索引（自动分块）
rag.index_sync(
    tenant="my_kb",
    texts=["Python 是一种高级编程语言...", "FAISS 是向量相似度搜索库..."],
    metadatas=[{"source": "wiki1.md"}, {"source": "wiki2.md"}],
)

# 搜索
results = rag.search_sync("向量检索", tenant="my_kb", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.content}")
```

## HTTP 服务

```bash
# 启动
python server.py
# 浏览器打开 http://localhost:8000 使用前端界面

# 或通过 API 调用：
# 建索引
curl -X POST http://localhost:8000/tenants/my_kb/index \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "foo bar"]}'

# 搜索
curl -X POST http://localhost:8000/tenants/my_kb/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "top_k": 3}'

# 列出知识库
curl http://localhost:8000/tenants

# 健康检查
curl http://localhost:8000/health
```

## 模型下载

首次使用会自动从 HuggingFace 下载模型。如需指定缓存路径，设置以下环境变量：

```bash
set HF_HOME=E:/ai_models/huggingface
set TRANSFORMERS_CACHE=E:/ai_models/transformers
set SENTENCE_TRANSFORMERS_HOME=E:/ai_models/sentence-transformers
```

## API

### RAGService

| 方法 | 说明 |
|------|------|
| `index_sync(tenant, texts, metadatas?)` | 索引文档（自动分块+向量化+建索引） |
| `search_sync(query, tenant, top_k=5)` | 搜索已索引的分块 |
| `delete_index_sync(tenant)` | 删除某个知识库 |
| `list_tenants()` | 列出全部已索引的知识库 |
| `index_stats(tenant)` | 获取知识库的分块数量和索引路径 |

### 配置参数

```python
rag = RAGService(
    index_dir="./data",         # 索引存储目录
    enable_rerank=True,         # 是否启用 reranker 精排
    chunk_max_chars=800,        # 单个分块最大字符数
    chunk_overlap=80,           # 分块之间重叠字符数
    recall_top_k=50,            # 混合检索召回候选数
    embedder=my_embedder,       # 注入自定义 embedder（测试用）
)
```
