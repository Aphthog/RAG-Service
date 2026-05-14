# RAG Service

基于 bge-m3 的混合检索 RAG 服务，支持稠密+稀疏向量混合检索、自适应分块、多租户隔离、可选的 HTTP 服务及前端界面。

## 使用场景

- **知识库问答** — 将企业文档、技术手册、产品说明等导入索引，基于语义搜索回答问题
- **行测/试题检索** — 对结构化题目数据（题干 + 选项 + 解析）建索引，按知识点或描述性查询检索
- **企业内部文档搜索** — 多租户隔离设计，不同团队/项目的文档独立索引，互不可见
- **RAG 原型开发** — 提供简洁的 Python API 和 REST API，快速集成到 LLM 应用

## 安装

### 依赖

- Python >= 3.11
- 推荐使用 GPU 加速（CUDA），特别是模型推理和 reranker 环节

### 从源码

```bash
git clone https://github.com/Aphthog/RAG-Service.git
cd rag-service
pip install -e .
```

### 带 HTTP 服务

```bash
pip install -e ".[server]"
```

依赖会自动从 `pyproject.toml` 安装，主要包含：

| 依赖 | 用途 |
|------|------|
| `sentence-transformers` | bge-m3 嵌入模型推理 |
| `FlagEmbedding` | bge-reranker-v2-m3 重排序模型 |
| `faiss-cpu` | 稠密向量索引（HNSW） |
| `scipy` | 稀疏向量索引（CSR） |
| `fastapi / uvicorn` | HTTP 服务（可选） |

### 模型下载

首次使用会自动从 HuggingFace 下载模型（约 2.2GB）。如需指定缓存路径：

```bash
set HF_HOME=E:/ai_models/huggingface
set TRANSFORMERS_CACHE=E:/ai_models/transformers
set SENTENCE_TRANSFORMERS_HOME=E:/ai_models/sentence-transformers
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
python server.py
# 浏览器打开 http://localhost:8000 使用前端界面
```

```bash
# 建索引
curl -X POST http://localhost:8000/tenants/my_kb/index \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world"]}'

# 搜索
curl -X POST http://localhost:8000/tenants/my_kb/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "top_k": 3}'
```

## 架构

```
输入文档 → 自适应分块 → 向量化(稠密+稀疏) → FAISS 建索引 →
用户 query → 向量化 → 混合检索 → [rerank] → 返回 top-k 结果
```

四大核心模块：

| 模块 | 说明 |
|------|------|
| **Chunker** | 自适应分块：Markdown 标题感知 + 语义相似度边界检测 |
| **Embedder** | bge-m3 模型，输出 1024 维稠密 + 65536 维稀疏向量 |
| **Indexer** | FAISS HNSW (稠密) + scipy CSR (稀疏)，多租户文件锁隔离 |
| **Retriever** | 混合检索（稠密 0.7 + 稀疏 0.3），可选 bge-reranker-v2-m3 精排 |

## 当前缺陷

- **分数归一化不稳定** — 检索器中的 min-max 归一化在同一批候选内做，多样本评分分布不均时可能导致排名失真（retriever.py 第 53-58 行）
- **缺少 query 理解/改写** — 虽预留了接口，但尚未实现查询改写、同义词扩展等
- **不支持文档更新** — 只能删除知识库后重建索引
- **不支持标量过滤** — 无法按元数据字段（如来源、日期）缩小搜索范围
- **模型冷启动** — 模型首次加载是懒加载，第一个请求会明显卡顿
- **全内存索引** — 所有索引加载到内存，无分片机制，大数据量下内存压力大
- **无监控/metrics** — 缺少请求延迟、索引大小、命中率等可观测性指标
- **HNSW 增量退化** — 增量追加超过总量 20% 后检索质量可能下降，建议全量重建

## 测试

```bash
pytest tests/ -v
```

## License

MIT
