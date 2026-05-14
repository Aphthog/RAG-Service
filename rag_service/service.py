import os
import shutil
import logging
from rag_service.chunker import AdaptiveChunker
from rag_service.embedder import Embedder
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
        embedder=None,
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

        self._embedder = embedder
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

        logger.info("")
        logger.info("=" * 50)
        logger.info("📥 开始索引：租户 '%s'，共 %d 篇文档", tenant, len(texts))
        logger.info("=" * 50)

        all_chunks = []
        all_chunk_dicts = []
        chunk_idx = 0
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            logger.info("  ▶  [%d/%d] 切分文档 (%.1fK 字符)...", i + 1, len(texts), len(text) / 1000)
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

        logger.info("  ✓ 切分完成：%d 篇 → %d 个文本块", len(texts), len(all_chunks))

        chunk_texts = [c.text for c in all_chunks]
        logger.info("  ▶ 编码 %d 个文本块（batch_size=%d）...", len(chunk_texts), self._embed_batch_size)
        embeddings = embedder.encode(chunk_texts, batch_size=self._embed_batch_size)
        logger.info(
            "Encoded %d chunks: dense=%s, sparse=%s",
            len(all_chunks), embeddings.dense.shape, embeddings.sparse.shape,
        )
        logger.info("  ✓ 编码完成：稠密向量 %s，稀疏向量 %s", str(embeddings.dense.shape), str(embeddings.sparse.shape))

        indexer = self._get_indexer(tenant)
        tenant_dir = os.path.join(self._index_dir, tenant)
        if os.path.exists(os.path.join(tenant_dir, "dense.faiss")):
            indexer.load()
            indexer.add(all_chunk_dicts, embeddings)
            logger.info("  ✓ 已添加到已有索引 '%s'，当前共 %d 个块", tenant, indexer.chunk_count())
        else:
            indexer.build(all_chunk_dicts, embeddings)
            logger.info("  ✓ 新建索引 '%s' 完成，共 %d 个块", tenant, len(all_chunks))

        total = indexer.chunk_count()
        logger.info("=" * 50)
        logger.info("✅ 索引完成！租户 '%s' 总计 %d 个文本块", tenant, total)
        logger.info("=" * 50)
        logger.info("")

        return total

    async def search(self, query: str, tenant: str, top_k: int = 5) -> list[SearchResult]:
        return self.search_sync(query, tenant, top_k)

    def search_sync(self, query: str, tenant: str, top_k: int = 5) -> list[SearchResult]:
        logger.info("")
        logger.info("=" * 50)
        logger.info("🔍 检索：租户 '%s'，top_k=%d", tenant, top_k)
        logger.info("   查询: %s%s", query[:120], "..." if len(query) > 120 else "")
        logger.info("=" * 50)

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
            rerank_top_k=self._rerank_top_k,
        )

        rewritten = None
        if self._query_rewriter:
            import asyncio
            if asyncio.iscoroutinefunction(self._query_rewriter):
                rewritten = asyncio.run(self._query_rewriter(query))
            else:
                rewritten = self._query_rewriter(query)

        results = retriever.retrieve_sync(query, top_k=top_k, rewritten_queries=rewritten)

        logger.info("  ✓ 召回 %d 个结果：", len(results))
        for i, r in enumerate(results):
            logger.info("    [%d] score=%.4f | %s...", i + 1, r.score, r.content[:80])
            if self._verbose:
                logger.debug("  score=%.4f chunk=%d content=%.80s", r.score, r.chunk_index, r.content)
        logger.info("=" * 50)
        logger.info("")

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
            if os.path.isdir(os.path.join(self._index_dir, d))
            and os.path.exists(os.path.join(self._index_dir, d, "dense.faiss"))
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
