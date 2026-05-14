import logging
import numpy as np
from rag_service.types import SearchResult

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
        rerank_top_k: int = 5,
        reranker_model: str | None = None,
    ):
        self._embedder = embedder
        self._indexer = indexer
        self._enable_rerank = enable_rerank
        self._recall_top_k = recall_top_k
        self._rerank_top_k = rerank_top_k
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

        n_queries = len(queries)
        n_chunks = self._indexer.chunk_count()
        logger.info("  ▶ 混合检索：%d 个查询 × %d 个候选块（稠密权重=%s, 稀疏权重=%s）",
                     n_queries, n_chunks, self.DENSE_WEIGHT, self.SPARSE_WEIGHT)

        all_candidates: dict[int, float] = {}
        for q in queries:
            emb = self._embedder.encode([q])
            k = min(self._recall_top_k, n_chunks)
            if k == 0:
                logger.info("  ⚠ 索引为空，无结果返回")
                return []

            dense_results = self._indexer.search_dense(emb.dense, k * 2)
            sparse_results = self._indexer.search_sparse(emb.sparse, k * 2)
            logger.info("    - 稠密召回 %d 个，稀疏召回 %d 个", len(dense_results), len(sparse_results))

            dense_scores = {cid: score for cid, score in dense_results}
            sparse_scores = {cid: score for cid, score in sparse_results}

            for scores in [dense_scores, sparse_scores]:
                if scores:
                    max_val = max(scores.values())
                    min_val = min(scores.values())
                    if max_val != min_val:
                        rng = max_val - min_val
                        for cid in scores:
                            scores[cid] = (scores[cid] - min_val) / rng
                    else:
                        # 单点（或全相等）：min-max 无跨度，归一化会得到 0；保留为 1.0 便于展示与融合
                        for cid in scores:
                            scores[cid] = 1.0

            all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
            for cid in all_ids:
                d_score = dense_scores.get(cid, 0.0)
                s_score = sparse_scores.get(cid, 0.0)
                combined = self.DENSE_WEIGHT * d_score + self.SPARSE_WEIGHT * s_score
                if cid not in all_candidates or combined > all_candidates[cid]:
                    all_candidates[cid] = combined

        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:self._recall_top_k]

        if self._enable_rerank and len(sorted_candidates) > top_k:
            rerank_input = sorted_candidates[:self._rerank_top_k]
            logger.info("  ▶ 重排序模型精排 %d 个候选...", len(rerank_input))
            reranked = self._rerank(query, rerank_input, top_k)
            sorted_candidates = reranked + sorted_candidates[len(rerank_input):]

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
        logger.info("  ✓ 最终返回 %d 个结果", len(results))
        return results

    def _rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        if not candidates:
            return candidates
        if self._reranker is None:
            self._load_reranker()

        chunks = self._indexer.get_chunks()
        pairs = [(query, chunks[cid]["text"]) for cid, _ in candidates]
        scores = self._reranker.compute_score(pairs, normalize=True)

        reranked = []
        for i, (cid, _) in enumerate(candidates):
            reranked.append((cid, float(scores[i])))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _load_reranker(self):
        logger.info("Loading reranker model (%s)...", self._reranker_model)
        from FlagEmbedding import FlagReranker
        device = getattr(self._embedder, '_device', None)
        self._reranker = FlagReranker(self._reranker_model, use_fp16=True, device=device)
        logger.info("Reranker model loaded.")
