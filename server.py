"""Optional FastAPI HTTP shell for RAGService. pip install rag-service[server] to use."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "./data/rag_indexes")
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    from rag_service import RAGService
    print(f"\n{'='*60}")
    print(f"🚀 RAG Service 启动中...")
    print(f"   索引目录: {INDEX_DIR}")
    print(f"{'='*60}")
    rag_service = RAGService(index_dir=INDEX_DIR, enable_rerank=False)
    tenants = rag_service.list_tenants()
    print(f"   已加载 {len(tenants)} 个租户: {tenants if tenants else '（无）'}")
    print(f"{'='*60}\n")
    yield
    print(f"\n🛑 RAG Service 关闭")


app = FastAPI(title="RAG Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print("  ✓ 健康检查 OK")
    return {"status": "ok"}


@app.post("/tenants/{name}/index")
async def index_tenant(name: str, req: IndexRequest):
    print(f"\n{'='*60}")
    print(f"📥 [HTTP] 收到索引请求: 租户={name}, 文档数={len(req.texts)}")
    print(f"{'='*60}")
    try:
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(
            None, rag_service.index_sync, name, req.texts, req.metadatas,
        )
        print(f"📥 [HTTP] 索引完成: {count} 个块")
        return {"tenant": name, "chunk_count": count}
    except asyncio.CancelledError:
        print(f"  ⚠ 请求被客户端取消")
        raise HTTPException(status_code=499, detail="Request cancelled by client")
    except Exception as e:
        print(f"  ❌ 索引失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tenants/{name}/search")
async def search_tenant(name: str, req: SearchRequest):
    print(f"\n{'='*60}")
    print(f"🔍 [HTTP] 检索请求: 租户={name}, top_k={req.top_k}")
    print(f"   查询: {req.query[:120]}{'...' if len(req.query) > 120 else ''}")
    print(f"{'='*60}")
    try:
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="query cannot be empty")
        results = await rag_service.search(req.query, tenant=name, top_k=req.top_k)
        print(f"🔍 [HTTP] 返回 {len(results)} 个结果")
        return {"results": [r.__dict__ for r in results]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"  ❌ 检索失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/tenants/{name}")
async def delete_tenant(name: str):
    print(f"\n🗑 [HTTP] 删除索引: 租户={name}")
    try:
        await rag_service.delete_index(name)
        print(f"  ✓ 已删除")
        return {"deleted": name}
    except Exception as e:
        print(f"  ⚠ 删除失败: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tenants")
async def list_tenants():
    tenants = rag_service.list_tenants()
    print(f"  📋 租户列表: {tenants if tenants else '（无）'}")
    return {"tenants": tenants}


@app.get("/tenants/{name}/stats")
async def tenant_stats(name: str):
    try:
        return rag_service.index_stats(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# Serve static frontend (mounted last so API routes match first)
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
