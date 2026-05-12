"""Optional FastAPI HTTP shell for RAGService. pip install rag-service[server] to use."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "./data/rag_indexes")
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    from rag_service import RAGService
    rag_service = RAGService(index_dir=INDEX_DIR, enable_rerank=True)
    yield


app = FastAPI(title="RAG Service", version="1.0.0", lifespan=lifespan)


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
    return {"status": "ok"}


@app.post("/tenants/{name}/index")
async def index_tenant(name: str, req: IndexRequest):
    try:
        count = await rag_service.index(name, req.texts, req.metadatas)
        return {"tenant": name, "chunk_count": count}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tenants/{name}/search")
async def search_tenant(name: str, req: SearchRequest):
    try:
        results = await rag_service.search(req.query, tenant=name, top_k=req.top_k)
        return {"results": [r.__dict__ for r in results]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/tenants/{name}")
async def delete_tenant(name: str):
    try:
        await rag_service.delete_index(name)
        return {"deleted": name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tenants")
async def list_tenants():
    return {"tenants": rag_service.list_tenants()}


@app.get("/tenants/{name}/stats")
async def tenant_stats(name: str):
    try:
        return rag_service.index_stats(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
