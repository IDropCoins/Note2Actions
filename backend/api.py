from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

from config import Settings
from extraction import extract_actions
from ingest.embedding import Embedder
from storage import ZvecStore


class HealthResponse(BaseModel):
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ExtractRequest(BaseModel):
    query: str
    top_k: int = 5


settings = Settings()
app = FastAPI(title="MinutesMind API")

embedder = Embedder(settings.embed_model)
store = ZvecStore(settings.zvec_db_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/search")
def search_notes(request: SearchRequest) -> Dict[str, Any]:
    query_vector = embedder.embed([request.query])[0]
    docs = store.search(query_vector, top_k=request.top_k)

    results: List[Dict[str, Any]] = []
    for doc in docs:
        fields = dict(doc.fields) if getattr(doc, "fields", None) else {}
        results.append(
            {
                "id": doc.id,
                "score": doc.score,
                "metadata": fields,
            }
        )

    return {
        "query": request.query,
        "results": results,
    }


@app.post("/extract")
def extract_notes(request: ExtractRequest) -> Dict[str, Any]:
    return extract_actions(query=request.query, top_k=request.top_k)
