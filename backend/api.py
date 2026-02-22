from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from config import Settings
from extraction import extract_actions
from ingest.embedding import Embedder
from storage import ZvecStore


class HealthResponse(BaseModel):
    status: str


class SearchFilters(BaseModel):
    source_file: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[SearchFilters] = None


class ExtractRequest(BaseModel):
    query: str
    top_k: int = 5


settings = Settings()
app = FastAPI(title="MinutesMind API")

embedder = Embedder(settings.embed_model)
store = ZvecStore(settings.zvec_db_path)


def _doc_to_result(doc: Any) -> Dict[str, Any]:
    fields = dict(doc.fields) if getattr(doc, "fields", None) else {}
    return {
        "id": doc.id,
        "score": doc.score,
        "metadata": fields,
    }


def _matches_search_filters(result: Dict[str, Any], filters: Optional[SearchFilters]) -> bool:
    if filters is None:
        return True

    metadata = result.get("metadata", {})
    meeting_date = metadata.get("meeting_date")
    source_file = metadata.get("source_file")

    if filters.source_file and source_file != filters.source_file:
        return False
    if filters.date_from and (not meeting_date or str(meeting_date) < filters.date_from):
        return False
    if filters.date_to and (not meeting_date or str(meeting_date) > filters.date_to):
        return False
    return True


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/search")
def search_notes(request: SearchRequest) -> Dict[str, Any]:
    query_vector = embedder.embed([request.query])[0]
    raw_docs = store.search(query_vector, top_k=max(request.top_k * 3, 1))

    raw_results = [_doc_to_result(doc) for doc in raw_docs]
    filtered_results = [
        result for result in raw_results if _matches_search_filters(result, request.filters)
    ]

    return {
        "query": request.query,
        "results": filtered_results[: request.top_k],
    }


@app.post("/extract")
def extract_notes(request: ExtractRequest) -> Dict[str, Any]:
    return extract_actions(query=request.query, top_k=request.top_k)
